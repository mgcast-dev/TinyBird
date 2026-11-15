"""Features v7, compartidas por entrenamiento, evaluación offline y serving.

Un único código construye las features en los tres sitios: así el modelo recibe en
producción exactamente lo que aprendió (sin train/serve skew).

Novedades frente a features.py:
  * contexto de la petición: device, page, region, hora, día de la semana, nº de vistas
  * tendencia por subcategoría (sub_lift): vistas de la última hora frente a lo esperado
    por las 24 h anteriores. En entrenamiento se calcula punto-en-el-tiempo con series
    horarias de history_events; en producción, con shop.events en vivo (misma fórmula)
  * features de CONTENIDO (precio vs subcategoría, conversión de subcategoría y marca)
    que permiten puntuar productos nuevos sin historial
  * posición (slot): se entrena con la posición real y se sirve siempre con slot=0,
    así el modelo separa "el producto es bueno" de "estaba arriba"
  * ranking por p(click) * p(compra | click)^g  (la métrica puntúa compras esperadas)
"""
from __future__ import annotations

import calendar
import json
import math
import time
from collections import defaultdict
from typing import Any

import numpy as np

VERSION = "v7-ctx-trend-buy"

FEATURE_NAMES = [
    # estadísticas del producto (se apagan en un producto frío)
    "log_purchases", "log_views", "conv_rate", "cart_rate", "is_cold_item",
    # contenido del producto (existe aunque el producto sea nuevo)
    "price", "price_vs_sub", "sub_conv", "brand_conv", "cat_code",
    # contexto de la petición
    "device_code", "page_code", "region_code", "n_views", "hour", "dow",
    # sesión x producto
    "same_sub", "same_cat", "same_brand", "price_ratio", "co_last", "co_total", "already_viewed",
    # señales en vivo y posición
    "sub_lift", "slot",
]
N_FEATURES = len(FEATURE_NAMES)
_F = {name: i for i, name in enumerate(FEATURE_NAMES)}
CATEGORICAL = ["cat_code", "device_code", "page_code", "region_code"]
CAT_IDX = [_F[n] for n in CATEGORICAL]
SLOT_IDX = _F["slot"]

NAN = float("nan")
DEVICES = {"web": 0.0, "ios": 1.0, "android": 2.0}
PAGES = {"home": 0.0, "product": 1.0, "category": 2.0}
PRIOR_CONV = 1.0 / 50.0      # (0 + 1) / (0 + 50): conversión de un producto sin datos
PRIOR_CART = 1.0 / 20.0


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
def epoch_of(ts: Any, default: float | None = None) -> float:
    """Segundos UTC de un ts ISO ('2026-11-14T16:02:11.482Z' o '2026-11-14 16:02:11.482000000')."""
    try:
        s = str(ts).strip().replace("T", " ")
        base = calendar.timegm(time.strptime(s[:19], "%Y-%m-%d %H:%M:%S"))
        frac = 0.0
        if len(s) > 20 and s[19] == ".":
            digits = ""
            for ch in s[20:]:
                if not ch.isdigit():
                    break
                digits += ch
            if digits:
                frac = float("0." + digits)
        return base + frac
    except Exception:
        return time.time() if default is None else default


def viewed_from_requests(recent_views: Any) -> list[str]:
    return [str(v["product_id"]) for v in (recent_views or [])
            if isinstance(v, dict) and "product_id" in v]


def _price(meta: dict) -> float:
    try:
        return float(meta.get("price_eur") or 25.0)
    except (TypeError, ValueError):
        return 25.0


def _lift(v1: float, expected: float) -> float:
    """log((vistas reales + 5) / (vistas esperadas + 5)), recortado a [-1.5, 1.5]."""
    return max(-1.5, min(1.5, math.log((v1 + 5.0) / (expected + 5.0))))


# ---------------------------------------------------------------------------
# estado: índices precalculados
# ---------------------------------------------------------------------------
def _totals(hv: dict) -> dict:
    tot: dict[int, float] = defaultdict(float)
    for series in hv.values():
        for h, v in series.items():
            tot[h] += v
    return dict(tot)


def hourly_from_rows(rows: list[dict]) -> dict[str, dict[int, float]]:
    """Filas SQL {sub, h (epoch s), v} -> {subcategoría: {hora_índice: vistas}}."""
    hv: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        hv[str(r["sub"])][int(float(r["h"]) // 3600)] = float(r["v"])
    return dict(hv)


def build_state(catalog: dict, prod_stats: dict, co_matrix: dict,
                extra: dict | None = None) -> dict[str, Any]:
    extra = extra or {}
    cat_index = extra.get("cat_index") or {
        c: float(i) for i, c in enumerate(sorted({str(m.get("category", "")) for m in catalog.values()}))}

    ranked = sorted((p for p in prod_stats if p in catalog),
                    key=lambda p: prod_stats[p]["purchases"], reverse=True)
    top_by_subcat: dict[str, list[str]] = defaultdict(list)
    top_by_cat: dict[str, list[str]] = defaultdict(list)
    for pid in ranked:
        m = catalog[pid]
        sc, ct = m.get("subcategory", ""), m.get("category", "")
        if sc and len(top_by_subcat[sc]) < 20:
            top_by_subcat[sc].append(pid)
        if ct and len(top_by_cat[ct]) < 12:
            top_by_cat[ct].append(pid)

    # agregados por subcategoría y marca: prior de contenido para productos sin historial
    sp, sv, bp, bv = (defaultdict(float) for _ in range(4))
    tp = tv = 0.0
    for pid, st in prod_stats.items():
        m = catalog.get(pid)
        if not m:
            continue
        p, v = st["purchases"], st["views"]
        sp[m.get("subcategory")] += p
        sv[m.get("subcategory")] += v
        bp[m.get("brand")] += p
        bv[m.get("brand")] += v
        tp += p
        tv += v
    glob = (tp + 1.0) / (tv + 50.0)
    sub_conv = {k: (sp[k] + glob * 50.0) / (sv[k] + 50.0) for k in sv}
    brand_conv = {k: (bp[k] + glob * 50.0) / (bv[k] + 50.0) for k in bv}

    prices: dict[str, list[float]] = defaultdict(list)
    for m in catalog.values():
        prices[m.get("subcategory")].append(_price(m))
    sub_price_mean = {k: sum(v) / len(v) for k, v in prices.items()}

    cold_by_subcat: dict[str, list[str]] = defaultdict(list)
    for pid, m in catalog.items():
        if pid not in prod_stats and m.get("subcategory"):
            cold_by_subcat[m["subcategory"]].append(pid)

    co_top = {v: [c for c, _ in sorted(recs.items(), key=lambda x: x[1], reverse=True)[:15]]
              for v, recs in co_matrix.items()}

    hv = extra.get("hv") or {}
    return {
        "catalog": catalog, "prod_stats": prod_stats, "co_matrix": co_matrix,
        "top_global": ranked, "top_by_subcat": dict(top_by_subcat), "top_by_cat": dict(top_by_cat),
        "cold_by_subcat": dict(cold_by_subcat), "new_products": [], "co_top": co_top,
        "sub_conv": sub_conv, "brand_conv": brand_conv, "glob_conv": glob,
        "sub_price_mean": sub_price_mean, "cat_index": cat_index,
        "region_index": extra.get("region_index") or {},
        "hv": hv, "hv_total": _totals(hv), "_lift_cache": {},
    }


def add_hourly(state: dict, rows: list[dict]) -> None:
    """Añade/sobrescribe horas en la serie de vistas (p. ej. el periodo de test del notebook)."""
    for sub, series in hourly_from_rows(rows).items():
        state["hv"].setdefault(sub, {}).update(series)
    state["hv_total"] = _totals(state["hv"])
    state["_lift_cache"].clear()


# ---------------------------------------------------------------------------
# tendencia por subcategoría
# ---------------------------------------------------------------------------
def lift_map_at(state: dict, t: float) -> dict[str, float]:
    """sub_lift de todas las subcategorías en el instante t, con las series horarias del estado
    (última hora completa frente a las 24 h anteriores). Es el mismo cálculo que compute_live_lift."""
    H = int(t // 3600)
    cache = state["_lift_cache"]
    if H in cache:
        return cache[H]
    hv, tot = state["hv"], state["hv_total"]
    out: dict[str, float] = {}
    if hv and tot:
        T1 = tot.get(H - 1, 0.0)
        T24 = sum(tot.get(h, 0.0) for h in range(H - 25, H - 1))
        if T1 > 0 and T24 > 0:
            for sub, series in hv.items():
                v1 = series.get(H - 1, 0.0)
                v24 = sum(series.get(h, 0.0) for h in range(H - 25, H - 1))
                out[sub] = _lift(v1, T1 * v24 / T24)
    cache[H] = out
    return out


def compute_live_lift(catalog: dict, rows: list[dict]) -> dict[str, float]:
    """rows: [{product_id, v1 (vistas última hora), v24 (vistas de las 24 h anteriores)}]."""
    v1s: dict[str, float] = defaultdict(float)
    v24s: dict[str, float] = defaultdict(float)
    T1 = T24 = 0.0
    for r in rows:
        v1, v24 = float(r["v1"]), float(r["v24"])
        T1 += v1
        T24 += v24
        sub = (catalog.get(str(r["product_id"])) or {}).get("subcategory")
        if sub:
            v1s[sub] += v1
            v24s[sub] += v24
    if T1 <= 0 or T24 <= 0:
        return {}
    return {sub: _lift(v1s[sub], T1 * v24s[sub] / T24) for sub in set(v1s) | set(v24s)}


def make_ctx(context: Any, ts: Any, state: dict | None = None,
             live_lift: dict | None = None) -> dict[str, Any]:
    c = context
    if isinstance(c, str):
        try:
            c = json.loads(c)
        except Exception:
            c = {}
    if not isinstance(c, dict):
        c = {}
    t = epoch_of(ts)
    if live_lift is not None:
        lift = live_lift
    elif state is not None:
        lift = lift_map_at(state, t)
    else:
        lift = {}
    return {"device": c.get("device"), "page": c.get("page"), "region": c.get("region"),
            "t": t, "lift": lift}


# ---------------------------------------------------------------------------
# candidatos
# ---------------------------------------------------------------------------
def get_candidates(state: dict, viewed: list[str], active, blocked=frozenset(),
                   lift: dict | None = None, max_cands: int = 130) -> list[str]:
    cands: list[str] = []
    seen: set[str] = set()
    catalog = state["catalog"]

    def add(pid: str) -> None:
        if pid in active and pid not in blocked and pid not in seen:
            seen.add(pid)
            cands.append(pid)

    for v in reversed(viewed):                       # lo visto, de más reciente a más antiguo
        for c in state["co_top"].get(v, [])[:12]:
            add(c)
        sub = (catalog.get(v) or {}).get("subcategory", "")
        for c in state["top_by_subcat"].get(sub, [])[:8]:
            add(c)
        for c in state["cold_by_subcat"].get(sub, [])[:3]:      # productos sin historial, misma subcategoría
            add(c)
        if len(cands) >= 40:
            break
    if lift:                                          # subcategorías en tendencia ahora mismo
        for sub, _ in sorted(lift.items(), key=lambda x: x[1], reverse=True)[:3]:
            for c in state["top_by_subcat"].get(sub, [])[:5]:
                add(c)
    for c in reversed(state["new_products"][-8:]):    # los productos añadidos durante el evento
        add(c)
    for c in state["top_global"][:30]:
        add(c)
    for items in state["top_by_cat"].values():        # diversidad: lo mejor de cada categoría
        for c in items[:8]:
            add(c)
    return cands[:max_cands]


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def build_matrix(state: dict, ctx: dict, viewed: list[str], cands: list[str],
                 slots=None, cold=None) -> np.ndarray:
    """Matriz (n_candidatos, N_FEATURES). `slots`: posición de cada candidato (serving: todo 0).
    `cold`: máscara booleana; en esas filas se ocultan las estadísticas del producto (dropout de
    entrenamiento que simula un producto nuevo)."""
    catalog, stats, co = state["catalog"], state["prod_stats"], state["co_matrix"]
    sub_conv, brand_conv, sub_price = state["sub_conv"], state["brand_conv"], state["sub_price_mean"]
    glob, cat_index = state["glob_conv"], state["cat_index"]
    lift = ctx.get("lift") or {}
    t = float(ctx.get("t") or time.time())
    dev = DEVICES.get(ctx.get("device"), NAN)
    page = PAGES.get(ctx.get("page"), NAN)
    reg = state["region_index"].get(ctx.get("region"), NAN)
    hour = float((t // 3600) % 24)
    dow = float((t // 86400 + 3) % 7)            # 1970-01-01 fue jueves
    nv = float(min(len(viewed), 10))

    sub_w: dict = defaultdict(float)
    cat_w: dict = defaultdict(float)
    brand_w: dict = defaultdict(float)
    prices: list[float] = []
    co_rows: list[tuple] = []
    for idx, v in enumerate(reversed(viewed)):   # idx 0 = vista más reciente
        meta = catalog.get(v)
        if not meta:
            continue
        w = 1.0 / (1.0 + 0.5 * idx)
        sub_w[meta.get("subcategory")] += w
        cat_w[meta.get("category")] += w
        brand_w[meta.get("brand")] += w
        prices.append(_price(meta))
        d = co.get(v)
        if d:
            co_rows.append((idx == 0, w, d))
    avg_price = sum(prices) / len(prices) if prices else None
    viewed_set = set(viewed)

    rows = []
    for i, c in enumerate(cands):
        meta = catalog.get(c) or {}
        sub, cat, brand = meta.get("subcategory"), meta.get("category"), meta.get("brand")
        price = _price(meta)
        st = stats.get(c)
        is_cold = st is None or (cold is not None and bool(cold[i]))
        if is_cold:
            lp = lv = 0.0
            conv, cart = PRIOR_CONV, PRIOR_CART
            co_last = co_total = 0.0
        else:
            lp, lv = math.log1p(st["purchases"]), math.log1p(st["views"])
            conv, cart = st["conv_rate"], st["cart_rate"]
            co_last = co_total = 0.0
            for first, w, d in co_rows:
                pair = d.get(c, 0.0)
                if pair:
                    co_total += w * pair
                    if first:
                        co_last = pair
        rows.append([
            lp, lv, conv, cart, 1.0 if is_cold else 0.0,
            price, price / max(sub_price.get(sub, price), 1.0),
            sub_conv.get(sub, glob), brand_conv.get(brand, glob), cat_index.get(cat, NAN),
            dev, page, reg, nv, hour, dow,
            sub_w.get(sub, 0.0) if sub is not None else 0.0,
            cat_w.get(cat, 0.0) if cat is not None else 0.0,
            brand_w.get(brand, 0.0) if brand is not None else 0.0,
            price / max(avg_price, 1.0) if avg_price is not None else 1.0,
            co_last, co_total, 1.0 if c in viewed_set else 0.0,
            lift.get(sub, 0.0), float(slots[i]) if slots is not None else 0.0,
        ])
    return np.asarray(rows, dtype=np.float32).reshape(len(cands), N_FEATURES)


# ---------------------------------------------------------------------------
# ranking (el mismo código en el notebook y en producción)
# ---------------------------------------------------------------------------
def score_candidates(models: dict, X: np.ndarray) -> np.ndarray:
    p = models["click"].predict_proba(X)[:, 1]
    buy, g = models.get("buy"), float(models.get("buy_exp", 1.0))
    if buy is not None and g > 0:
        p = p * np.power(buy.predict_proba(X)[:, 1], g)
    return p


def rank_request(state: dict, models: dict, ctx: dict, viewed: list[str], active,
                 blocked=frozenset(), max_cands: int = 130) -> list[str]:
    """Candidatos ordenados de mejor a peor (puede devolver menos de 10: complétalo con top10)."""
    cands = get_candidates(state, viewed, active, blocked, ctx.get("lift"), max_cands)
    if not cands:
        return []
    p = score_candidates(models, build_matrix(state, ctx, viewed, cands))
    return [cands[i] for i in np.argsort(-p, kind="stable")]


def top10(ranked, fallback=()) -> list[str]:
    out: list[str] = []
    used: set[str] = set()
    for pid in list(ranked) + list(fallback):
        if pid not in used:
            used.add(pid)
            out.append(pid)
            if len(out) == 10:
                break
    return out


# ---------------------------------------------------------------------------
# ejemplos de entrenamiento
# ---------------------------------------------------------------------------
def build_examples(state: dict, sessions: list[dict], rng, drop_p: float = 0.15) -> dict[str, np.ndarray]:
    """Una fila por producto mostrado. `sessions` debe venir ordenada por tiempo.
    Etiquetas: clic, compra. Pesos: IPS (corrige el sesgo del sistema antiguo)."""
    mats, yc, yb, w, clk, sidx = [], [], [], [], [], []
    for si, s in enumerate(sessions):
        shown = [str(x) for x in (s.get("shown") or [])]
        if not shown:
            continue
        viewed = viewed_from_requests(s.get("recent_views"))
        ctx = make_ctx(s.get("context"), s.get("ts"), state=state)
        cold = rng.random(len(shown)) < drop_p
        mats.append(build_matrix(state, ctx, viewed, shown, slots=list(range(len(shown))), cold=cold))
        props = s.get("propensity") or []
        clicked = s.get("clicked") or None
        purchased = s.get("purchased") or None
        for i, cand in enumerate(shown):
            p = float(props[i]) if i < len(props) and props[i] else 0.1
            w.append(min(1.0 / max(p, 0.02), 10.0))
            is_buy = cand == purchased
            is_click = is_buy or cand == clicked
            yc.append(1 if is_click else 0)
            yb.append(1 if is_buy else 0)
            clk.append(is_click)
            sidx.append(si)
    return {"X": np.vstack(mats), "y_click": np.asarray(yc, dtype=np.int32),
            "y_buy": np.asarray(yb, dtype=np.int32), "w": np.asarray(w, dtype=np.float32),
            "is_click_row": np.asarray(clk, dtype=bool), "sess": np.asarray(sidx, dtype=np.int64)}
