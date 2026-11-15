"""Recomendador LightGBM v8 con perfiles de usuario y refresco en segundo plano."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
for env_file in (ROOT / ".env", Path(__file__).resolve().parent / ".env", Path.cwd() / ".env"):
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com").rstrip("/")
API_KEY = os.environ.get("RAWTREE_API_KEY", "")
TEAM = os.environ.get("TEAM_ID", "team_31")
if not re.fullmatch(r"[A-Za-z0-9_]+", TEAM):
    raise RuntimeError("TEAM_ID solo puede contener letras, números y guion bajo")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
http_fast, http_bg = requests.Session(), requests.Session()
http_fast.headers.update(HEADERS)
http_bg.headers.update(HEADERS)
Record = dict[str, Any]
MODELS = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS / "modelo_lgbm8.pkl" if (MODELS / "modelo_lgbm8.pkl").exists() else ROOT / "modelo_lgbm8.pkl"
DATA_PATH = MODELS / "features_data4.pkl" if (MODELS / "features_data4.pkl").exists() else ROOT / "features_data4.pkl"
FEATURE_NAMES = ["purchases", "conv_rate", "cart_rate", "price", "price_ratio", "same_subcat",
                 "same_cat", "same_brand", "co_score_last", "co_score_total", "already_viewed",
                 "slot", "purchases_24h", "views_24h", "carts_24h", "product_age_days",
                 "recent_popularity", "user_subcat_share", "user_brand_share", "user_price_ratio",
                 "user_past_item_status", "ctx_subcat_affinity"]

try:
    LGB_MODEL = joblib.load(MODEL_PATH)
    FEAT_DATA = joblib.load(DATA_PATH)
    CATALOG: dict[str, Record] = FEAT_DATA["catalog"]
    PROD_STATS: dict[str, dict[str, float]] = FEAT_DATA["prod_stats"]
    CO_MATRIX: dict[str, dict[str, float]] = FEAT_DATA["co_matrix"]
    USER_PROFILES: dict[str, Record] = FEAT_DATA.get("user_profiles", {})
    CONTEXT_PRICES: dict[str, list[float]] = FEAT_DATA.get("context_prices", {})
    CONTEXT_SUBCATS: dict[str, dict[str, float]] = FEAT_DATA.get("context_subcats", {})
    CONTEXT_TOTALS: dict[str, float] = FEAT_DATA.get("context_totals", {})
    if FEAT_DATA.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError("El orden de features del artefacto no coincide con v8")
except Exception as exc:
    print(f"[AVISO CRÍTICO] No se pudieron cargar los artefactos: {exc}")
    LGB_MODEL, FEAT_DATA = None, {}
    CATALOG, PROD_STATS, CO_MATRIX, USER_PROFILES = {}, {}, {}, {}
    CONTEXT_PRICES, CONTEXT_SUBCATS, CONTEXT_TOTALS = {}, {}, {}

DEFAULT_STATS = {"views": 0., "carts": 0., "purchases": 0., "base_purchases": 0.,
                 "conv_rate": .01, "cart_rate": .02, "base_conv": .01, "base_cart": .02,
                 "views_24h": 0., "carts_24h": 0., "purchases_24h": 0., "recent_popularity": 0.,
                 "product_age_days": 0.}
TOP_GLOBAL_HIST = sorted(PROD_STATS, key=lambda pid: PROD_STATS[pid].get("purchases", 0.), reverse=True)
TOP_BY_SUBCAT: dict[str, list[str]] = defaultdict(list)
for product_id in TOP_GLOBAL_HIST:
    subcat = CATALOG.get(product_id, {}).get("subcategory", "")
    if subcat and len(TOP_BY_SUBCAT[subcat]) < 30:
        TOP_BY_SUBCAT[subcat].append(product_id)
CO_TOP_LIST = {pid: [candidate for candidate, _ in sorted(rows.items(), key=lambda item: item[1], reverse=True)[:15]]
               for pid, rows in CO_MATRIX.items()}
TOP_TRENDING_LIVE: list[str] = []
ACTIVE_IDS: set[str] = {pid for pid, product in CATALOG.items() if product.get("is_active", True)}
UNAVAILABLE_SET: set[str] = {pid for pid, product in CATALOG.items() if not product.get("is_active", True)}
DATA_LOCK = threading.RLock()
LAST_LIVE_EVENT_TS: str | None = None

UNAVAILABLE_SQL = """
SELECT product_id::String AS product_id
FROM shop.catalog
WHERE toString(is_active) IN ('false', '0')
LIMIT 5000
"""
LIVE_STATS_SQL = """
SELECT product_id::String AS product_id,
 countIf(event_type='view') AS views_live,
 countIf(event_type='add_to_cart') AS carts_live,
 countIf(event_type='purchase') AS purchases_live
FROM shop.events
WHERE ts >= now() - INTERVAL 1 HOUR
GROUP BY product_id
LIMIT 5000
"""
NEW_REQUESTS_SQL = """
SELECT request_id::String AS request_id, user_id::String AS user_id,
       context, recent_views
FROM shop.requests
WHERE ts >= now() - INTERVAL 6 SECOND
ORDER BY ts DESC
LIMIT 100
"""
FEEDBACK_SQL = f"""
SELECT
    count() AS requests_2m,
    countIf(served_by = 'team') AS served_by_team,
    countIf(served_by = 'fallback') AS served_by_fallback,
    countIf(served_by = 'fallback' AND latency_ms IS NULL) AS fb_no_sub,
    countIf(served_by = 'fallback' AND latency_ms > 3000) AS fb_late,
    countIf(served_by = 'fallback' AND latency_ms <= 3000) AS fb_rejected,
    countIf(clicked != '') AS clicks,
    countIf(purchased != '') AS purchases,
    round(avgIf(latency_ms, served_by = 'team'), 1) AS avg_latency_ms,
    round(maxIf(latency_ms, served_by = 'team'), 1) AS max_latency_ms,
    round(sumIf(revenue_eur, served_by = 'team'), 2) AS revenue_eur
FROM {TEAM}.feedback
WHERE ts >= now() - INTERVAL 2 MINUTE
  AND request_id != '__init__'
"""


def _context(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return value if isinstance(value, list) else []


def _ctx_keys(context: dict[str, Any]) -> list[str]:
    region, device = str(context.get("region", "")), str(context.get("device", ""))
    return [f"{region}|{device}", f"{region}|", f"|{device}"]


def _default_profile() -> Record:
    return {"subcat_weights": {}, "brand_weights": {}, "price_sum": 0., "price_weight": 0., "items": {}, "event_count": 0}


def _profile(user_id: str) -> Record:
    return USER_PROFILES.get(str(user_id or ""), _default_profile())


def _reference_price(profile: Record, context: dict[str, Any]) -> float:
    if profile.get("price_weight", 0.) > 0:
        return float(profile["price_sum"]) / float(profile["price_weight"])
    for key in _ctx_keys(context):
        values = CONTEXT_PRICES.get(key)
        if values and len(values) >= 2 and values[1]:
            return float(values[0]) / float(values[1])
    return 25.


def _item_status(profile: Record, product_id: str) -> float:
    state = profile.get("items", {}).get(product_id, {})
    if state.get("purchase"):
        return -1.
    if state.get("cart"):
        return 2.
    return 1. if state.get("view") else 0.


def build_feature_row(cand_id: str, viewed_ids: list[str], user_id: str = "",
                      context: dict[str, Any] | None = None) -> list[float]:
    """Produce las 22 features en el orden entrenado; no hace consultas de red."""
    context = context or {}
    with DATA_LOCK:
        product = CATALOG.get(cand_id, {})
        st = PROD_STATS.get(cand_id, DEFAULT_STATS)
        profile = _profile(user_id)
        price = float(product.get("price_eur") or 25.)
        category, subcat, brand = (str(product.get(k) or "") for k in ("category", "subcategory", "brand"))
        same_subcat = same_cat = same_brand = co_last = co_total = 0.
        viewed_prices = []
        for idx, viewed in enumerate(reversed(viewed_ids)):
            info = CATALOG.get(viewed)
            if not info:
                continue
            weight = 1. / (1. + .5 * idx)
            same_subcat += weight if info.get("subcategory") == subcat else 0.
            same_cat += weight if info.get("category") == category else 0.
            same_brand += weight if info.get("brand") == brand else 0.
            score = CO_MATRIX.get(viewed, {}).get(cand_id, 0.)
            co_total += weight * score
            if idx == 0:
                co_last = score
            viewed_prices.append(float(info.get("price_eur") or price))
        reference = _reference_price(profile, context)
        price_ratio = price / max(sum(viewed_prices) / len(viewed_prices) if viewed_prices else reference, 1.)
        subcat_weights, brand_weights = profile.get("subcat_weights", {}), profile.get("brand_weights", {})
        subcat_total, brand_total = sum(subcat_weights.values()), sum(brand_weights.values())
        user_subcat = subcat_weights.get(subcat, 0.) / subcat_total if subcat_total else 0.
        user_brand = brand_weights.get(brand, 0.) / brand_total if brand_total else 0.
        user_price_ratio = price / max(reference, 1.)
        ctx_affinity = 0.
        for key in _ctx_keys(context):
            total = CONTEXT_TOTALS.get(key, 0.)
            if total:
                ctx_affinity = min(1., CONTEXT_SUBCATS.get(key, {}).get(subcat, 0.) / total)
                break
        return [float(st.get("purchases", 0.)), float(st.get("conv_rate", .01)),
                float(st.get("cart_rate", .02)), price, price_ratio, same_subcat, same_cat,
                same_brand, co_last, co_total, float(cand_id in viewed_ids), 1.,
                float(st.get("purchases_24h", 0.)), float(st.get("views_24h", 0.)),
                float(st.get("carts_24h", 0.)), float(st.get("product_age_days", 0.)),
                float(st.get("recent_popularity", 0.)), user_subcat, user_brand,
                user_price_ratio, _item_status(profile, cand_id), ctx_affinity]


def get_candidates(viewed_ids: list[str], active: set[str], user_id: str = "",
                   context: dict[str, Any] | None = None) -> list[str]:
    """Cubre retargeting, perfil histórico y cold start contextual hasta 80 IDs."""
    context = context or {}
    candidates: list[str] = []
    seen: set[str] = set()

    def add(pid: str) -> None:
        pid = str(pid)
        if pid in active and pid not in seen and len(candidates) < 80:
            seen.add(pid)
            candidates.append(pid)

    with DATA_LOCK:
        profile = _profile(user_id)
        has_profile = profile.get("event_count", 0) or profile.get("items")
        if viewed_ids:
            for pid in reversed(viewed_ids[-2:]):
                add(pid)
            for viewed in reversed(viewed_ids):
                for pid in CO_TOP_LIST.get(viewed, [])[:10]:
                    add(pid)
                subcat = CATALOG.get(viewed, {}).get("subcategory", "")
                for pid in TOP_BY_SUBCAT.get(subcat, [])[:8]:
                    add(pid)
                if len(candidates) >= 24:
                    break
        if has_profile:
            abandoned = [pid for pid, state in profile.get("items", {}).items()
                         if state.get("cart") and not state.get("purchase")]
            abandoned.sort(key=lambda pid: PROD_STATS.get(pid, {}).get("purchases", 0.), reverse=True)
            for pid in abandoned[:12]:
                add(pid)
            favorites = sorted(profile.get("subcat_weights", {}).items(), key=lambda item: item[1], reverse=True)[:3]
            for subcat, _ in favorites:
                for pid in TOP_BY_SUBCAT.get(subcat, [])[:10]:
                    add(pid)
        else:
            contextual: dict[str, float] = defaultdict(float)
            for key in _ctx_keys(context):
                for subcat, score in CONTEXT_SUBCATS.get(key, {}).items():
                    contextual[subcat] += float(score)
                if contextual:
                    break
            for subcat, _ in sorted(contextual.items(), key=lambda item: item[1], reverse=True)[:5]:
                for pid in TOP_BY_SUBCAT.get(subcat, [])[:8]:
                    add(pid)
            for pid in TOP_TRENDING_LIVE[:5]:
                add(pid)
        # Asegura que también usuarios con vistas reciben tendencia fresca.
        for pid in TOP_TRENDING_LIVE[:5]:
            add(pid)
        for pid in TOP_GLOBAL_HIST:
            add(pid)
            if len(candidates) >= 80:
                break
    return candidates


def best_sellers(until: str | None = None) -> list[str]:
    return TOP_GLOBAL_HIST if TOP_GLOBAL_HIST else list(CATALOG)


def recommend(recent_views: list[Record], best: list[str] | None = None,
              unavailable: set[str] = frozenset(), user_id: str = "",
              context: dict[str, Any] | None = None) -> list[str]:
    viewed = [str(v["product_id"]) for v in (recent_views or [])
              if isinstance(v, dict) and v.get("product_id")]
    active = set(ACTIVE_IDS) - UNAVAILABLE_SET - set(unavailable)
    fallback = [pid for pid in (best if best is not None else TOP_GLOBAL_HIST)
                if pid in active]
    if LGB_MODEL is None:
        return fallback[:10]
    candidates = get_candidates(viewed, active, user_id, context)
    if not candidates:
        return fallback[:10]
    
    matrix = np.asarray(
        [build_feature_row(pid, viewed, user_id, context) for pid in candidates],
        dtype=np.float32,
    )
    scores = LGB_MODEL.booster_.predict(matrix)
    ranked = [candidates[i] for i in np.argsort(scores)[::-1] if candidates[i] in active][:10]
    for pid in fallback:
        if len(ranked) >= 10:
            break
        if pid not in ranked:
            ranked.append(pid)
    return ranked[:10]


def predict_batch(requests_batch: list[Record]) -> list[Record]:
    active = set(ACTIVE_IDS) - UNAVAILABLE_SET
    if LGB_MODEL is None:
        available_best = [pid for pid in best_sellers() if pid in active]
        return [{"request_id": req.get("request_id", ""), "items": available_best[:10], "model_version": "v8-tlitsg-optimus-prime+"}
                for req in requests_batch]
    rows: list[list[float]] = []
    records: list[tuple[str, list[str], int, int]] = []
    for req in requests_batch:
        views = _list(req.get("recent_views"))
        viewed = [str(v["product_id"]) for v in views if isinstance(v, dict) and v.get("product_id")]
        context = _context(req.get("context"))
        uid = str(req.get("user_id") or "")
        cands = get_candidates(viewed, active, uid, context)
        start = len(rows)
        rows.extend(build_feature_row(pid, viewed, uid, context) for pid in cands)
        records.append((str(req.get("request_id", "")), cands, start, len(rows)))
    if not rows:
        return []

    matrix = np.asarray(rows, dtype=np.float32)
    scores = LGB_MODEL.booster_.predict(matrix)
    result = []
    for request_id, candidates, start, end in records:
        if not candidates:
            items = [pid for pid in TOP_GLOBAL_HIST if pid in active][:10]
        else:
            order = np.argsort(scores[start:end])[::-1][:10]
            items = [candidates[i] for i in order]
        result.append({"request_id": request_id, "items": items, "model_version": "v8-tlitsg-optimus-prime+"})
    return result


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def _apply_live_event(row: Record) -> None:
    uid, pid, kind = str(row.get("user_id") or ""), str(row.get("product_id") or ""), str(row.get("event_type") or "view")
    if not uid or not pid:
        return
    event_count = max(1, int(row.get("event_count", 1) or 1))
    weight = {"view": 1., "add_to_cart": 2., "purchase": 5.}.get(kind, 1.) * event_count
    info = CATALOG.get(pid, {})
    price = float(row.get("price_eur") or info.get("price_eur") or 0.)
    subcat, brand = str(info.get("subcategory") or ""), str(info.get("brand") or "")
    profile = USER_PROFILES.setdefault(uid, _default_profile())
    if subcat:
        profile["subcat_weights"][subcat] = profile["subcat_weights"].get(subcat, 0.) + weight
    if brand:
        profile["brand_weights"][brand] = profile["brand_weights"].get(brand, 0.) + weight
    if price > 0:
        profile["price_sum"] += price * weight
        profile["price_weight"] += weight
    profile["event_count"] = profile.get("event_count", 0) + event_count
    state = profile["items"].setdefault(pid, {"view": False, "cart": False, "purchase": False})
    state_key = {"view": "view", "add_to_cart": "cart", "purchase": "purchase"}.get(kind)
    if state_key:
        state[state_key] = True
    context = {"region": row.get("region", ""), "device": row.get("device", "")}
    for key in _ctx_keys(context):
        if price > 0:
            values = CONTEXT_PRICES.setdefault(key, [0., 0.])
            values[0] += price * event_count
            values[1] += event_count
        if subcat:
            subcats = CONTEXT_SUBCATS.setdefault(key, {})
            subcats[subcat] = subcats.get(subcat, 0.) + weight
        CONTEXT_TOTALS[key] = CONTEXT_TOTALS.get(key, 0.) + weight


def bg_worker() -> None:
    global ACTIVE_IDS, UNAVAILABLE_SET, TOP_TRENDING_LIVE, LAST_LIVE_EVENT_TS
    last_live = 0.
    last_feedback = 0.
    feedback_first_read = False
    while True:
        try:
            response = http_bg.post(
                f"{API}/v1/query",
                json={"sql": UNAVAILABLE_SQL},
                timeout=(0.5, 3.0),
            )
            response.raise_for_status()
            unavailable_rows = response.json().get("data", [])
            with DATA_LOCK:
                UNAVAILABLE_SET = {str(row["product_id"]) for row in unavailable_rows if row.get("product_id")}
                ACTIVE_IDS = set(CATALOG) - UNAVAILABLE_SET

            if time.time() - last_live >= 30.:
                last_live = time.time()
                live_response = http_bg.post(
                    f"{API}/v1/query",
                    json={"sql": LIVE_STATS_SQL},
                    timeout=(0.5, 3.0),
                )
                live_response.raise_for_status()
                live_rows = live_response.json().get("data", [])
                trend: list[tuple[float, str]] = []
                with DATA_LOCK:
                    # Reinicia la foto anterior para que los productos que salen
                    # de la ventana móvil de una hora no conserven cifras obsoletas.
                    for st in PROD_STATS.values():
                        st.setdefault("base_purchases", st.get("purchases", 0.))
                        st.setdefault("base_conv", st.get("conv_rate", .01))
                        st.setdefault("base_cart", st.get("cart_rate", .02))
                        st["purchases"] = st["base_purchases"]
                        st["conv_rate"] = st["base_conv"]
                        st["cart_rate"] = st["base_cart"]
                        st["views_24h"] = st["carts_24h"] = st["purchases_24h"] = 0.
                        st["recent_popularity"] = 0.
                    for row in live_rows:
                        pid = str(row["product_id"])
                        views = float(row.get("views_live", 0) or 0)
                        carts = float(row.get("carts_live", 0) or 0)
                        purchases = float(row.get("purchases_live", 0) or 0)
                        popularity = purchases * 10. + carts * 3. + views
                        st = PROD_STATS.setdefault(pid, dict(DEFAULT_STATS))
                        st.setdefault("base_purchases", st.get("purchases", 0.))
                        st.setdefault("base_conv", st.get("conv_rate", .01))
                        st.setdefault("base_cart", st.get("cart_rate", .02))
                        st["purchases"] = st["base_purchases"] + purchases
                        live_conv = (purchases + 1.) / (views + 20.)
                        live_cart = (carts + 1.) / (views + 15.)
                        st["conv_rate"] = .7 * st["base_conv"] + .3 * live_conv
                        st["cart_rate"] = .7 * st["base_cart"] + .3 * live_cart
                        st["views_24h"], st["carts_24h"], st["purchases_24h"] = views, carts, purchases
                        st["recent_popularity"] = popularity
                        trend.append((popularity, pid))
                    trend.sort(reverse=True)
                    TOP_TRENDING_LIVE = [pid for _, pid in trend[:5]]

                # Perfil incremental acotado: solo eventos recientes, agregados en
                # RawTree y con límite explícito en la respuesta.
                cursor = _parse_time(LAST_LIVE_EVENT_TS)
                cursor_clause = ""
                if cursor:
                    cursor_value = cursor.strftime("%Y-%m-%d %H:%M:%S.%f")
                    cursor_clause = f"AND ts > toDateTime64('{cursor_value}', 3, 'UTC')"
                event_sql = f"""SELECT max(ts::DateTime64(3)) AS last_ts,
 user_id::String AS user_id, product_id::String AS product_id,
 event_type::String AS event_type, device::String AS device, region::String AS region,
 argMax(price_eur::Float64, ts::DateTime64(3)) AS price_eur, count() AS event_count
FROM shop.events
WHERE ts >= now() - INTERVAL 1 HOUR {cursor_clause}
GROUP BY user_id, product_id, event_type, device, region
ORDER BY last_ts ASC
LIMIT 5000"""
                events_response = http_bg.post(
                    f"{API}/v1/query",
                    json={"sql": event_sql},
                    timeout=(0.5, 3.0),
                )
                events_response.raise_for_status()
                live_events = events_response.json().get("data", [])
                for event in live_events:
                    with DATA_LOCK:
                        _apply_live_event(event)
                        LAST_LIVE_EVENT_TS = str(event.get("last_ts") or LAST_LIVE_EVENT_TS or "")

            # El feedback aparece después de la primera submission válida. Una
            # tabla aún inexistente o una respuesta de error no detiene el worker.
            if time.time() - last_feedback >= 30.:
                last_feedback = time.time()
                try:
                    feedback_response = http_bg.post(
                        f"{API}/v1/query",
                        json={"sql": FEEDBACK_SQL},
                        timeout=(0.5, 2.0),
                    )
                    feedback_response.raise_for_status()
                    rows = feedback_response.json().get("data", [])
                    stats = rows[0] if rows else {}
                    if not feedback_first_read:
                        print("[FEEDBACK 2m] Lectura de feedback correcta; monitor conectado.", flush=True)
                        feedback_first_read = True
                    print(
                        "[FEEDBACK 2m] "
                        f"requests={stats.get('requests_2m', 0)} "
                        f"team={stats.get('served_by_team', 0)} "
                        f"fallback={stats.get('served_by_fallback', 0)} "
                        f"fb_no_sub={stats.get('fb_no_sub', 0)} "
                        f"fb_late={stats.get('fb_late', 0)} "
                        f"fb_rejected={stats.get('fb_rejected', 0)} "
                        f"clicks={stats.get('clicks', 0)} "
                        f"purchases={stats.get('purchases', 0)} "
                        f"avg_latency_ms={stats.get('avg_latency_ms', 0)} "
                        f"max_latency_ms={stats.get('max_latency_ms', 0)} "
                        f"revenue_eur={stats.get('revenue_eur', 0)}",
                        flush=True,
                    )
                except requests.exceptions.HTTPError as feedback_error:
                    status = feedback_error.response.status_code if feedback_error.response is not None else "?"
                    body = feedback_error.response.text[:240] if feedback_error.response is not None else str(feedback_error)
                    print(
                        f"[FEEDBACK 2m] Tabla todavía no disponible o consulta rechazada "
                        f"(HTTP {status}): {body}",
                        flush=True,
                    )
                except (requests.exceptions.RequestException, ValueError, KeyError) as feedback_error:
                    print(f"[FEEDBACK 2m] Lectura temporalmente fallida: {feedback_error}", flush=True)
        except requests.exceptions.HTTPError as err:
            print(f"[BG aviso] HTTP 400 en SQL: {err.response.text if hasattr(err, 'response') else err}")
        except Exception as exc:
            print(f"[BG aviso] {exc}")
        time.sleep(2.)


def refresh_unavailable() -> None:
    """Carga la lista de stock agotado antes de empezar a aceptar peticiones."""
    global ACTIVE_IDS, UNAVAILABLE_SET
    response = http_fast.post(
        f"{API}/v1/query",
        json={"sql": UNAVAILABLE_SQL},
        timeout=(1.0, 2.0),
    )
    response.raise_for_status()
    rows = response.json().get("data", [])
    with DATA_LOCK:
        UNAVAILABLE_SET = {str(row["product_id"]) for row in rows if row.get("product_id")}
        ACTIVE_IDS = set(CATALOG) - UNAVAILABLE_SET
    print(f"[STOCK] Inicializado: {len(UNAVAILABLE_SET)} productos inactivos.", flush=True)


def main() -> None:
    # No arranca el polling hasta conocer el inventario inactivo actual.
    refresh_unavailable()
    threading.Thread(target=bg_worker, daemon=True, name="tinybird-bg-worker").start()
    SEEN_REQUESTS: set[str] = set()
    print("Recomendador v8-lgbm22-live activo")
    while True:
        had_new_requests = False
        try:
            if len(SEEN_REQUESTS) > 100_000:
                SEEN_REQUESTS.clear()
            response = http_fast.post(
                f"{API}/v1/query",
                json={"sql": NEW_REQUESTS_SQL},
                timeout=(1.0, 2.0),
            )
            response.raise_for_status()
            request_rows = response.json().get("data", [])
            fresh: list[Record] = []
            for request in request_rows:
                request_id = str(request.get("request_id", ""))
                if request_id and request_id not in SEEN_REQUESTS:
                    fresh.append(request)
            if fresh:
                had_new_requests = True
                answers = predict_batch(fresh)
                if answers:
                    http_fast.post(
                        f"{API}/v1/tables/submissions",
                        params={"database": TEAM},
                        json=answers,
                        timeout=(1.0, 2.0),
                    ).raise_for_status()
                    SEEN_REQUESTS.update(str(answer["request_id"]) for answer in answers)
                    print(f"answered {len(answers)} (v8-lgbm22-live)")
        except Exception as exc:
            print(f"error: {exc}")
        time.sleep(0.02 if had_new_requests else 0.15)


if __name__ == "__main__":
    main()
