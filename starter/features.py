"""Features compartidas por entrenamiento, evaluación offline y serving.

Un único código para construir features evita el train/serve skew:
lo que aprende el modelo es exactamente lo que recibe en producción.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

N_FEATURES = 11
DEFAULT_STATS = {"views": 0.0, "carts": 0.0, "purchases": 0.0,
                 "conv_rate": 0.01, "cart_rate": 0.02}


def viewed_from_requests(recent_views: Any) -> list[str]:
    return [v["product_id"] for v in (recent_views or [])
            if isinstance(v, dict) and "product_id" in v]


def build_state(catalog: dict, prod_stats: dict, co_matrix: dict) -> dict[str, Any]:
    """Índices precalculados a partir de catálogo, estadísticas y co-visitas."""
    top_global = [pid for pid, _ in sorted(prod_stats.items(),
                                           key=lambda x: x[1]["purchases"], reverse=True)]
    top_by_subcat: dict[str, list[str]] = defaultdict(list)
    for pid in top_global:
        sc = catalog.get(pid, {}).get("subcategory", "")
        if sc and len(top_by_subcat[sc]) < 20:
            top_by_subcat[sc].append(pid)

    co_top = {
        v: [c for c, _ in sorted(recs.items(), key=lambda x: x[1], reverse=True)[:15]]
        for v, recs in co_matrix.items()
    }
    return {
        "catalog": catalog,
        "prod_stats": prod_stats,
        "co_matrix": co_matrix,
        "top_global": top_global,
        "top_by_subcat": dict(top_by_subcat),
        "co_top": co_top,
    }


def build_feature_row(state: dict, cand_id: str, viewed_ids: list[str]) -> list[float]:
    """11 features para un par (sesión, candidato). Sin posición ni señales en vivo."""
    catalog = state["catalog"]
    co = state["co_matrix"]
    c = catalog.get(cand_id, {})
    st = state["prod_stats"].get(cand_id, DEFAULT_STATS)
    c_price = float(c.get("price_eur", 25.0))
    c_cat = c.get("category", "")
    c_sub = c.get("subcategory", "")
    c_brand = c.get("brand", "")

    same_sub = same_cat = same_brand = 0.0
    co_total = co_last = 0.0
    prices: list[float] = []

    for idx, v_id in enumerate(reversed(viewed_ids)):   # idx 0 = vista más reciente
        v = catalog.get(v_id)
        if not v:
            continue
        w = 1.0 / (1.0 + 0.5 * idx)
        if v.get("subcategory") == c_sub:
            same_sub += w
        if v.get("category") == c_cat:
            same_cat += w
        if v.get("brand") == c_brand:
            same_brand += w
        pair = co.get(v_id, {}).get(cand_id, 0.0)
        co_total += w * pair
        if idx == 0:
            co_last = pair
        prices.append(float(v.get("price_eur", c_price)))

    avg_price = sum(prices) / len(prices) if prices else c_price
    return [
        float(st["purchases"]),
        float(st["conv_rate"]),
        float(st["cart_rate"]),
        c_price,
        c_price / max(avg_price, 1.0),
        same_sub,
        same_cat,
        same_brand,
        co_last,
        co_total,
        1.0 if cand_id in viewed_ids else 0.0,
    ]


def get_candidates(state: dict, viewed_ids: list[str], active: set[str],
                   max_cands: int = 35) -> list[str]:
    cands: list[str] = []
    seen: set[str] = set()

    def add(pid: str) -> None:
        if pid in active and pid not in seen:
            seen.add(pid)
            cands.append(pid)

    for v_id in reversed(viewed_ids):
        for cid in state["co_top"].get(v_id, [])[:10]:
            add(cid)
        sc = state["catalog"].get(v_id, {}).get("subcategory", "")
        for cid in state["top_by_subcat"].get(sc, [])[:8]:
            add(cid)
        if len(cands) >= 25:
            break

    for cid in state["top_global"]:
        add(cid)
        if len(cands) >= max_cands:
            break
    return cands