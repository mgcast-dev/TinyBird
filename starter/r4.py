"""Recomendador en Tiempo Real con LightGBM (17 features + Refresco en Vivo de Estadísticas)."""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import requests

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ.get("TEAM_ID", "team_31")
HEADERS = {"Authorization": f"Bearer {os.environ.get('RAWTREE_API_KEY', '')}"}

http_fast = requests.Session()
http_fast.headers.update(HEADERS)
http_bg = requests.Session()
http_bg.headers.update(HEADERS)

Record = dict[str, Any]

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "modelo_lgbm8.pkl" if (MODELS_DIR / "modelo_lgbm8.pkl").exists() else Path("modelo_lgbm8.pkl")
FEAT_PATH = MODELS_DIR / "features_data8.pkl" if (MODELS_DIR / "features_data8.pkl").exists() else Path("features_data8.pkl")

print(f"Cargando modelo LightGBM (17 features) desde {MODEL_PATH}...")
try:
    LGB_MODEL = joblib.load(MODEL_PATH)
    FEAT_DATA = joblib.load(FEAT_PATH)
    CATALOG: dict[str, Record] = FEAT_DATA["catalog"]
    PROD_STATS: dict[str, dict[str, float]] = FEAT_DATA["prod_stats"]
    CO_MATRIX: dict[str, dict[str, float]] = FEAT_DATA["co_matrix"]
    print("¡Modelo de 17 features cargado correctamente!")
except Exception as e:
    print(f"[AVISO CRÍTICO]: No se encontraron los archivos .pkl ({e}).")
    LGB_MODEL = None
    FEAT_DATA, CATALOG, PROD_STATS, CO_MATRIX = {}, {}, {}, {}

TOP_GLOBAL_HIST = [
    pid for pid, _ in sorted(PROD_STATS.items(), key=lambda x: x[1].get("purchases", 0.0), reverse=True)
] if PROD_STATS else []

TOP_BY_SUBCAT: dict[str, list[str]] = defaultdict(list)
for pid in TOP_GLOBAL_HIST:
    sc = CATALOG.get(pid, {}).get("subcategory", "")
    if sc and len(TOP_BY_SUBCAT[sc]) < 20:
        TOP_BY_SUBCAT[sc].append(pid)

CO_TOP_LIST: dict[str, list[str]] = {}
for v_id, recs in CO_MATRIX.items():
    CO_TOP_LIST[v_id] = [
        cid for cid, _ in sorted(recs.items(), key=lambda x: x[1], reverse=True)[:15]
    ]

TOP_TRENDING_LIVE: list[str] = []

ACTIVE_IDS: set[str] = {pid for pid, c in CATALOG.items() if c.get("is_active", True)}

CATALOG_ACTIVE_SQL = """
SELECT product_id,
       argMax(category, v)    AS category,
       argMax(subcategory, v) AS subcategory,
       argMax(brand, v)       AS brand,
       argMax(price_eur, v)   AS price_eur,
       argMax(is_active, v)   AS is_active
FROM (
    SELECT product_id::String AS product_id, category::String AS category,
           subcategory::String AS subcategory, brand::String AS brand,
           price_eur::Float64 AS price_eur, is_active::Bool AS is_active, '' AS v
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String, category::String, subcategory::String, brand::String,
           price_eur::Float64, is_active::Bool, toString(ts)
    FROM shop.catalog_updates
)
GROUP BY product_id
"""

LIVE_STATS_SQL = """
SELECT product_id::String AS product_id,
       countIf(event_type = 'view') AS views_live,
       countIf(event_type = 'add_to_cart') AS carts_live,
       countIf(event_type = 'purchase') AS purchases_live
FROM shop.events
WHERE ts >= now() - INTERVAL 24 HOUR
GROUP BY product_id
"""

NEW_REQUESTS_SQL = """
SELECT request_id, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 400
"""


def bg_worker() -> None:
    global ACTIVE_IDS, TOP_TRENDING_LIVE
    last_live = 0.0
    while True:
        try:
            r = http_bg.post(f"{API}/v1/query", json={"sql": CATALOG_ACTIVE_SQL}, timeout=8)
            if r.ok:
                rows = r.json()["data"]
                ACTIVE_IDS = {row["product_id"] for row in rows if row.get("is_active")}
                for row in rows:
                    pid = row["product_id"]
                    if pid not in CATALOG:
                        CATALOG[pid] = row
                        PROD_STATS[pid] = {
                            "views": 0.0, "carts": 0.0, "purchases": 0.0,
                            "conv_rate": 0.02, "cart_rate": 0.04,
                            "views_24h": 0.0, "carts_24h": 0.0, "purchases_24h": 0.0,
                            "recent_conv_rate": 0.02, "recent_popularity": 5.0,
                            "product_age_days": 0.0,
                        }
                    else:
                        CATALOG[pid]["price_eur"] = row.get("price_eur", CATALOG[pid].get("price_eur", 25.0))

            if time.time() - last_live >= 30.0:
                last_live = time.time()
                r_live = http_bg.post(f"{API}/v1/query", json={"sql": LIVE_STATS_SQL}, timeout=10)
                if r_live.ok:
                    live_rows = r_live.json()["data"]
                    trending_scores: list[tuple[float, str]] = []

                    for row in live_rows:
                        pid = row["product_id"]
                        v_live = float(row["views_live"])
                        c_live = float(row["carts_live"])
                        p_live = float(row["purchases_live"])

                        rec_pop = p_live * 10.0 + c_live * 3.0 + v_live * 1.0
                        live_conv = (p_live + 1.0) / (v_live + 20.0)
                        live_cart = (c_live + 1.0) / (v_live + 15.0)

                        if pid not in PROD_STATS:
                            PROD_STATS[pid] = {
                                "base_purchases": 0.0, "base_conv": live_conv, "base_cart": live_cart,
                                "views": v_live, "carts": c_live, "purchases": p_live,
                                "conv_rate": live_conv, "cart_rate": live_cart,
                                "views_24h": v_live, "carts_24h": c_live, "purchases_24h": p_live,
                                "recent_conv_rate": live_conv, "recent_popularity": rec_pop,
                                "product_age_days": 0.0,
                            }
                        else:
                            st = PROD_STATS[pid]
                            if "base_purchases" not in st:
                                st["base_purchases"] = st["purchases"]
                                st["base_conv"] = st["conv_rate"]
                                st["base_cart"] = st["cart_rate"]

                            st["purchases"] = st["base_purchases"] + p_live
                            st["conv_rate"] = st.get("base_conv", st["conv_rate"])
                            st["cart_rate"] = 0.7 * st.get("base_cart", st["cart_rate"]) + 0.3 * live_cart

                            st["views_24h"] = v_live
                            st["carts_24h"] = c_live
                            st["purchases_24h"] = p_live
                            st["recent_popularity"] = rec_pop
                            st["recent_conv_rate"] = live_conv

                        trending_scores.append((rec_pop, pid))

                    trending_scores.sort(reverse=True)
                    TOP_TRENDING_LIVE = [pid for _, pid in trending_scores[:15]]

        except Exception as e:
            print(f"[BG aviso]: {e}")
        time.sleep(15.0)


def build_feature_row(cand_id: str, viewed_ids: list[str]) -> list[float]:
    c_info = CATALOG.get(cand_id, {})
    st = PROD_STATS.get(cand_id, {
        "purchases": 0.0, "conv_rate": 0.01, "cart_rate": 0.02,
        "purchases_24h": 0.0, "views_24h": 0.0, "carts_24h": 0.0,
        "product_age_days": 0.0, "recent_popularity": 0.0,
    })

    c_price = float(c_info.get("price_eur", 25.0))
    c_cat = c_info.get("category", "")
    c_subcat = c_info.get("subcategory", "")
    c_brand = c_info.get("brand", "")

    same_subcat = 0.0
    same_cat = 0.0
    same_brand = 0.0
    co_score_total = 0.0
    co_score_last = 0.0
    viewed_prices = []

    for idx, v_id in enumerate(reversed(viewed_ids)):
        v_info = CATALOG.get(v_id)
        if not v_info:
            continue
        weight = 1.0 / (1.0 + 0.5 * idx)
        if v_info.get("subcategory") == c_subcat:
            same_subcat += weight
        if v_info.get("category") == c_cat:
            same_cat += weight
        if v_info.get("brand") == c_brand:
            same_brand += weight

        pair_score = CO_MATRIX.get(v_id, {}).get(cand_id, 0.0)
        co_score_total += weight * pair_score
        if idx == 0:
            co_score_last = pair_score
        viewed_prices.append(float(v_info.get("price_eur", c_price)))

    avg_session_price = sum(viewed_prices) / len(viewed_prices) if viewed_prices else c_price
    price_ratio = c_price / max(avg_session_price, 1.0)
    already_viewed = 1.0 if cand_id in viewed_ids else 0.0

    return [
        st["purchases"],
        st["conv_rate"],
        st["cart_rate"],
        c_price,
        price_ratio,
        same_subcat,
        same_cat,
        same_brand,
        co_score_last,
        co_score_total,
        already_viewed,
        1.0,
        st["purchases_24h"],
        st["views_24h"],
        st["carts_24h"],
        st["product_age_days"],
        st["recent_popularity"],
    ]


def get_candidates(viewed_ids: list[str], active: set[str]) -> list[str]:
    cands: list[str] = []
    seen_c: set[str] = set()

    def add_c(pid: str) -> None:
        if pid in active and pid not in seen_c:
            seen_c.add(pid)
            cands.append(pid)

    for v_id in reversed(viewed_ids[-2:]):
        add_c(v_id)

    for v_id in reversed(viewed_ids):
        for cid in CO_TOP_LIST.get(v_id, [])[:10]:
            add_c(cid)
        sc = CATALOG.get(v_id, {}).get("subcategory", "")
        if sc:
            for cid in TOP_BY_SUBCAT.get(sc, [])[:8]:
                add_c(cid)
        if len(cands) >= 26:
            break

    for cid in TOP_TRENDING_LIVE[:5]:
        add_c(cid)

    for cid in TOP_GLOBAL_HIST:
        add_c(cid)
        if len(cands) >= 40:
            break

    return cands


def diversify_and_fill(top_10: list[str], active: set[str]) -> list[str]:
    diverse_10 = []
    subcat_counts = defaultdict(int)
    
    for pid in top_10:
        sc = CATALOG.get(pid, {}).get("subcategory", "unknown")
        if subcat_counts[sc] < 3:
            diverse_10.append(pid)
            subcat_counts[sc] += 1
            
    if len(diverse_10) < 10:
        for pid in TOP_TRENDING_LIVE + TOP_GLOBAL_HIST:
            if pid not in diverse_10 and pid in active:
                diverse_10.append(pid)
            if len(diverse_10) == 10:
                break
    return diverse_10[:10]


def best_sellers(until: str | None = None) -> list[str]:
    return TOP_GLOBAL_HIST if TOP_GLOBAL_HIST else list(CATALOG.keys())


def recommend(recent_views: list[Record], best: list[str] | None = None, unavailable: set[str] = frozenset()) -> list[str]:
    viewed_ids = [
        v["product_id"] for v in (recent_views or [])
        if isinstance(v, dict) and "product_id" in v
    ]
    active = set(ACTIVE_IDS) if ACTIVE_IDS else set(CATALOG.keys())

    if LGB_MODEL is None:
        fallback_list = best if best is not None else TOP_GLOBAL_HIST
        return [p for p in fallback_list if p not in unavailable][:10]

    cands = get_candidates(viewed_ids, active)
    if not cands:
        fallback_list = best if best is not None else TOP_GLOBAL_HIST
        return [p for p in fallback_list if p not in unavailable][:10]

    all_rows = [build_feature_row(cid, viewed_ids) for cid in cands]
    X_batch = np.array(all_rows, dtype=np.float32)
    probs = LGB_MODEL.predict_proba(X_batch)[:, 1]

    prices = np.array([float(CATALOG.get(c, {}).get("price_eur", 25.0)) for c in cands])
    final_scores = probs * (prices ** 0.2)
    
    ranked_indices = np.argsort(final_scores)[::-1]
    top_10 = [cands[i] for i in ranked_indices if cands[i] not in unavailable][:10]
    
    top_10 = diversify_and_fill(top_10, active)

    if len(top_10) < 10:
        fallback_list = best if best is not None else TOP_GLOBAL_HIST
        for p in fallback_list:
            if p not in top_10 and p not in unavailable:
                top_10.append(p)
            if len(top_10) == 10:
                break

    return top_10[:10]


def predict_batch(requests_batch: list[Record]) -> list[Record]:
    active = ACTIVE_IDS if ACTIVE_IDS else set(CATALOG.keys())
    all_rows: list[list[float]] = []
    req_slices: list[tuple[str, list[str], int, int]] = []

    for req in requests_batch:
        r_views = req.get("recent_views") or []
        viewed_ids = [v["product_id"] for v in r_views if isinstance(v, dict) and "product_id" in v]
        cands = get_candidates(viewed_ids, active)

        start_idx = len(all_rows)
        for cid in cands:
            all_rows.append(build_feature_row(cid, viewed_ids))
        end_idx = len(all_rows)
        req_slices.append((req["request_id"], cands, start_idx, end_idx))

    if not all_rows:
        return []

    X_batch = np.array(all_rows, dtype=np.float32)
    probs = LGB_MODEL.predict_proba(X_batch)[:, 1]

    submissions: list[Record] = []
    for req_id, cands, start_idx, end_idx in req_slices:
        req_probs = probs[start_idx:end_idx]
        
        prices = np.array([float(CATALOG.get(c, {}).get("price_eur", 25.0)) for c in cands])
        final_scores = req_probs * (prices ** 0.2)
        
        ranked_indices = np.argsort(final_scores)[::-1][:10]
        top_10 = [cands[i] for i in ranked_indices]
        
        top_10 = diversify_and_fill(top_10, active)
        
        submissions.append({
            "request_id": req_id,
            "items": top_10,
            "model_version": "v7-lgbm17",
        })

    return submissions


def main() -> None:
    threading.Thread(target=bg_worker, daemon=True).start()
    seen: OrderedDict[str, None] = OrderedDict()
    print("¡Recomendador v7-lgbm17 activo!")

    while True:
        try:
            r = http_fast.post(f"{API}/v1/query", json={"sql": NEW_REQUESTS_SQL}, timeout=4)
            r.raise_for_status()
            fresh = []
            for req in r.json()["data"]:
                if req["request_id"] not in seen:
                    seen[req["request_id"]] = None
                    fresh.append(req)
                    if len(seen) > 100_000:
                        seen.popitem(last=False)

            if fresh:
                answers = predict_batch(fresh)
                http_fast.post(
                    f"{API}/v1/tables/submissions",
                    params={"database": TEAM},
                    json=answers,
                    timeout=8,
                ).raise_for_status()
                print(f"answered {len(answers)} (v7-lgbm17-money)")
        except Exception as e:
            print(f"error: {e}")
            time.sleep(0.5)

        time.sleep(0.05)


if __name__ == "__main__":
    main()