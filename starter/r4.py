from __future__ import annotations

import os
import random
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import chain
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
MODEL_VERSION = "v9-lgbm17-cold-buy"
DEADLINE_S = 3.0
SAFETY_S = 0.4
BATCH = 20

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "modelo_lgbm_v9.pkl"
FEAT_PATH = MODELS_DIR / "features_data_v9.pkl"

print(f"Cargando modelo LightGBM (17 features) desde {MODEL_PATH}...")
LGB_MODEL = joblib.load(MODEL_PATH)
FEAT_DATA = joblib.load(FEAT_PATH)
CATALOG: dict[str, Record] = FEAT_DATA["catalog"]
PROD_STATS: dict[str, dict[str, float]] = FEAT_DATA["prod_stats"]
CO_MATRIX: dict[str, dict[str, float]] = FEAT_DATA["co_matrix"]
SUB_CONV: dict[str, float] = FEAT_DATA.get("sub_conv", {})
assert LGB_MODEL.n_features_in_ == 17

TOP_GLOBAL_HIST = [pid for pid, _ in sorted(PROD_STATS.items(), key=lambda x: x[1].get("purchases", 0.0), reverse=True)]
FALLBACK = TOP_GLOBAL_HIST[:300]

TOP_BY_SUBCAT: dict[str, list[str]] = defaultdict(list)
for pid in TOP_GLOBAL_HIST:
    sc = CATALOG.get(pid, {}).get("subcategory", "")
    if sc and len(TOP_BY_SUBCAT[sc]) < 20:
        TOP_BY_SUBCAT[sc].append(pid)

CO_TOP_LIST: dict[str, list[str]] = {
    v_id: [cid for cid, _ in sorted(recs.items(), key=lambda x: x[1], reverse=True)[:15]]
    for v_id, recs in CO_MATRIX.items()
}

NEW_POOL: list[str] = [
    pid for pid, st in sorted(PROD_STATS.items(), key=lambda x: x[1].get("product_age_days", 1e9))
    if st.get("product_age_days", 1e9) < 14
][:300]

ACTIVE_IDS: set[str] = {pid for pid, c in CATALOG.items() if c.get("is_active", True)}
OUT: set[str] = set()
BOUGHT: dict[str, set[str]] = {}
TOP_TRENDING_LIVE: list[str] = []

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

OUT_OF_STOCK_SQL = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'
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

PURCHASES_SQL = """
SELECT user_id::String AS user_id, product_id::String AS product_id
FROM shop.history_events
WHERE event_type = 'purchase'
UNION ALL
SELECT user_id::String AS user_id, product_id::String AS product_id
FROM shop.events
WHERE event_type = 'purchase'
"""

NEW_REQUESTS_SQL = """
SELECT request_id, ts, user_id, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 60
"""


def age_s(ts: Any) -> float:
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return time.time() - dt.timestamp()
    except ValueError:
        return 0.0


def viewed_from(recent_views: Any) -> list[str]:
    return [v["product_id"] for v in (recent_views or []) if isinstance(v, dict) and "product_id" in v]


def bg_worker() -> None:
    global ACTIVE_IDS, OUT, TOP_TRENDING_LIVE, BOUGHT
    last_out = last_catalog = last_live = last_bought = 0.0
    while True:
        try:
            now = time.time()

            if now - last_out >= 3.0:
                last_out = now
                r = http_bg.post(f"{API}/v1/query", json={"sql": OUT_OF_STOCK_SQL}, timeout=10)
                if r.ok:
                    OUT = {row["product_id"] for row in r.json()["data"]}

            if now - last_catalog >= 10.0:
                last_catalog = now
                r = http_bg.post(f"{API}/v1/query", json={"sql": CATALOG_ACTIVE_SQL}, timeout=15)
                if r.ok:
                    rows = r.json()["data"]
                    ACTIVE_IDS = {row["product_id"] for row in rows if row.get("is_active")}
                    for row in rows:
                        pid = row["product_id"]
                        if pid not in CATALOG:
                            CATALOG[pid] = row
                            prior = SUB_CONV.get(row.get("subcategory", ""), 0.02)
                            PROD_STATS[pid] = {
                                "views": 0.0, "carts": 0.0, "purchases": 0.0,
                                "conv_rate": prior, "cart_rate": 0.04,
                                "views_24h": 0.0, "carts_24h": 0.0, "purchases_24h": 0.0,
                                "recent_conv_rate": prior, "recent_popularity": 5.0,
                                "product_age_days": 0.0,
                            }
                            NEW_POOL.append(pid)
                            del NEW_POOL[:-300]
                        else:
                            CATALOG[pid]["price_eur"] = float(row.get("price_eur", CATALOG[pid].get("price_eur", 25.0)))

            if now - last_live >= 30.0:
                last_live = now
                r_live = http_bg.post(f"{API}/v1/query", json={"sql": LIVE_STATS_SQL}, timeout=20)
                if r_live.ok:
                    scores: list[tuple[float, str]] = []
                    for row in r_live.json()["data"]:
                        pid = row["product_id"]
                        v_live = float(row["views_live"])
                        c_live = float(row["carts_live"])
                        p_live = float(row["purchases_live"])
                        rec_pop = p_live * 10.0 + c_live * 3.0 + v_live
                        prior = SUB_CONV.get(CATALOG.get(pid, {}).get("subcategory", ""), 0.01)
                        live_conv = (p_live + 20.0 * prior) / (v_live + 20.0)
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
                            st["conv_rate"] = 0.7 * st["base_conv"] + 0.3 * live_conv
                            st["cart_rate"] = 0.7 * st["base_cart"] + 0.3 * live_cart
                            st["views_24h"] = v_live
                            st["carts_24h"] = c_live
                            st["purchases_24h"] = p_live
                            st["recent_popularity"] = rec_pop
                            st["recent_conv_rate"] = live_conv
                        scores.append((rec_pop, pid))
                    scores.sort(reverse=True)
                    TOP_TRENDING_LIVE = [pid for _, pid in scores[:15]]

            if now - last_bought >= 600.0:
                last_bought = now
                r_p = http_bg.post(f"{API}/v1/query", json={"sql": PURCHASES_SQL}, timeout=90)
                if r_p.ok:
                    nb: dict[str, set[str]] = defaultdict(set)
                    for row in r_p.json()["data"]:
                        nb[row["user_id"]].add(row["product_id"])
                    BOUGHT = dict(nb)
        except Exception as e:
            print(f"[BG aviso]: {e}")
        time.sleep(1.0)


def build_feature_row(cand_id: str, viewed_ids: list[str]) -> list[float]:
    c_info = CATALOG.get(cand_id, {})
    st = PROD_STATS.get(cand_id) or {
        "purchases": 0.0, "conv_rate": 0.01, "cart_rate": 0.02,
        "purchases_24h": 0.0, "views_24h": 0.0, "carts_24h": 0.0,
        "product_age_days": 0.0, "recent_popularity": 0.0,
    }
    c_price = float(c_info.get("price_eur", 25.0))
    c_cat = c_info.get("category", "")
    c_subcat = c_info.get("subcategory", "")
    c_brand = c_info.get("brand", "")

    same_subcat = same_cat = same_brand = 0.0
    co_score_total = co_score_last = 0.0
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

    avg_price = sum(viewed_prices) / len(viewed_prices) if viewed_prices else c_price
    price_ratio = c_price / max(avg_price, 1.0)
    already_viewed = 1.0 if cand_id in viewed_ids else 0.0

    return [
        float(st["purchases"]), float(st["conv_rate"]), float(st["cart_rate"]),
        c_price, price_ratio, same_subcat, same_cat, same_brand,
        co_score_last, co_score_total, already_viewed, 1.0,
        float(st["purchases_24h"]), float(st["views_24h"]), float(st["carts_24h"]),
        float(st.get("product_age_days", 0.0)), float(st["recent_popularity"]),
    ]


def get_candidates(viewed_ids: list[str], active: set[str], user_id: str | None) -> list[str]:
    cands: list[str] = []
    seen_c: set[str] = set()
    bought = BOUGHT.get(user_id, set()) if user_id else set()

    def add_c(pid: str) -> None:
        if pid in active and pid not in seen_c and pid not in bought:
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

    viewed_subs = {CATALOG.get(v, {}).get("subcategory", "") for v in viewed_ids}
    pool = list(NEW_POOL)
    new_match = [p for p in pool if CATALOG.get(p, {}).get("subcategory", "") in viewed_subs]
    for cid in new_match[:3]:
        add_c(cid)
    for cid in random.sample(pool, min(2, len(pool))):
        add_c(cid)

    for cid in TOP_TRENDING_LIVE[:5]:
        add_c(cid)

    for cid in TOP_GLOBAL_HIST:
        add_c(cid)
        if len(cands) >= 40:
            break

    return cands


def finalize(items: list[str], user_id: str | None) -> list[str]:
    active = ACTIVE_IDS if ACTIVE_IDS else set(CATALOG.keys())
    bought = BOUGHT.get(user_id, set()) if user_id else set()
    out: list[str] = []
    used: set[str] = set()
    for pid in chain(items, FALLBACK, active):
        if pid in active and pid not in OUT and pid not in bought and pid not in used:
            used.add(pid)
            out.append(pid)
            if len(out) == 10:
                break
    return out


def predict_batch(batch: list[Record]) -> list[Record]:
    active = ACTIVE_IDS if ACTIVE_IDS else set(CATALOG.keys())
    rows: list[list[float]] = []
    slices: list[tuple[str, str | None, list[str], int, int]] = []

    for req in batch:
        viewed = viewed_from(req.get("recent_views"))
        uid = req.get("user_id")
        cands = get_candidates(viewed, active, uid)
        start = len(rows)
        rows.extend(build_feature_row(c, viewed) for c in cands)
        slices.append((req["request_id"], uid, cands, start, len(rows)))

    probs = (LGB_MODEL.predict_proba(np.asarray(rows, dtype=np.float32))[:, 1]
             if rows else np.array([]))

    answers: list[Record] = []
    for rid, uid, cands, s, e in slices:
        ranked: list[str] = []
        if e > s:
            order = np.argsort(probs[s:e])[::-1]
            ranked = [cands[i] for i in order]
        answers.append({
            "request_id": rid,
            "items": finalize(ranked, uid),
            "model_version": MODEL_VERSION,
        })
    return answers


def push_with_retry(answers: list[Record]) -> bool:
    for _ in range(2):
        try:
            http_fast.post(
                f"{API}/v1/tables/submissions",
                params={"database": TEAM},
                json=answers,
                timeout=4,
            ).raise_for_status()
            return True
        except Exception as e:
            print(f"push fallido: {e}")
    return False


def main() -> None:
    threading.Thread(target=bg_worker, daemon=True).start()
    seen: set[str] = set()
    print(f"Recomendador {MODEL_VERSION} activo.")

    while True:
        try:
            if len(seen) > 200_000:
                seen.clear()
            r = http_fast.post(f"{API}/v1/query", json={"sql": NEW_REQUESTS_SQL}, timeout=4.0)
            r.raise_for_status()
            fresh = [q for q in r.json()["data"] if q["request_id"] not in seen]
            fresh.sort(key=lambda q: str(q.get("ts")))

            for i in range(0, len(fresh), BATCH):
                chunk = [q for q in fresh[i:i + BATCH] if age_s(q.get("ts")) < DEADLINE_S - SAFETY_S]
                if chunk:
                    push_with_retry(predict_batch(chunk))
                seen.update(q["request_id"] for q in fresh[i:i + BATCH])
        except Exception as e:
            print(f"error: {e}")
            time.sleep(0.2)
        time.sleep(0.05)


if __name__ == "__main__":
    main()