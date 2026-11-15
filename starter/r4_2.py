"""Recomendador LightGBM v12: candidatos, exclusión de compras recientes, selección diversificada y respuestas validadas.

Diseño: responder SIEMPRE a tiempo con 10 productos válidos. El modelo no decide
nada de eso; lo decide la capa de validación y envío.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import requests

from features import (N_FEATURES, build_feature_row, build_state,
                      get_candidates, viewed_from_requests)

for line in (Path(".env").read_text().splitlines() if Path(".env").exists() else []):
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
HEADERS = {"Authorization": f"Bearer {os.environ['RAWTREE_API_KEY']}"}

http_fast = requests.Session()
http_fast.headers.update(HEADERS)
http_bg = requests.Session()
http_bg.headers.update(HEADERS)

Record = dict[str, Any]
MODEL_VERSION = "v12-click-buy-div"
DEADLINE_S = 3.0
SAFETY_S = 0.5
BATCH = 20
CLICK_W = 0.25
DIV_DECAY = 0.6
MAX_CANDS = 60
HIST_REFRESH_S = 1800.0
LIVE_REFRESH_S = 15.0

MODELS = joblib.load("modelo_lgbm_v12.pkl")
CLICK_MODEL = MODELS["click"]
BUY_MODEL = MODELS["buy"]
FEAT = joblib.load("features_data_v12.pkl")
assert FEAT.get("n_features") == N_FEATURES, "features_data no coincide con features.py"
assert CLICK_MODEL.n_features_in_ == N_FEATURES, "el modelo no coincide con features.py"
assert BUY_MODEL.n_features_in_ == N_FEATURES, "el modelo no coincide con features.py"

STATE = build_state(FEAT["catalog"], FEAT["prod_stats"], FEAT["co_matrix"])
CATALOG: dict[str, Record] = STATE["catalog"]
ACTIVE: set[str] = {pid for pid, c in CATALOG.items() if c.get("is_active", True)}
OUT: set[str] = set()
HIST_BOUGHT: dict[str, set[str]] = {}
LIVE_BOUGHT: dict[str, set[str]] = {}

TOP_BY_CAT: dict[str, list[str]] = {}
for pid in STATE["top_global"]:
    cat = CATALOG.get(pid, {}).get("category", "")
    if cat:
        TOP_BY_CAT.setdefault(cat, [])
        if len(TOP_BY_CAT[cat]) < 30:
            TOP_BY_CAT[cat].append(pid)

CATALOG_ACTIVE_SQL = """
SELECT product_id,
       argMax(category, v)    AS category,
       argMax(subcategory, v) AS subcategory,
       argMax(brand, v)       AS brand,
       argMax(price_eur, v)   AS price_eur,
       argMax(is_active, v)   AS is_active
FROM (
    SELECT product_id::String AS product_id,
           category::String AS category, subcategory::String AS subcategory,
           brand::String AS brand, price_eur::Float64 AS price_eur,
           is_active::Bool AS is_active, '' AS v
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String, category::String, subcategory::String,
           brand::String, price_eur::Float64, is_active::Bool, toString(ts)
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

HIST_PURCHASE_SQL = """
SELECT user_id::String AS user_id, product_id::String AS product_id
FROM shop.history_events
WHERE event_type = 'purchase' AND ts::DateTime64(3) > now() - INTERVAL 30 DAY
"""

LIVE_PURCHASE_SQL = """
SELECT user_id::String AS user_id, product_id::String AS product_id
FROM shop.events
WHERE event_type = 'purchase'
"""

NEW_REQUESTS_SQL = """
SELECT request_id, ts, user_id, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 100
"""


def purchase_map(rows: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in rows:
        out.setdefault(str(row["user_id"]), set()).add(str(row["product_id"]))
    return out


def bought_by(uid: str) -> set[str]:
    if not uid:
        return set()
    return HIST_BOUGHT.get(uid, set()) | LIVE_BOUGHT.get(uid, set())


def bg_worker() -> None:
    global ACTIVE, OUT, HIST_BOUGHT, LIVE_BOUGHT
    last_catalog = 0.0
    last_hist = 0.0
    last_live = 0.0
    while True:
        try:
            r = http_bg.post(f"{API}/v1/query", json={"sql": OUT_OF_STOCK_SQL}, timeout=3)
            if r.ok:
                OUT = {row["product_id"] for row in r.json()["data"]}

            if time.time() - last_catalog >= 10.0:
                last_catalog = time.time()
                r = http_bg.post(f"{API}/v1/query", json={"sql": CATALOG_ACTIVE_SQL}, timeout=8)
                if r.ok:
                    rows = r.json()["data"]
                    ACTIVE = {row["product_id"] for row in rows if row.get("is_active")}
                    for row in rows:
                        pid = row["product_id"]
                        if pid not in CATALOG:
                            CATALOG[pid] = row
                        else:
                            CATALOG[pid]["price_eur"] = float(
                                row.get("price_eur", CATALOG[pid].get("price_eur", 25.0)))

            if time.time() - last_hist >= HIST_REFRESH_S:
                last_hist = time.time()
                r = http_bg.post(f"{API}/v1/query", json={"sql": HIST_PURCHASE_SQL}, timeout=30)
                if r.ok:
                    HIST_BOUGHT = purchase_map(r.json()["data"])

            if time.time() - last_live >= LIVE_REFRESH_S:
                last_live = time.time()
                r = http_bg.post(f"{API}/v1/query", json={"sql": LIVE_PURCHASE_SQL}, timeout=8)
                if r.ok:
                    LIVE_BOUGHT = purchase_map(r.json()["data"])
        except Exception as e:
            print(f"[BG aviso]: {e}")
        time.sleep(1.0)


FALLBACK = STATE["top_global"][:200]


def finalize(items: list[str], blocked: set[str]) -> list[str]:
    out: list[str] = []
    used: set[str] = set()
    for pid in list(items) + FALLBACK:
        if pid in ACTIVE and pid not in blocked and pid not in used:
            used.add(pid)
            out.append(pid)
            if len(out) == 10:
                break
    return out


def candidates(viewed: list[str], blocked: set[str]) -> list[str]:
    base = get_candidates(STATE, viewed, ACTIVE, max_cands=MAX_CANDS)
    cands = [p for p in base if p not in blocked]
    seen = set(base)
    for v in reversed(viewed[-3:]):
        cat = CATALOG.get(v, {}).get("category", "")
        for pid in TOP_BY_CAT.get(cat, [])[:10]:
            if pid in ACTIVE and pid not in seen and pid not in blocked:
                seen.add(pid)
                cands.append(pid)
    return cands


def diversified_list(cands: list[str], scores: np.ndarray) -> list[str]:
    chosen: list[str] = []
    sub_count: dict[str, int] = {}
    remaining = list(range(len(cands)))
    while remaining and len(chosen) < 10:
        best_i = remaining[0]
        best_val = -1.0
        for i in remaining:
            sub = CATALOG.get(cands[i], {}).get("subcategory", "")
            val = float(scores[i]) * (DIV_DECAY ** sub_count.get(sub, 0))
            if val > best_val:
                best_val = val
                best_i = i
        remaining.remove(best_i)
        chosen.append(cands[best_i])
        sub = CATALOG.get(cands[best_i], {}).get("subcategory", "")
        sub_count[sub] = sub_count.get(sub, 0) + 1
    return chosen


def predict_batch(batch: list[Record]) -> list[Record]:
    rows: list[list[float]] = []
    slices: list[tuple[str, list[str], int, int, set[str]]] = []

    for req in batch:
        viewed = viewed_from_requests(req.get("recent_views"))
        blocked = bought_by(str(req.get("user_id", ""))) | OUT
        cands = candidates(viewed, blocked)
        start = len(rows)
        rows.extend(build_feature_row(STATE, c, viewed) for c in cands)
        slices.append((req["request_id"], cands, start, len(rows), blocked))

    if rows:
        X = np.asarray(rows, dtype=np.float32)
        scores = (BUY_MODEL.predict_proba(X)[:, 1]
                  + CLICK_W * CLICK_MODEL.predict_proba(X)[:, 1])
    else:
        scores = np.array([])

    answers: list[Record] = []
    for rid, cands, s, e, blocked in slices:
        ranked = diversified_list(cands, scores[s:e]) if e > s else []
        answers.append({
            "request_id": rid,
            "items": finalize(ranked, blocked),
            "model_version": MODEL_VERSION,
        })
    return answers


def age_s(ts: Any) -> float:
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return time.time() - dt.timestamp()
    except ValueError:
        return 0.0


def push_with_retry(answers: list[Record]) -> bool:
    for attempt in range(2):
        try:
            http_fast.post(
                f"{API}/v1/tables/submissions",
                params={"database": TEAM},
                json=answers,
                timeout=2,
            ).raise_for_status()
            return True
        except Exception as e:
            print(f"push intento {attempt + 1} fallido: {e}")
    return False


def main() -> None:
    threading.Thread(target=bg_worker, daemon=True).start()
    seen: set[str] = set()
    lat_ms: list[float] = []
    print(f"Recomendador {MODEL_VERSION} activo.")

    while True:
        try:
            if len(seen) > 100_000:
                seen.clear()

            r = http_fast.post(f"{API}/v1/query", json={"sql": NEW_REQUESTS_SQL}, timeout=1.5)
            r.raise_for_status()
            fresh = [q for q in r.json()["data"] if q["request_id"] not in seen]

            fresh.sort(key=lambda q: str(q.get("ts")))

            for q in fresh:
                if age_s(q.get("ts")) > DEADLINE_S - SAFETY_S:
                    seen.add(q["request_id"])
            live = [q for q in fresh if q["request_id"] not in seen]

            for i in range(0, len(live), BATCH):
                chunk = live[i:i + BATCH]
                answers = predict_batch(chunk)
                if push_with_retry(answers):
                    seen.update(q["request_id"] for q in chunk)
                    for q in chunk:
                        lat_ms.append(age_s(q.get("ts")) * 1000)
                    if len(lat_ms) >= 200:
                        lat_ms.sort()
                        print(f"latencia p50={lat_ms[len(lat_ms)//2]:.0f} ms  "
                              f"p95={lat_ms[int(len(lat_ms)*0.95)]:.0f} ms")
                        lat_ms.clear()
        except Exception as e:
            print(f"error: {e}")
            time.sleep(0.5)

        time.sleep(0.1)


if __name__ == "__main__":
    main()