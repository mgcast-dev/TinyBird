"""Recomendador en Tiempo Real con LightGBM (Two-Stage Candidate Generation + Boosting Re-ranking)."""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import requests

# =====================================================================
# 1. CONFIGURACIÓN Y CARGA DEL MODELO LIGHTGBM EN MEMORIA
# ================================================  =====================
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

print("Cargando modelo LightGBM y diccionarios de features desde disco...")
# Definimos la ruta de la carpeta models relativa a este archivo
MODELS_DIR = Path(__file__).resolve().parent / "models"

print("Cargando modelo LightGBM desde starter/models/...")

try:
    LGB_MODEL = joblib.load(MODELS_DIR / "modelo_lgbm.pkl")
    FEAT_DATA = joblib.load(MODELS_DIR / "features_data.pkl")

    CATALOG: dict[str, Record] = FEAT_DATA["catalog"]
    PROD_STATS: dict[str, dict[str, float]] = FEAT_DATA["prod_stats"]
    CO_MATRIX: dict[str, dict[str, float]] = FEAT_DATA["co_matrix"]
    print("¡Modelos y features cargados correctamente!")

except FileNotFoundError as e:
    print(f"[AVISO CRÍTICO]: No se encontró el archivo de modelo o features: {e}")
    print("Asegúrate de haber ejecutado el script de entrenamiento antes de arrancar el recomendador.")
    LGB_MODEL = None
    FEAT_DATA, CATALOG, PROD_STATS, CO_MATRIX = {}, {}, {}, {}

except Exception as e:
    print(f"[ERROR INESPERADO] al cargar los modelos: {e}")
    LGB_MODEL = None
    FEAT_DATA, CATALOG, PROD_STATS, CO_MATRIX = {}, {}, {}, {}

TOP_GLOBAL_HIST = [
    pid for pid, _ in sorted(PROD_STATS.items(), key=lambda x: x[1]["purchases"], reverse=True)
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


# =====================================================================
# 2. FUNCIONES REQUERIDAS POR EL OFFLINE EVAL (offline_eval.ipynb)
# =====================================================================
def best_sellers(until: str | None = None) -> list[str]:
    """Devuelve la lista de productos ordenados por ventas históricas para el evaluador offline."""
    if TOP_GLOBAL_HIST:
        return TOP_GLOBAL_HIST
    # Fallback si no hay modelo entrenado aún
    return [pid for pid in CATALOG.keys()]


def recommend(recent_views: list[Record], best: list[str] | None = None, unavailable: set[str] = frozenset()) -> list[str]:
    """Función que usa el notebook offline_eval.ipynb para evaluar vuestra política con LightGBM."""
    viewed_ids = [
        v["product_id"] for v in (recent_views or [])
        if isinstance(v, dict) and "product_id" in v
    ]
    active = set(CATALOG.keys()) or unavailable
    
    # Si no hay modelo cargado, devolvemos el top básico
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

    ranked_indices = np.argsort(probs)[::-1]
    top_10 = [cands[i] for i in ranked_indices if cands[i] not in unavailable][:10]
    
    # Rellenar por si acaso faltan hasta 10
    if len(top_10) < 10:
        fallback_list = best if best is not None else TOP_GLOBAL_HIST
        for p in fallback_list:
            if p not in top_10 and p not in unavailable:
                top_10.append(p)
            if len(top_10) == 10:
                break

    return top_10[:10]


# =====================================================================
# 3. ACTUALIZADOR EN SEGUNDO PLANO (Producción en vivo)
# =====================================================================
ACTIVE_IDS: set[str] = {pid for pid, c in CATALOG.items() if c.get("is_active", True)}
LIVE_BOOST: dict[str, float] = {}

CATALOG_ACTIVE_SQL = """
SELECT product_id,
       argMax(subcategory, v) AS subcategory,
       argMax(price_eur, v)   AS price_eur,
       argMax(is_active, v)   AS is_active
FROM (
    SELECT product_id::String AS product_id, subcategory::String AS subcategory,
           price_eur::Float64 AS price_eur, is_active::Bool AS is_active, '' AS v
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String, subcategory::String, price_eur::Float64, is_active::Bool, toString(ts)
    FROM shop.catalog_updates
)
GROUP BY product_id
"""

LIVE_1H_SQL = """
SELECT product_id::String AS product_id,
       (countIf(event_type = 'purchase') + 0.5) / (countIf(event_type = 'view') + 15.0) AS live_conv
FROM shop.events
WHERE ts > now() - INTERVAL 1 HOUR
GROUP BY product_id
"""

NEW_REQUESTS_SQL = """
SELECT request_id, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 60
"""


def bg_worker() -> None:
    global ACTIVE_IDS, LIVE_BOOST
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
                    else:
                        CATALOG[pid]["price_eur"] = row.get("price_eur", CATALOG[pid].get("price_eur", 25.0))

            if time.time() - last_live >= 30.0:
                last_live = time.time()
                r_live = http_bg.post(f"{API}/v1/query", json={"sql": LIVE_1H_SQL}, timeout=8)
                if r_live.ok:
                    LIVE_BOOST = {row["product_id"]: float(row["live_conv"]) for row in r_live.json()["data"]}
        except Exception as e:
            print(f"[BG aviso]: {e}")
        time.sleep(2.0)


# =====================================================================
# 4. GENERACIÓN DE CANDIDATOS + PREDICCIÓN CON LIGHTGBM
# =====================================================================
def build_feature_row(cand_id: str, viewed_ids: list[str]) -> list[float]:
    c_info = CATALOG.get(cand_id, {})
    st = PROD_STATS.get(cand_id, {"views": 0.0, "purchases": 0.0, "conv_rate": 0.01, "cart_rate": 0.02})
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

    conv_rate = 0.7 * st["conv_rate"] + 0.3 * LIVE_BOOST.get(cand_id, st["conv_rate"])

    return [
        st["purchases"],
        conv_rate,
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
    ]


def get_candidates(viewed_ids: list[str], active: set[str]) -> list[str]:
    cands: list[str] = []
    seen_c: set[str] = set()

    def add_c(pid: str) -> None:
        if pid in active and pid not in seen_c:
            seen_c.add(pid)
            cands.append(pid)

    for v_id in reversed(viewed_ids):
        for cid in CO_TOP_LIST.get(v_id, [])[:10]:
            add_c(cid)
        sc = CATALOG.get(v_id, {}).get("subcategory", "")
        if sc:
            for cid in TOP_BY_SUBCAT.get(sc, [])[:8]:
                add_c(cid)
        if len(cands) >= 25:
            break

    for cid in TOP_GLOBAL_HIST:
        add_c(cid)
        if len(cands) >= 35:
            break

    return cands


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
        ranked_indices = np.argsort(req_probs)[::-1][:10]
        top_10 = [cands[i] for i in ranked_indices]
        submissions.append({
            "request_id": req_id,
            "items": top_10,
            "model_version": "v4-lgbm-ranker",
        })

    return submissions


def main() -> None:
    threading.Thread(target=bg_worker, daemon=True).start()
    seen: set[str] = set()
    print("¡Recomendador LightGBM activo y respondiendo en < 0.5s!")

    while True:
        try:
            if len(seen) > 100_000:
                seen.clear()

            r = http_fast.post(f"{API}/v1/query", json={"sql": NEW_REQUESTS_SQL}, timeout=3)
            r.raise_for_status()
            fresh = []
            for req in r.json()["data"]:
                if req["request_id"] not in seen:
                    seen.add(req["request_id"])
                    fresh.append(req)

            if fresh:
                answers = predict_batch(fresh)
                http_fast.post(
                    f"{API}/v1/tables/submissions",
                    params={"database": TEAM},
                    json=answers,
                    timeout=4,
                ).raise_for_status()
                print(f"answered {len(answers)} (v4-lgbm-ranker)")
        except Exception as e:
            print(f"error: {e}")
            time.sleep(0.5)

        time.sleep(0.25)


if __name__ == "__main__":
    main()