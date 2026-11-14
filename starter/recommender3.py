"""Recomendador en Tiempo Real con LightGBM.
Compatible con training_boosting_3.py
"""

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
# 1. CONFIGURACIÓN
# =====================================================================

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ.get("TEAM_ID", "team_31")

HEADERS = {
    "Authorization": f"Bearer {os.environ.get('RAWTREE_API_KEY', '')}"
}

http_fast = requests.Session()
http_fast.headers.update(HEADERS)

http_bg = requests.Session()
http_bg.headers.update(HEADERS)

Record = dict[str, Any]


# =====================================================================
# 2. CARGA DEL MODELO Y FEATURES
# =====================================================================

print("Cargando modelo LightGBM y features...")

MODELS_DIR = Path(__file__).resolve().parent / "models"

try:
    # IMPORTANTE:
    # Estos dos archivos son los generados por training_boosting_3.py
    LGB_MODEL = joblib.load(
        MODELS_DIR / "modelo_lgbm7.pkl"
    )

    FEAT_DATA = joblib.load(
        MODELS_DIR / "features_data7.pkl"
    )

    CATALOG: dict[str, Record] = FEAT_DATA["catalog"]
    PROD_STATS: dict[str, dict[str, float]] = FEAT_DATA["prod_stats"]
    CO_MATRIX: dict[str, dict[str, float]] = FEAT_DATA["co_matrix"]

    print("Modelo cargado correctamente.")
    print(f"Productos en catálogo: {len(CATALOG)}")
    print(f"Productos con estadísticas: {len(PROD_STATS)}")
    print(f"Productos en co-matrix: {len(CO_MATRIX)}")

    # Comprobación MUY IMPORTANTE:
    # training_boosting_3.py entrena con 17 features.
    expected_features = getattr(LGB_MODEL, "n_features_in_", None)

    if expected_features is not None:
        print(f"Features que espera LightGBM: {expected_features}")

        if expected_features != 17:
            raise RuntimeError(
                f"ERROR: el modelo cargado espera {expected_features} features, "
                f"pero recommender3.py está preparado para 17. "
                f"Revisa modelo_lgbm7.pkl."
            )

except FileNotFoundError as e:
    print(f"[ERROR CRÍTICO] No se encontró un archivo: {e}")
    print("Ejecuta primero training_boosting_3.py.")
    raise

except Exception as e:
    print(f"[ERROR CRÍTICO] No se pudieron cargar los modelos: {e}")
    raise


# =====================================================================
# 3. PRE-CÁLCULOS PARA GENERACIÓN DE CANDIDATOS
# =====================================================================

TOP_GLOBAL_HIST = [
    pid
    for pid, _ in sorted(
        PROD_STATS.items(),
        key=lambda x: x[1].get("purchases", 0.0),
        reverse=True,
    )
]


TOP_BY_SUBCAT: dict[str, list[str]] = defaultdict(list)

for pid in TOP_GLOBAL_HIST:
    subcat = CATALOG.get(pid, {}).get("subcategory", "")

    if subcat and len(TOP_BY_SUBCAT[subcat]) < 20:
        TOP_BY_SUBCAT[subcat].append(pid)


CO_TOP_LIST: dict[str, list[str]] = {}

for viewed_pid, recs in CO_MATRIX.items():
    CO_TOP_LIST[viewed_pid] = [
        pid
        for pid, _ in sorted(
            recs.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:15]
    ]


# =====================================================================
# 4. BEST SELLERS
# =====================================================================

def best_sellers(until: str | None = None) -> list[str]:
    """Productos ordenados por compras históricas."""

    if TOP_GLOBAL_HIST:
        return TOP_GLOBAL_HIST

    return list(CATALOG.keys())


# =====================================================================
# 5. RECOMMEND
# =====================================================================

def recommend(
    recent_views: list[Record],
    best: list[str] | None = None,
    unavailable: set[str] = frozenset(),
) -> list[str]:

    viewed_ids = [
        v["product_id"]
        for v in (recent_views or [])
        if isinstance(v, dict) and "product_id" in v
    ]

    active = ACTIVE_IDS if ACTIVE_IDS else set(CATALOG.keys())

    cands = get_candidates(viewed_ids, active)

    if not cands:
        fallback = best if best is not None else TOP_GLOBAL_HIST

        return [
            p
            for p in fallback
            if p not in unavailable and p in active
        ][:10]

    all_rows = [
        build_feature_row(cid, viewed_ids)
        for cid in cands
    ]

    X_batch = np.array(
        all_rows,
        dtype=np.float32,
    )

    probs = LGB_MODEL.predict_proba(X_batch)[:, 1]

    ranked_indices = np.argsort(probs)[::-1]

    top_10 = [
        cands[i]
        for i in ranked_indices
        if cands[i] not in unavailable
    ][:10]

    # Relleno
    if len(top_10) < 10:

        fallback = best if best is not None else TOP_GLOBAL_HIST

        for pid in fallback:

            if (
                pid not in top_10
                and pid not in unavailable
                and pid in active
            ):
                top_10.append(pid)

            if len(top_10) == 10:
                break

    return top_10[:10]


# =====================================================================
# 6. ACTUALIZACIÓN DEL CATÁLOGO EN SEGUNDO PLANO
# =====================================================================

ACTIVE_IDS: set[str] = {
    pid
    for pid, product in CATALOG.items()
    if product.get("is_active", True)
}


CATALOG_ACTIVE_SQL = """
SELECT
    product_id,
    argMax(subcategory, v) AS subcategory,
    argMax(price_eur, v) AS price_eur,
    argMax(is_active, v) AS is_active
FROM (
    SELECT
        product_id::String AS product_id,
        subcategory::String AS subcategory,
        price_eur::Float64 AS price_eur,
        is_active::Bool AS is_active,
        '' AS v
    FROM shop.catalog

    UNION ALL

    SELECT
        product_id::String,
        subcategory::String,
        price_eur::Float64,
        is_active::Bool,
        toString(ts)
    FROM shop.catalog_updates
)
GROUP BY product_id
"""


NEW_REQUESTS_SQL = """
SELECT
    request_id,
    recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 60
"""


# =====================================================================
# 7. BACKGROUND WORKER
# =====================================================================

def bg_worker() -> None:

    global ACTIVE_IDS

    while True:

        try:

            r = http_bg.post(
                f"{API}/v1/query",
                json={"sql": CATALOG_ACTIVE_SQL},
                timeout=8,
            )

            if r.ok:

                rows = r.json()["data"]

                ACTIVE_IDS = {
                    row["product_id"]
                    for row in rows
                    if row.get("is_active")
                }

                for row in rows:

                    pid = row["product_id"]

                    if pid not in CATALOG:
                        CATALOG[pid] = row

                    else:
                        CATALOG[pid]["price_eur"] = row.get(
                            "price_eur",
                            CATALOG[pid].get("price_eur", 25.0),
                        )

        except Exception as e:

            print(f"[BG aviso] {e}")

        time.sleep(2.0)


# =====================================================================
# 8. CONSTRUCCIÓN DE LAS 17 FEATURES
# =====================================================================

def build_feature_row(
    cand_id: str,
    viewed_ids: list[str],
) -> list[float]:

    c_info = CATALOG.get(cand_id, {})

    st = PROD_STATS.get(
        cand_id,
        {
            "views": 0.0,
            "carts": 0.0,
            "purchases": 0.0,
            "conv_rate": 0.01,
            "cart_rate": 0.02,
            "views_24h": 0.0,
            "carts_24h": 0.0,
            "purchases_24h": 0.0,
            "product_age_days": 30.0,
            "recent_popularity": 0.0,
        },
    )

    c_price = float(
        c_info.get("price_eur", 25.0)
    )

    c_cat = c_info.get("category", "")
    c_subcat = c_info.get("subcategory", "")
    c_brand = c_info.get("brand", "")

    same_subcat = 0.0
    same_cat = 0.0
    same_brand = 0.0

    co_score_total = 0.0
    co_score_last = 0.0

    viewed_prices = []

    # Mismo cálculo que en training_boosting_3.py
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

        pair_score = CO_MATRIX.get(
            v_id,
            {},
        ).get(
            cand_id,
            0.0,
        )

        co_score_total += weight * pair_score

        if idx == 0:
            co_score_last = pair_score

        viewed_prices.append(
            float(
                v_info.get(
                    "price_eur",
                    c_price,
                )
            )
        )

    if viewed_prices:

        avg_session_price = (
            sum(viewed_prices)
            / len(viewed_prices)
        )

    else:

        avg_session_price = c_price

    price_ratio = (
        c_price
        / max(avg_session_price, 1.0)
    )

    already_viewed = (
        1.0
        if cand_id in viewed_ids
        else 0.0
    )

    # ================================================================
    # EXACTAMENTE LAS 17 FEATURES DEL TRAINING
    # ================================================================

    return [

        # 0
        float(st.get("purchases", 0.0)),

        # 1
        float(st.get("conv_rate", 0.01)),

        # 2
        float(st.get("cart_rate", 0.02)),

        # 3
        c_price,

        # 4
        price_ratio,

        # 5
        same_subcat,

        # 6
        same_cat,

        # 7
        same_brand,

        # 8
        co_score_last,

        # 9
        co_score_total,

        # 10
        already_viewed,

        # 11
        1.0,

        # 12
        float(st.get("purchases_24h", 0.0)),

        # 13
        float(st.get("views_24h", 0.0)),

        # 14
        float(st.get("carts_24h", 0.0)),

        # 15
        float(st.get("product_age_days", 30.0)),

        # 16
        float(st.get("recent_popularity", 0.0)),
    ]


# =====================================================================
# 9. GENERACIÓN DE CANDIDATOS
# =====================================================================

def get_candidates(
    viewed_ids: list[str],
    active: set[str],
) -> list[str]:

    cands: list[str] = []
    seen_c: set[str] = set()

    def add_c(pid: str) -> None:

        if (
            pid in active
            and pid not in seen_c
        ):

            seen_c.add(pid)
            cands.append(pid)

    # ---------------------------------------------------------------
    # 1. Co-ocurrencias
    # 2. Misma subcategoría
    # ---------------------------------------------------------------

    for v_id in reversed(viewed_ids):

        # Co-purchases / co-occurrence
        for cid in CO_TOP_LIST.get(v_id, [])[:10]:
            add_c(cid)

        # Misma subcategoría
        subcat = CATALOG.get(
            v_id,
            {},
        ).get(
            "subcategory",
            "",
        )

        if subcat:

            for cid in TOP_BY_SUBCAT.get(
                subcat,
                [],
            )[:8]:

                add_c(cid)

        if len(cands) >= 25:
            break

    # ---------------------------------------------------------------
    # 3. Best sellers para completar candidatos
    # ---------------------------------------------------------------

    for cid in TOP_GLOBAL_HIST:

        add_c(cid)

        if len(cands) >= 35:
            break

    return cands


# =====================================================================
# 10. PREDICCIÓN BATCH
# =====================================================================

def predict_batch(
    requests_batch: list[Record],
) -> list[Record]:

    active = (
        ACTIVE_IDS
        if ACTIVE_IDS
        else set(CATALOG.keys())
    )

    all_rows: list[list[float]] = []

    req_slices: list[
        tuple[str, list[str], int, int]
    ] = []

    for req in requests_batch:

        r_views = req.get(
            "recent_views"
        ) or []

        viewed_ids = [
            v["product_id"]
            for v in r_views
            if (
                isinstance(v, dict)
                and "product_id" in v
            )
        ]

        cands = get_candidates(
            viewed_ids,
            active,
        )

        start_idx = len(all_rows)

        for cid in cands:

            all_rows.append(
                build_feature_row(
                    cid,
                    viewed_ids,
                )
            )

        end_idx = len(all_rows)

        req_slices.append(
            (
                req["request_id"],
                cands,
                start_idx,
                end_idx,
            )
        )

    if not all_rows:
        return []

    X_batch = np.array(
        all_rows,
        dtype=np.float32,
    )

    # Comprobación de seguridad
    if X_batch.shape[1] != 17:

        raise RuntimeError(
            f"ERROR DE FEATURES: X tiene "
            f"{X_batch.shape[1]} columnas y "
            f"el modelo necesita 17."
        )

    probs = LGB_MODEL.predict_proba(
        X_batch
    )[:, 1]

    submissions: list[Record] = []

    for (
        req_id,
        cands,
        start_idx,
        end_idx,
    ) in req_slices:

        req_probs = probs[
            start_idx:end_idx
        ]

        ranked_indices = np.argsort(
            req_probs
        )[::-1][:10]

        top_10 = [
            cands[i]
            for i in ranked_indices
        ]

        # Seguridad: exactamente hasta 10,
        # sin duplicados
        top_10 = list(dict.fromkeys(top_10))[:10]

        submissions.append(
            {
                "request_id": req_id,
                "items": top_10,
                "model_version": "v5-lgbm7-17features",
            }
        )

    return submissions


# =====================================================================
# 11. MAIN
# =====================================================================

def main() -> None:

    threading.Thread(
        target=bg_worker,
        daemon=True,
    ).start()

    seen: set[str] = set()

    print(
        "¡Recomendador LightGBM activo!"
    )
    print(
        "Modelo: modelo_lgbm7.pkl"
    )
    print(
        "Features: features_data7.pkl"
    )
    print(
        "Número de features: 17"
    )

    while True:

        try:

            if len(seen) > 100_000:
                seen.clear()

            r = http_fast.post(
                f"{API}/v1/query",
                json={
                    "sql": NEW_REQUESTS_SQL
                },
                timeout=3,
            )

            r.raise_for_status()

            fresh = []

            for req in r.json()["data"]:

                request_id = req["request_id"]

                if request_id not in seen:

                    seen.add(request_id)
                    fresh.append(req)

            if fresh:

                answers = predict_batch(
                    fresh
                )

                if answers:

                    response = http_fast.post(
                        f"{API}/v1/tables/submissions",
                        params={
                            "database": TEAM
                        },
                        json=answers,
                        timeout=4,
                    )

                    response.raise_for_status()

                    print(
                        f"answered {len(answers)} "
                        f"(features_data7.pkl)"
                    )

        except Exception as e:

            print(
                f"error: {e}"
            )

            time.sleep(0.5)

        time.sleep(0.25)


# =====================================================================
# 12. ARRANQUE
# =====================================================================

if __name__ == "__main__":
    main()