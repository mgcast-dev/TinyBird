from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import requests

from sklearn.linear_model import LogisticRegression


# =====================================================================
# 1. CONFIGURACIÓN Y CONEXIÓN A RAWTREE
# =====================================================================

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get(
    "RAWTREE_API_URL",
    "https://api.rawtree.com"
)

TEAM = os.environ["TEAM_ID"]

http = requests.Session()
http.headers["Authorization"] = (
    f"Bearer {os.environ['RAWTREE_API_KEY']}"
)


def query(sql: str, timeout: int = 45):
    r = http.post(
        f"{API}/v1/query",
        json={"sql": sql},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["data"]


# =====================================================================
# 2. CATÁLOGO
# =====================================================================

print("1/4 Descargando catálogo y estadísticas históricas...")


cat_rows = query("""
SELECT product_id::String AS product_id,
       category::String AS category,
       subcategory::String AS subcategory,
       brand::String AS brand,
       price_eur::Float64 AS price_eur
FROM shop.catalog
""")


catalog = {
    r["product_id"]: r
    for r in cat_rows
}


# =====================================================================
# 3. ESTADÍSTICAS HISTÓRICAS
# =====================================================================

stats_rows = query("""
SELECT product_id::String AS product_id,
       countIf(event_type = 'view') AS views,
       countIf(event_type = 'add_to_cart') AS carts,
       countIf(event_type = 'purchase') AS purchases
FROM shop.history_events
GROUP BY product_id
""")


prod_stats = {}


for r in stats_rows:

    v = float(r["views"])
    c = float(r["carts"])
    p = float(r["purchases"])

    prod_stats[r["product_id"]] = {
        "views": v,
        "carts": c,
        "purchases": p,
        "conv_rate": (p + 1.0) / (v + 50.0),
        "cart_rate": (c + 1.0) / (v + 20.0),
    }


# =====================================================================
# 4. CO-COMPRAS
# =====================================================================

print("2/4 Calculando co-compras...")


co_rows = query("""
SELECT a.product_id AS visto,
       b.product_id AS recomendado,
       countIf(b.event_type = 'purchase') * 15
       + countIf(b.event_type = 'add_to_cart') * 4
       + countIf(b.event_type = 'view') AS score
FROM (
    SELECT session_id::String AS session_id,
           product_id::String AS product_id
    FROM shop.history_events
    WHERE event_type = 'view'
      AND ts > now() - INTERVAL 20 DAY
) AS a
INNER JOIN (
    SELECT session_id::String AS session_id,
           product_id::String AS product_id,
           event_type::String AS event_type
    FROM shop.history_events
    WHERE ts > now() - INTERVAL 20 DAY
) AS b
ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY visto, recomendado
ORDER BY score DESC
LIMIT 25 BY visto
""", timeout=60)


co_matrix = defaultdict(dict)


for r in co_rows:

    co_matrix[r["visto"]][r["recomendado"]] = (
        float(r["score"])
    )


# =====================================================================
# 5. HISTÓRICO DE SESIONES PARA ENTRENAMIENTO
# =====================================================================

print("3/4 Descargando sesiones de entrenamiento...")


sessions = query("""
SELECT recent_views,
       shown,
       propensity,
       clicked,
       purchased,
       context
FROM shop.history_sessions
WHERE clicked IS NOT NULL
   OR rand() % 8 = 0
LIMIT 45000
""", timeout=60)


# =====================================================================
# 6. MISMA FUNCIÓN DE FEATURES QUE USAREMOS EN PRODUCCIÓN
# =====================================================================

def build_features(
    cand_id: str,
    viewed_ids: list[str],
    slot: int,
) -> list[float]:

    c_info = catalog.get(
        cand_id,
        {}
    )

    st = prod_stats.get(
        cand_id,
        {
            "views": 0.0,
            "purchases": 0.0,
            "conv_rate": 0.01,
            "cart_rate": 0.02,
        }
    )

    c_price = float(
        c_info.get(
            "price_eur",
            25.0
        )
    )

    c_cat = c_info.get(
        "category",
        ""
    )

    c_subcat = c_info.get(
        "subcategory",
        ""
    )

    c_brand = c_info.get(
        "brand",
        ""
    )

    same_subcat = 0.0
    same_cat = 0.0
    same_brand = 0.0

    co_score_total = 0.0
    co_score_last = 0.0

    viewed_prices = []


    for idx, v_id in enumerate(
        reversed(viewed_ids)
    ):

        v_info = catalog.get(v_id)

        if not v_info:
            continue

        weight = (
            1.0 /
            (1.0 + 0.5 * idx)
        )

        if (
            v_info.get("subcategory")
            == c_subcat
        ):
            same_subcat += weight

        if (
            v_info.get("category")
            == c_cat
        ):
            same_cat += weight

        if (
            v_info.get("brand")
            == c_brand
        ):
            same_brand += weight

        pair_score = (
            co_matrix
            .get(v_id, {})
            .get(cand_id, 0.0)
        )

        co_score_total += (
            weight * pair_score
        )

        if idx == 0:
            co_score_last = pair_score

        viewed_prices.append(
            float(
                v_info.get(
                    "price_eur",
                    c_price
                )
            )
        )


    avg_session_price = (
        sum(viewed_prices)
        /
        len(viewed_prices)
        if viewed_prices
        else c_price
    )


    price_ratio = (
        c_price /
        max(avg_session_price, 1.0)
    )


    already_viewed = (
        1.0
        if cand_id in viewed_ids
        else 0.0
    )


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
        float(slot),
    ]


# =====================================================================
# 7. CREAR DATASET DE ENTRENAMIENTO
# =====================================================================

X_train = []
y_train = []
weights = []


for s in sessions:

    r_views = (
        s.get("recent_views")
        or []
    )

    viewed_ids = [
        v["product_id"]
        for v in r_views
        if (
            isinstance(v, dict)
            and "product_id" in v
        )
    ]


    shown = (
        s.get("shown")
        or []
    )

    props = (
        s.get("propensity")
        or [0.1] * len(shown)
    )

    clicked = s.get("clicked")
    purchased = s.get("purchased")


    for idx, cand_id in enumerate(
        shown
    ):

        slot = 1

        X_train.append(
            build_features(
                cand_id,
                viewed_ids,
                slot
            )
        )


        is_buy = (
            cand_id == purchased
        )

        is_click = (
            cand_id == clicked
        )


        y_train.append(
            1
            if (
                is_buy
                or is_click
            )
            else 0
        )


        prop = (
            float(props[idx])
            if (
                idx < len(props)
                and props[idx]
            )
            else 0.1
        )


        ips_w = min(
            1.0 /
            max(prop, 0.02),
            10.0
        )


        if is_buy:
            ips_w *= 4.0


        weights.append(
            ips_w
        )


X_np = np.array(
    X_train,
    dtype=np.float32
)

y_np = np.array(
    y_train,
    dtype=np.int32
)

w_np = np.array(
    weights,
    dtype=np.float32
)


print(
    f"Ejemplos de entrenamiento: {len(X_np)}"
)

print(
    f"Positivos: {int(y_np.sum())}"
)


# =====================================================================
# 8. ENTRENAMIENTO REGRESIÓN LOGÍSTICA
# =====================================================================

print(
    "4/4 Entrenando Regresión Logística..."
)


model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)


model.fit(
    X_np,
    y_np,
    sample_weight=w_np
)


# =====================================================================
# 9. GUARDAR MODELO Y FEATURES
# =====================================================================

joblib.dump(
    model,
    "modelo_logistico.pkl"
)


joblib.dump(
    {
        "catalog": catalog,
        "prod_stats": prod_stats,
        "co_matrix": dict(co_matrix),
    },
    "features_data.pkl"
)


print()
print(
    "¡ÉXITO!"
)

print(
    "modelo_logistico.pkl creado."
)

print(
    "features_data.pkl creado."
)