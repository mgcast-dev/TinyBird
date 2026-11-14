"""Entrenamiento de LightGBM sin fugas temporales y con las features compartidas.

    python training_boosting.py                   # modelo final, todos los datos
    python training_boosting.py --cut 2026-11-07  # modelo para evaluar offline (datos < corte)

Con --cut los ficheros llevan sufijo _cut: modelo_lgbm_cut.pkl y features_data_cut.pkl.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import requests

from features import N_FEATURES, build_feature_row, build_state, viewed_from_requests

parser = argparse.ArgumentParser()
parser.add_argument("--cut", default="", help="YYYY-MM-DD: usar solo datos anteriores")
parser.add_argument("--sessions", type=int, default=45000)
args = parser.parse_args()

CUT = args.cut
SUFFIX = "_cut" if CUT else ""
NOW = f"toDateTime('{CUT}')" if CUT else "now()"

# 1. Conexión a RawTree
for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"


def query(sql: str, timeout: int = 60) -> list[dict]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    r.raise_for_status()
    return r.json()["data"]


print(f"Corte temporal: {CUT or 'ninguno (modelo final)'}")

# 2. Catálogo y estadísticas, solo con datos anteriores al corte
print("1/4 Catálogo y estadísticas por producto...")
cat_rows = query("""
SELECT product_id::String AS product_id,
       category::String AS category,
       subcategory::String AS subcategory,
       brand::String AS brand,
       price_eur::Float64 AS price_eur
FROM shop.catalog
""")
catalog = {r["product_id"]: r for r in cat_rows}

stats_rows = query(f"""
SELECT product_id::String AS product_id,
       countIf(event_type = 'view') AS views,
       countIf(event_type = 'add_to_cart') AS carts,
       countIf(event_type = 'purchase') AS purchases
FROM shop.history_events
WHERE ts::DateTime64(3) < {NOW}
GROUP BY product_id
""")
prod_stats: dict[str, dict[str, float]] = {}
for r in stats_rows:
    v, c, p = float(r["views"]), float(r["carts"]), float(r["purchases"])
    prod_stats[r["product_id"]] = {
        "views": v, "carts": c, "purchases": p,
        "conv_rate": (p + 1.0) / (v + 50.0),
        "cart_rate": (c + 1.0) / (v + 20.0),
    }

# 3. Co-visitas y co-compras en los 20 días anteriores al corte
print("2/4 Pares de co-visitas y co-compras...")
co_rows = query(f"""
SELECT a.product_id AS visto,
       b.product_id AS recomendado,
       countIf(b.event_type = 'purchase') * 15
       + countIf(b.event_type = 'add_to_cart') * 4
       + countIf(b.event_type = 'view') AS score
FROM (
    SELECT session_id::String AS session_id, product_id::String AS product_id
    FROM shop.history_events
    WHERE event_type = 'view'
      AND ts::DateTime64(3) < {NOW}
      AND ts::DateTime64(3) > {NOW} - INTERVAL 20 DAY
) AS a
INNER JOIN (
    SELECT session_id::String AS session_id,
           product_id::String AS product_id,
           event_type::String AS event_type
    FROM shop.history_events
    WHERE ts::DateTime64(3) < {NOW}
      AND ts::DateTime64(3) > {NOW} - INTERVAL 20 DAY
) AS b ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY visto, recomendado
ORDER BY score DESC
LIMIT 25 BY visto
""", timeout=120)

co_matrix: dict[str, dict[str, float]] = defaultdict(dict)
for r in co_rows:
    co_matrix[r["visto"]][r["recomendado"]] = float(r["score"])
co_matrix = dict(co_matrix)

# 4. Sesiones de entrenamiento (con clic, más una muestra de negativos)
print("3/4 Sesiones de shop.history_sessions...")
sessions = query(f"""
SELECT recent_views, shown, propensity, clicked, purchased
FROM shop.history_sessions
WHERE ts::DateTime64(3) < {NOW}
  AND ((clicked IS NOT NULL AND clicked != '') OR rand() % 8 = 0)
LIMIT {args.sessions}
""", timeout=120)

# 5. Ejemplos: una fila por producto mostrado, con las features compartidas
state = build_state(catalog, prod_stats, co_matrix)

X: list[list[float]] = []
y: list[int] = []
w: list[float] = []
for s in sessions:
    viewed = viewed_from_requests(s.get("recent_views"))
    shown = s.get("shown") or []
    props = s.get("propensity") or []
    clicked = s.get("clicked") or None
    purchased = s.get("purchased") or None

    for idx, cand in enumerate(shown):
        X.append(build_feature_row(state, cand, viewed))
        is_buy = cand == purchased
        is_click = cand == clicked
        y.append(1 if (is_buy or is_click) else 0)

        # IPS (corrige el sesgo del sistema antiguo) y compra valorada 4x el clic
        prop = float(props[idx]) if idx < len(props) and props[idx] else 0.1
        ips = min(1.0 / max(prop, 0.02), 10.0)
        w.append(ips * (4.0 if is_buy else 1.0))

X_np = np.asarray(X, dtype=np.float32)
y_np = np.asarray(y, dtype=np.int32)
w_np = np.asarray(w, dtype=np.float32)
assert X_np.shape[1] == N_FEATURES, "número de features distinto al de features.py"
print(f"Ejemplos: {len(y_np)}  positivos: {y_np.mean():.4f}")

# 6. Entrenamiento y guardado
print("4/4 Entrenando LightGBM...")
model = lgb.LGBMClassifier(
    n_estimators=180,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_np, y_np, sample_weight=w_np)

joblib.dump(model, f"modelo_lgbm{SUFFIX}.pkl")
joblib.dump({
    "catalog": catalog,
    "prod_stats": prod_stats,
    "co_matrix": co_matrix,
    "n_features": N_FEATURES,
}, f"features_data{SUFFIX}.pkl")

print(f"¡Listo! modelo_lgbm{SUFFIX}.pkl y features_data{SUFFIX}.pkl guardados.")
