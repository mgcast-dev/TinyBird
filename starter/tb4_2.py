"""Entrenamiento de LightGBM sin fugas temporales y con las features compartidas.

    python tb4_3.py                   # modelo final, todos los datos
    python tb4_3.py --cut 2026-11-07  # modelo para evaluar offline (datos < corte)

Con --cut los ficheros llevan sufijo _v12_cut. Sin --cut, sufijo _v12.
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
parser.add_argument("--sessions", type=int, default=200000)
args = parser.parse_args()

CUT = args.cut
SUFFIX = "_v12_cut" if CUT else "_v12"
NOW = f"toDateTime('{CUT}')" if CUT else "now()"

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

print("3/4 Sesiones de shop.history_sessions...")
sessions = query(f"""
SELECT recent_views, shown, propensity, clicked, purchased
FROM shop.history_sessions
WHERE ts::DateTime64(3) < {NOW}
  AND ((clicked IS NOT NULL AND clicked != '')
       OR (purchased IS NOT NULL AND purchased != '')
       OR rand() % 8 = 0)
LIMIT {args.sessions}
""", timeout=180)

state = build_state(catalog, prod_stats, co_matrix)

X: list[list[float]] = []
y_click: list[int] = []
y_buy: list[int] = []
w_click: list[float] = []
w_buy: list[float] = []
for s in sessions:
    viewed = viewed_from_requests(s.get("recent_views"))
    shown = s.get("shown") or []
    props = s.get("propensity") or []
    clicked = s.get("clicked") or None
    purchased = s.get("purchased") or None
    sampled_negative = 8.0 if (clicked is None and purchased is None) else 1.0

    for idx, cand in enumerate(shown):
        X.append(build_feature_row(state, cand, viewed))
        y_click.append(1 if cand == clicked else 0)
        y_buy.append(1 if cand == purchased else 0)

        prop = float(props[idx]) if idx < len(props) and props[idx] else 0.1
        ips = min(1.0 / max(prop, 0.02), 10.0) * sampled_negative
        w_click.append(ips)
        w_buy.append(ips)

X_np = np.asarray(X, dtype=np.float32)
y_click_np = np.asarray(y_click, dtype=np.int32)
y_buy_np = np.asarray(y_buy, dtype=np.int32)
w_click_np = np.asarray(w_click, dtype=np.float32)
w_buy_np = np.asarray(w_buy, dtype=np.float32)
assert X_np.shape[1] == N_FEATURES, "número de features distinto al de features.py"
print(f"Ejemplos: {len(X_np)}  clics: {y_click_np.mean():.4f}  compras: {y_buy_np.mean():.5f}")

print("4/4 Entrenando LightGBM (clic y compra)...")
params = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
click_model = lgb.LGBMClassifier(**params)
click_model.fit(X_np, y_click_np, sample_weight=w_click_np)
buy_model = lgb.LGBMClassifier(**params)
buy_model.fit(X_np, y_buy_np, sample_weight=w_buy_np)

joblib.dump({"click": click_model, "buy": buy_model}, f"modelo_lgbm{SUFFIX}.pkl")
joblib.dump({
    "catalog": catalog,
    "prod_stats": prod_stats,
    "co_matrix": co_matrix,
    "n_features": N_FEATURES,
}, f"features_data{SUFFIX}.pkl")

print(f"¡Listo! modelo_lgbm{SUFFIX}.pkl y features_data{SUFFIX}.pkl guardados.")