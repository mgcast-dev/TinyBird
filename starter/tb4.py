"""Training Boosting 4 (17 features + money weighting)"""

import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import requests
import lightgbm as lgb

for line in (Path(".env").read_text().splitlines() if Path(".env").exists() else []):
    key, _, value = line.partition("=")
    if key.strip():
        os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
API_KEY = os.environ.get("RAWTREE_API_KEY")

if not API_KEY:
    raise RuntimeError("Falta RAWTREE_API_KEY en .env")

http = requests.Session()
http.headers["Authorization"] = f"Bearer {API_KEY}"

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "modelo_lgbm8.pkl"
FEATURES_PATH = MODELS_DIR / "features_data8.pkl"

def query(sql: str, timeout: int = 45):
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    r.raise_for_status()
    return r.json()["data"]

print("1/6 Descargando catálogo...")
cat_rows = query("""
SELECT product_id::String AS product_id, category::String AS category,
       subcategory::String AS subcategory, brand::String AS brand,
       price_eur::Float64 AS price_eur, created_at
FROM shop.catalog
""")
catalog = {r["product_id"]: r for r in cat_rows}
print(f"   Productos en catálogo: {len(catalog)}")

print("2/6 Calculando estadísticas de productos...")
stats_rows = query("""
SELECT product_id::String AS product_id,
    countIf(event_type = 'view') AS views,
    countIf(event_type = 'add_to_cart') AS carts,
    countIf(event_type = 'purchase') AS purchases,
    countIf(event_type = 'view' AND ts >= (SELECT max(ts) - INTERVAL 24 HOUR FROM shop.history_events)) AS views_24h,
    countIf(event_type = 'add_to_cart' AND ts >= (SELECT max(ts) - INTERVAL 24 HOUR FROM shop.history_events)) AS carts_24h,
    countIf(event_type = 'purchase' AND ts >= (SELECT max(ts) - INTERVAL 24 HOUR FROM shop.history_events)) AS purchases_24h
FROM shop.history_events
GROUP BY product_id
""")

prod_stats = {}
for r in stats_rows:
    v, c, p = float(r.get("views", 0)), float(r.get("carts", 0)), float(r.get("purchases", 0))
    v24, c24, p24 = float(r.get("views_24h", 0)), float(r.get("carts_24h", 0)), float(r.get("purchases_24h", 0))

    conv_rate = (p + 1.0) / (v + 50.0)
    cart_rate = (c + 1.0) / (v + 20.0)
    recent_conv_rate = (p24 + 1.0) / (v24 + 20.0)
    recent_popularity = (p24 * 10.0 + c24 * 3.0 + v24)

    prod_stats[r["product_id"]] = {
        "views": v, "carts": c, "purchases": p,
        "conv_rate": conv_rate, "cart_rate": cart_rate,
        "views_24h": v24, "carts_24h": c24, "purchases_24h": p24,
        "recent_conv_rate": recent_conv_rate, "recent_popularity": recent_popularity,
    }

print("3/6 Calculando edad de los productos...")
now_ts = datetime.now(timezone.utc)
DEFAULT_STATS = {
    "views": 0.0, "carts": 0.0, "purchases": 0.0,
    "conv_rate": 0.01, "cart_rate": 0.02,
    "views_24h": 0.0, "carts_24h": 0.0, "purchases_24h": 0.0,
    "recent_conv_rate": 0.01, "recent_popularity": 0.0,
}

for product_id, info in catalog.items():
    created_at = info.get("created_at")
    age_days = 0.0
    if created_at:
        try:
            if isinstance(created_at, str):
                created_str = created_at.replace("Z", "+00:00")
                created_dt = datetime.fromisoformat(created_str)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (now_ts - created_dt).total_seconds() / 86400.0)
        except Exception:
            age_days = 0.0

    if product_id not in prod_stats:
        prod_stats[product_id] = DEFAULT_STATS.copy()
    prod_stats[product_id]["product_age_days"] = age_days

print("4/6 Calculando pares de Co-Compras...")
co_rows = query("""
SELECT a.product_id AS visto, b.product_id AS recomendado,
    countIf(b.event_type = 'purchase') * 15 + countIf(b.event_type = 'add_to_cart') * 4 + countIf(b.event_type = 'view') AS score
FROM (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events WHERE event_type = 'view' AND ts > now() - INTERVAL 20 DAY) AS a
INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id, event_type::String AS event_type FROM shop.history_events WHERE ts > now() - INTERVAL 20 DAY) AS b
ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY visto, recomendado ORDER BY score DESC LIMIT 25 BY visto
""", timeout=60)

co_matrix = defaultdict(dict)
for r in co_rows:
    co_matrix[r["visto"]][r["recomendado"]] = float(r["score"])

print("5/6 Descargando sesiones de entrenamiento...")
sessions = query("""
SELECT recent_views, shown, propensity, clicked, purchased, context
FROM shop.history_sessions
WHERE clicked IS NOT NULL OR cityHash64(request_id) % 8 = 0
ORDER BY cityHash64(request_id)
LIMIT 45000
""", timeout=45)
print(f"   Sesiones descargadas: {len(sessions)}")

def build_features(cand_id: str, viewed_ids: list[str], slot: int) -> list[float]:
    c_info = catalog.get(cand_id, {})
    st = prod_stats.get(cand_id, DEFAULT_STATS)
    c_price = float(c_info.get("price_eur", 25.0))
    c_cat = c_info.get("category", "")
    c_subcat = c_info.get("subcategory", "")
    c_brand = c_info.get("brand", "")

    same_subcat, same_cat, same_brand = 0.0, 0.0, 0.0
    co_score_total, co_score_last = 0.0, 0.0
    viewed_prices = []

    for idx, v_id in enumerate(reversed(viewed_ids)):
        v_info = catalog.get(v_id)
        if not v_info:
            continue
        weight = 1.0 / (1.0 + 0.5 * idx)
        if v_info.get("subcategory") == c_subcat: same_subcat += weight
        if v_info.get("category") == c_cat: same_cat += weight
        if v_info.get("brand") == c_brand: same_brand += weight

        pair_score = co_matrix.get(v_id, {}).get(cand_id, 0.0)
        co_score_total += weight * pair_score
        if idx == 0: co_score_last = pair_score
        viewed_prices.append(float(v_info.get("price_eur", c_price)))

    avg_session_price = sum(viewed_prices) / len(viewed_prices) if viewed_prices else c_price
    price_ratio = c_price / max(avg_session_price, 1.0)
    already_viewed = 1.0 if cand_id in viewed_ids else 0.0

    return [
        float(st["purchases"]), float(st["conv_rate"]), float(st["cart_rate"]),
        c_price, price_ratio, same_subcat, same_cat, same_brand,
        co_score_last, co_score_total, already_viewed, float(slot),
        float(st["purchases_24h"]), float(st["views_24h"]), float(st["carts_24h"]),
        float(st["product_age_days"]), float(st["recent_popularity"]),
    ]

print("   Construyendo dataset...")
X_train, y_train, weights = [], [], []

for s in sessions:
    r_views = (s.get("recent_views") or [])
    viewed_ids = [v["product_id"] for v in r_views if isinstance(v, dict) and "product_id" in v]
    shown = (s.get("shown") or [])
    props = (s.get("propensity") or [0.1] * len(shown))
    clicked = s.get("clicked")
    purchased = s.get("purchased")

    for idx, cand_id in enumerate(shown):
        slot = 1
        X_train.append(build_features(cand_id, viewed_ids, slot))
        is_buy = (cand_id == purchased)
        is_click = (cand_id == clicked)
        y_train.append(1 if (is_buy or is_click) else 0)

        prop = float(props[idx]) if idx < len(props) and props[idx] else 0.1
        ips_w = min(1.0 / max(prop, 0.02), 10.0)
        if is_buy: ips_w *= 4.0
        weights.append(ips_w)

X_np = np.array(X_train, dtype=np.float32)
y_np = np.array(y_train, dtype=np.int32)
w_np = np.array(weights, dtype=np.float32)

if X_np.ndim != 2 or X_np.shape[1] != 17:
    raise RuntimeError(f"ERROR: el dataset no tiene 17 features. Tiene {X_np.shape}")

print(f"Dataset creado: {len(X_np)} ejemplos x {X_np.shape[1]} features")

print("6/6 Entrenando LightGBM...")
model = lgb.LGBMClassifier(
    n_estimators=180,
    learning_rate=0.05,
    num_leaves=31,
    subsample_for_bin=50000,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_np, y_np, sample_weight=w_np)

if not hasattr(model, "n_features_in_") or model.n_features_in_ != 17:
    raise RuntimeError("ERROR: el modelo no espera 17 features")

EXPECTED_FEATURE_NAMES = [
    "purchases", "conv_rate", "cart_rate", "price", "price_ratio",
    "same_subcat", "same_cat", "same_brand", "co_score_last", "co_score_total",
    "already_viewed", "slot", "purchases_24h", "views_24h", "carts_24h",
    "product_age_days", "recent_popularity",
]

joblib.dump(model, MODEL_PATH)
joblib.dump({
    "catalog": catalog,
    "prod_stats": prod_stats,
    "co_matrix": dict(co_matrix),
    "feature_names": EXPECTED_FEATURE_NAMES,
}, FEATURES_PATH)

print("=" * 60)
print("ENTRENAMIENTO 4 TERMINADO")
print("=" * 60)
print(f"Modelo: {MODEL_PATH}")
print(f"Datos:  {FEATURES_PATH}")
print(f"Features: {X_np.shape[1]}")
print(f"Ejemplos: {len(X_np)}")
print("¡Listo!")