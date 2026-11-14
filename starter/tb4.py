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
MODEL_PATH = MODELS_DIR / "modelo_lgbm_v9.pkl"
FEATURES_PATH = MODELS_DIR / "features_data_v9.pkl"
K_PRIOR = 50.0

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

print("2/6 Estadísticas de productos...")
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

raw = {}
sub_p = defaultdict(float)
sub_v = defaultdict(float)
for r in stats_rows:
    pid = r["product_id"]
    rec = {k: float(r.get(k, 0) or 0) for k in
           ("views", "carts", "purchases", "views_24h", "carts_24h", "purchases_24h")}
    raw[pid] = rec
    sc = catalog.get(pid, {}).get("subcategory", "")
    sub_p[sc] += rec["purchases"]
    sub_v[sc] += rec["views"]

SUB_CONV = {sc: (sub_p[sc] + 1.0) / (sub_v[sc] + K_PRIOR) for sc in sub_p}

def default_stats(pid: str) -> dict:
    prior = SUB_CONV.get(catalog.get(pid, {}).get("subcategory", ""), 0.01)
    return {
        "views": 0.0, "carts": 0.0, "purchases": 0.0,
        "conv_rate": prior, "cart_rate": 0.02,
        "views_24h": 0.0, "carts_24h": 0.0, "purchases_24h": 0.0,
        "recent_conv_rate": prior, "recent_popularity": 0.0,
        "product_age_days": 0.0,
    }

prod_stats = {}
for pid, rec in raw.items():
    v, c, p = rec["views"], rec["carts"], rec["purchases"]
    v24, c24, p24 = rec["views_24h"], rec["carts_24h"], rec["purchases_24h"]
    prior = SUB_CONV.get(catalog.get(pid, {}).get("subcategory", ""), 0.01)
    prod_stats[pid] = {
        "views": v, "carts": c, "purchases": p,
        "conv_rate": (p + K_PRIOR * prior) / (v + K_PRIOR),
        "cart_rate": (c + 1.0) / (v + 20.0),
        "views_24h": v24, "carts_24h": c24, "purchases_24h": p24,
        "recent_conv_rate": (p24 + 1.0) / (v24 + 20.0),
        "recent_popularity": p24 * 10.0 + c24 * 3.0 + v24,
        "product_age_days": 0.0,
    }

print("3/6 Edad de los productos...")
now_ts = datetime.now(timezone.utc)
for pid, info in catalog.items():
    age = 0.0
    created_at = info.get("created_at")
    if created_at and isinstance(created_at, str):
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age = max(0.0, (now_ts - dt).total_seconds() / 86400.0)
        except Exception:
            age = 0.0
    if pid not in prod_stats:
        prod_stats[pid] = default_stats(pid)
    prod_stats[pid]["product_age_days"] = age

print("4/6 Co-compras...")
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

print("5/6 Sesiones de entrenamiento...")
sessions = query("""
SELECT recent_views, shown, propensity, clicked, purchased
FROM shop.history_sessions
WHERE (clicked IS NOT NULL AND clicked != '') OR rand() % 8 = 0
LIMIT 90000
""", timeout=120)
print(f"   Sesiones: {len(sessions)}")

def build_features(cand_id: str, viewed_ids: list[str], slot: int) -> list[float]:
    c_info = catalog.get(cand_id, {})
    st = prod_stats.get(cand_id) or default_stats(cand_id)
    c_price = float(c_info.get("price_eur", 25.0))
    c_cat = c_info.get("category", "")
    c_subcat = c_info.get("subcategory", "")
    c_brand = c_info.get("brand", "")

    same_subcat = same_cat = same_brand = 0.0
    co_score_total = co_score_last = 0.0
    viewed_prices = []

    for idx, v_id in enumerate(reversed(viewed_ids)):
        v_info = catalog.get(v_id)
        if not v_info:
            continue
        weight = 1.0 / (1.0 + 0.5 * idx)
        if v_info.get("subcategory") == c_subcat:
            same_subcat += weight
        if v_info.get("category") == c_cat:
            same_cat += weight
        if v_info.get("brand") == c_brand:
            same_brand += weight
        pair_score = co_matrix.get(v_id, {}).get(cand_id, 0.0)
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
        co_score_last, co_score_total, already_viewed, float(slot),
        float(st["purchases_24h"]), float(st["views_24h"]), float(st["carts_24h"]),
        float(st.get("product_age_days", 0.0)), float(st["recent_popularity"]),
    ]

print("6/6 Construyendo dataset y entrenando...")
X_train, y_train, weights = [], [], []
for s in sessions:
    viewed_ids = [v["product_id"] for v in (s.get("recent_views") or [])
                  if isinstance(v, dict) and "product_id" in v]
    shown = s.get("shown") or []
    props = s.get("propensity") or [0.1] * len(shown)
    clicked = s.get("clicked") or None
    purchased = s.get("purchased") or None

    for idx, cand_id in enumerate(shown):
        X_train.append(build_features(cand_id, viewed_ids, idx + 1))
        is_buy = cand_id == purchased
        is_click = cand_id == clicked
        y_train.append(1 if (is_buy or is_click) else 0)
        prop = float(props[idx]) if idx < len(props) and props[idx] else 0.1
        ips_w = min(1.0 / max(prop, 0.02), 10.0)
        if is_buy:
            ips_w *= 4.0
        weights.append(ips_w)

X_np = np.array(X_train, dtype=np.float32)
y_np = np.array(y_train, dtype=np.int32)
w_np = np.array(weights, dtype=np.float32)

if X_np.ndim != 2 or X_np.shape[1] != 17:
    raise RuntimeError(f"ERROR: dataset con shape {X_np.shape}, se esperaban 17 features")

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.04,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_np, y_np, sample_weight=w_np)

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
    "sub_conv": SUB_CONV,
    "feature_names": EXPECTED_FEATURE_NAMES,
}, FEATURES_PATH)

print("ENTRENAMIENTO TERMINADO")
print(f"Ejemplos: {len(X_np)} | Features: {X_np.shape[1]}")
print(f"Modelo: {MODEL_PATH}")
print(f"Datos:  {FEATURES_PATH}")