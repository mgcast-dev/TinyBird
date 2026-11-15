"""Entrena LightGBM v8 con las 22 features activas (Point-in-Time sin truncamiento de eventos)."""
from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
STARTER = Path(__file__).resolve().parent
MODELS = STARTER / "models"
MODELS.mkdir(exist_ok=True)
MODEL_PATH, DATA_PATH = MODELS / "modelo_lgbm8.pkl", MODELS / "features_data4.pkl"

FEATURES = [
    "purchases", "conv_rate", "cart_rate", "price", "price_ratio", "same_subcat",
    "same_cat", "same_brand", "co_score_last", "co_score_total", "already_viewed",
    "slot", "purchases_24h", "views_24h", "carts_24h", "product_age_days",
    "recent_popularity", "user_subcat_share", "user_brand_share", "user_price_ratio",
    "user_past_item_status", "ctx_subcat_affinity"
]
EVENT_WEIGHT = {"view": 1.0, "add_to_cart": 2.0, "purchase": 5.0}

for env_file in (ROOT / ".env", STARTER / ".env", Path.cwd() / ".env"):
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com").rstrip("/")
API_KEY = os.environ.get("RAWTREE_API_KEY")
if not API_KEY:
    raise RuntimeError("Falta RAWTREE_API_KEY en .env")
HTTP = requests.Session()
HTTP.headers["Authorization"] = f"Bearer {API_KEY}"


def query(sql: str, timeout: int = 180) -> list[dict[str, Any]]:
    response = HTTP.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    if not response.ok:
        print("\n[ERROR SQL DE RAWTREE]:", response.text)
    response.raise_for_status()
    return response.json().get("data", [])


def parse_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def unpack(value: Any, expected: type) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return expected()
    return value if isinstance(value, expected) else expected()


print("1/6 Descargando catálogo...")
catalog_rows = query("""
SELECT product_id::String AS product_id, category::String AS category,
       subcategory::String AS subcategory, brand::String AS brand,
       price_eur::Float64 AS price_eur, created_at
FROM shop.catalog
""")
catalog = {str(r["product_id"]): r for r in catalog_rows}

print("2/6 Descargando sesiones y eventos recientes...")
# Traemos las sesiones más recientes directamente ordenadas sin filtros de fecha frágiles
sessions = query("""
SELECT request_id::String AS request_id, user_id::String AS user_id, ts,
       context, recent_views, shown, propensity, clicked, purchased
FROM shop.history_sessions
WHERE clicked IS NOT NULL OR purchased IS NOT NULL OR rand() % 8 = 0
ORDER BY ts DESC
LIMIT 45000
""")
sessions.sort(key=lambda r: parse_ts(r.get("ts")) or datetime.min.replace(tzinfo=timezone.utc))

# Descargamos los eventos agrupados más recientes
events = query("""
SELECT
    toStartOfHour(ts) AS ts,
    event_type::String AS event_type,
    user_id::String AS user_id,
    product_id::String AS product_id,
    count() AS cnt
FROM shop.history_events
WHERE event_type IN ('purchase', 'add_to_cart', 'view')
GROUP BY ts, event_type, user_id, product_id
ORDER BY ts DESC
LIMIT 250000
""", timeout=240)
events.sort(key=lambda r: parse_ts(r.get("ts")) or datetime.min.replace(tzinfo=timezone.utc))
print(f"   -> {len(sessions)} sesiones y {len(events)} bloques de eventos cargados correctamente.")

print("3/6 Calculando agregados base de producto, tendencias y co-compras...")
base_stats_rows = query("""
SELECT product_id::String AS product_id,
 countIf(event_type='view') AS views,
 countIf(event_type='add_to_cart') AS carts,
 countIf(event_type='purchase') AS purchases,
 countIf(event_type='view' AND ts >= now() - INTERVAL 24 HOUR) AS views_24h,
 countIf(event_type='add_to_cart' AND ts >= now() - INTERVAL 24 HOUR) AS carts_24h,
 countIf(event_type='purchase' AND ts >= now() - INTERVAL 24 HOUR) AS purchases_24h
FROM shop.history_events 
GROUP BY product_id
""")

prod_stats: dict[str, dict[str, float]] = {}
pit_stats: dict[str, dict[str, float]] = {}

for r in base_stats_rows:
    pid = str(r["product_id"])
    v, c, p = (float(r.get(k, 0) or 0) for k in ("views", "carts", "purchases"))
    v24, c24, p24 = (float(r.get(k, 0) or 0) for k in ("views_24h", "carts_24h", "purchases_24h"))
    prod_stats[pid] = {
        "views": v, "carts": c, "purchases": p,
        "conv_rate": (p + 1.) / (v + 50.), "cart_rate": (c + 1.) / (v + 20.),
        "base_purchases": p, "base_conv": (p + 1.) / (v + 50.), "base_cart": (c + 1.) / (v + 20.),
        "views_24h": v24, "carts_24h": c24, "purchases_24h": p24,
        "recent_conv_rate": (p24 + 1.) / (v24 + 20.), "recent_popularity": p24 * 10. + c24 * 3. + v24
    }
    pit_stats[pid] = {
        "views": v * 0.5, "carts": c * 0.5, "purchases": p * 0.5,
        "conv_rate": (p + 1.) / (v + 50.), "cart_rate": (c + 1.) / (v + 20.),
        "views_24h": 0., "carts_24h": 0., "purchases_24h": 0., "recent_popularity": 0.
    }

defaults = {
    "views": 0., "carts": 0., "purchases": 0., "conv_rate": .01, "cart_rate": .02,
    "base_purchases": 0., "base_conv": .01, "base_cart": .02, "views_24h": 0.,
    "carts_24h": 0., "purchases_24h": 0., "recent_conv_rate": .01,
    "recent_popularity": 0., "product_age_days": 0.
}
for pid, product in catalog.items():
    st = prod_stats.setdefault(pid, defaults.copy())
    pit_stats.setdefault(pid, {
        "views": 0., "carts": 0., "purchases": 0., "conv_rate": .01, "cart_rate": .02,
        "views_24h": 0., "carts_24h": 0., "purchases_24h": 0., "recent_popularity": 0.
    })
    created = parse_ts(product.get("created_at"))
    st["product_age_days"] = max(0., (datetime.now(timezone.utc) - created).total_seconds() / 86400) if created else 0.

co_rows = query("""
SELECT
    a.product_id AS visto,
    b.product_id AS recomendado,
    countIf(b.event_type = 'purchase') * 15
    + countIf(b.event_type = 'add_to_cart') * 4
    + countIf(b.event_type = 'view') AS score
FROM (
    SELECT session_id::String AS session_id, product_id::String AS product_id
    FROM shop.history_events
    WHERE event_type = 'view' AND ts > now() - INTERVAL 25 DAY
) AS a
INNER JOIN (
    SELECT session_id::String AS session_id, product_id::String AS product_id, event_type::String AS event_type
    FROM shop.history_events
    WHERE ts > now() - INTERVAL 25 DAY
) AS b ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY visto, recomendado
ORDER BY score DESC
LIMIT 25 BY visto
""", timeout=240)
co_matrix: dict[str, dict[str, float]] = defaultdict(dict)
for r in co_rows:
    co_matrix[str(r["visto"])][str(r["recomendado"])] = float(r.get("score", 0) or 0)

user_profiles: dict[str, dict[str, Any]] = {}
ctx_prices: dict[str, list[float]] = defaultdict(lambda: [0., 0.])
ctx_subcats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
ctx_totals: dict[str, float] = defaultdict(float)


def ctx_keys(context: dict[str, Any]) -> list[str]:
    region, device = str(context.get("region", "")), str(context.get("device", ""))
    return [f"{region}|{device}", f"{region}|", f"|{device}"]


def new_profile() -> dict[str, Any]:
    return {"subcat_weights": {}, "brand_weights": {}, "price_sum": 0., "price_weight": 0.,
            "items": {}, "event_count": 0}


def accumulate(uid: str, pid: str, kind: str, cnt: float = 1.0) -> None:
    if not uid or not pid:
        return
    weight = EVENT_WEIGHT.get(kind, 1.) * cnt
    product = catalog.get(pid, {})
    subcat, brand = str(product.get("subcategory") or ""), str(product.get("brand") or "")
    price = float(product.get("price_eur") or 0.)
    profile = user_profiles.setdefault(uid, new_profile())
    if subcat:
        profile["subcat_weights"][subcat] = profile["subcat_weights"].get(subcat, 0.) + weight
    if brand:
        profile["brand_weights"][brand] = profile["brand_weights"].get(brand, 0.) + weight
    if price > 0:
        profile["price_sum"] += price * weight
        profile["price_weight"] += weight
    profile["event_count"] += int(cnt)
    state = profile["items"].setdefault(pid, {"view": False, "cart": False, "purchase": False})
    state_key = {"view": "view", "add_to_cart": "cart", "purchase": "purchase"}.get(kind)
    if state_key:
        state[state_key] = True


def accumulate_session_context(context: dict[str, Any], target_pid: str, weight: float) -> None:
    product = catalog.get(target_pid, {})
    subcat = str(product.get("subcategory") or "")
    price = float(product.get("price_eur") or 0.)
    for key in ctx_keys(context):
        if price > 0:
            ctx_prices[key][0] += price
            ctx_prices[key][1] += 1.0
        if subcat:
            ctx_subcats[key][subcat] += weight
            ctx_totals[key] += weight


def past_status(profile: dict[str, Any], pid: str) -> float:
    state = profile.get("items", {}).get(pid, {})
    if state.get("purchase"):
        return -1.
    if state.get("cart"):
        return 2.
    return 1. if state.get("view") else 0.


def ref_price(profile: dict[str, Any], context: dict[str, Any]) -> float:
    if profile.get("price_weight", 0) > 0:
        return profile["price_sum"] / profile["price_weight"]
    for key in ctx_keys(context):
        data = ctx_prices.get(key)
        if data and data[1]:
            return data[0] / data[1]
    return 25.


def build_features(pid: str, viewed: list[str], slot: int, session_ts: datetime,
                   profile: dict[str, Any], context: dict[str, Any],
                   stats_override: dict[str, dict[str, float]]) -> list[float]:
    info, st = catalog.get(pid, {}), stats_override.get(pid, defaults)
    price = float(info.get("price_eur") or 25.)
    subcat, category, brand = (str(info.get(k) or "") for k in ("subcategory", "category", "brand"))
    ssub = scat = sbrand = colast = cototal = 0.
    view_prices = []
    for index, vid in enumerate(reversed(viewed)):
        vi = catalog.get(vid)
        if not vi:
            continue
        weight = 1. / (1. + .5 * index)
        ssub += weight if vi.get("subcategory") == subcat else 0.
        scat += weight if vi.get("category") == category else 0.
        sbrand += weight if vi.get("brand") == brand else 0.
        pair = co_matrix.get(vid, {}).get(pid, 0.)
        cototal += weight * pair
        if index == 0:
            colast = pair
        view_prices.append(float(vi.get("price_eur") or price))

    reference = ref_price(profile, context)
    price_ratio = price / max(sum(view_prices) / len(view_prices) if view_prices else reference, 1.)
    sw = sum(profile.get("subcat_weights", {}).values())
    bw = sum(profile.get("brand_weights", {}).values())
    usub = profile.get("subcat_weights", {}).get(subcat, 0.) / sw if sw else 0.
    ubrand = profile.get("brand_weights", {}).get(brand, 0.) / bw if bw else 0.
    context_price = ref_price({}, context)
    user_ratio = price / max(reference if profile.get("price_weight", 0.) else context_price, 1.)

    ctx_affinity = 0.
    for key in ctx_keys(context):
        denom = ctx_totals.get(key, 0.)
        if denom:
            ctx_affinity = min(1., ctx_subcats[key].get(subcat, 0.) / denom)
            break

    created = parse_ts(info.get("created_at"))
    age = max(0., (session_ts - created).total_seconds() / 86400.) if created else 0.

    return [
        float(st.get("purchases", 0)), float(st.get("conv_rate", .01)), float(st.get("cart_rate", .02)),
        price, price_ratio, ssub, scat, sbrand, colast, cototal, float(pid in viewed),
        float(slot),
        float(st.get("purchases_24h", 0)), float(st.get("views_24h", 0)), float(st.get("carts_24h", 0)),
        age, float(st.get("recent_popularity", 0)), usub, ubrand, user_ratio,
        past_status(profile, pid), ctx_affinity
    ]


print("4/6 Construyendo filas con eventos estrictamente anteriores a ts - 5 min...")
X, y, sample_weights = [], [], []
event_index = 0
recent_events: deque[tuple[float, str, str, float]] = deque()


def advance_pit_event(event: dict[str, Any]) -> None:
    pid, kind = str(event.get("product_id") or ""), str(event.get("event_type") or "view")
    uid = str(event.get("user_id") or "")
    cnt = float(event.get("cnt", 1.0) or 1.0)
    ts = parse_ts(event.get("ts"))
    if not pid or not ts:
        return
    # Si es vistas muestreadas (rand() % 3 = 0), multiplicamos x3 para mantener la escala real
    real_cnt = cnt * 3.0 if kind == "view" else cnt
    accumulate(uid, pid, kind, cnt)
    key = {"view": "views", "add_to_cart": "carts", "purchase": "purchases"}.get(kind)
    if not key:
        return
    state = pit_stats.setdefault(pid, {
        "views": 0., "carts": 0., "purchases": 0., "conv_rate": .01, "cart_rate": .02,
        "views_24h": 0., "carts_24h": 0., "purchases_24h": 0., "recent_popularity": 0.
    })
    state[key] += real_cnt
    state[f"{key}_24h"] += real_cnt
    state["recent_popularity"] += {"views": 1., "carts": 3., "purchases": 10.}[key] * real_cnt
    state["conv_rate"] = (state["purchases"] + 1.) / (state["views"] + 50.)
    state["cart_rate"] = (state["carts"] + 1.) / (state["views"] + 20.)
    recent_events.append((ts.timestamp(), pid, key, real_cnt))


def expire_pit_events(session_ts: datetime) -> None:
    threshold = session_ts.timestamp() - 300. - 86400.
    while recent_events and recent_events[0][0] < threshold:
        _, pid, key, real_cnt = recent_events.popleft()
        pit_stats[pid][f"{key}_24h"] = max(0., pit_stats[pid][f"{key}_24h"] - real_cnt)
        pit_stats[pid]["recent_popularity"] = max(0., pit_stats[pid]["recent_popularity"] - {"views": 1., "carts": 3., "purchases": 10.}[key] * real_cnt)


for session in sessions:
    ts = parse_ts(session.get("ts"))
    if not ts:
        continue
    cutoff = ts.timestamp() - 300.
    while event_index < len(events):
        event_ts = parse_ts(events[event_index].get("ts"))
        if event_ts is None or event_ts.timestamp() >= cutoff:
            break
        advance_pit_event(events[event_index])
        event_index += 1
    expire_pit_events(ts)

    uid = str(session.get("user_id") or "")
    profile = user_profiles.get(uid, new_profile())
    context = unpack(session.get("context"), dict)
    recent = unpack(session.get("recent_views"), list)
    viewed = [str(v["product_id"]) for v in recent if isinstance(v, dict) and v.get("product_id")]
    shown = [str(pid) for pid in unpack(session.get("shown"), list) if pid]
    props = unpack(session.get("propensity"), list)
    clicked, purchased = str(session.get("clicked") or ""), str(session.get("purchased") or "")

    for idx, pid in enumerate(shown):
        slot = idx + 1
        X.append(build_features(pid, viewed, slot, ts, profile, context, pit_stats))
        buy, click = pid == purchased, pid == clicked
        y.append(int(buy or click))
        try:
            propensity = float(props[idx]) if idx < len(props) else .1
        except (TypeError, ValueError):
            propensity = .1
        sample_weights.append(min(1. / max(propensity, .02), 10.) * (4. if buy else 1.))

    # Alimentamos también de forma Point-in-Time con lo que este usuario acaba de hacer en esta sesión
    # para sus sesiones futuras:
    if uid:
        for vid in viewed:
            accumulate(uid, vid, "view", 1.0)
        if clicked:
            accumulate(uid, clicked, "add_to_cart", 1.0)
        if purchased:
            accumulate(uid, purchased, "purchase", 1.0)

    if purchased:
        accumulate_session_context(context, purchased, 5.0)
    elif clicked:
        accumulate_session_context(context, clicked, 2.0)

# Para el .pkl final de producción, completamos los perfiles con los últimos eventos y todos los compradores/carritos del histórico
while event_index < len(events):
    advance_pit_event(events[event_index])
    event_index += 1

print("   -> Enriqueciendo perfiles finales para producción con histórico de compras/carritos...")
full_user_rows = query("""
WITH (SELECT max(ts) FROM shop.history_events) AS max_ts
SELECT user_id::String AS user_id, product_id::String AS product_id,
       event_type::String AS event_type, count() AS cnt
FROM shop.history_events
WHERE user_id IS NOT NULL AND user_id::String != ''
  AND ts >= max_ts - INTERVAL 30 DAY
  AND event_type IN ('purchase', 'add_to_cart')
GROUP BY user_id, product_id, event_type
LIMIT 200000
""", timeout=180)
for r in full_user_rows:
    accumulate(str(r["user_id"]), str(r["product_id"]), str(r["event_type"]), float(r.get("cnt", 1.0)))

user_profiles.pop("", None)

X_np = np.asarray(X, dtype=np.float32)
y_np = np.asarray(y, dtype=np.int32)
w_np = np.asarray(sample_weights, dtype=np.float32)

print(f"5/6 Entrenando LightGBM sobre {len(X_np)} filas x {X_np.shape[1]} features...")
model = lgb.LGBMClassifier(
    n_estimators=180, learning_rate=.05, num_leaves=31,
    subsample_for_bin=50000, random_state=42, n_jobs=-1,
    importance_type="gain"
)
model.fit(X_np, y_np, sample_weight=w_np, feature_name=FEATURES)

data = {
    "catalog": catalog, "prod_stats": prod_stats, "co_matrix": dict(co_matrix),
    "user_profiles": user_profiles, "context_prices": dict(ctx_prices),
    "context_subcats": {k: dict(v) for k, v in ctx_subcats.items()},
    "context_totals": dict(ctx_totals), "feature_names": FEATURES
}
joblib.dump(model, MODEL_PATH)
joblib.dump(data, DATA_PATH)
shutil.copy2(MODEL_PATH, ROOT / MODEL_PATH.name)
shutil.copy2(DATA_PATH, ROOT / DATA_PATH.name)

print("\n6/6 Importancia por gain: porcentaje del total y número de splits")
gain = model.booster_.feature_importance(importance_type="gain")
splits = model.booster_.feature_importance(importance_type="split")
total_gain = float(gain.sum()) or 1.
print(f"{'#':>2} {'feature':<24} {'ganancia':>10} {'splits':>8}")
print("-" * 48)
for i in np.argsort(gain)[::-1]:
    print(f"{i:>2} {FEATURES[i]:<24} {gain[i] / total_gain * 100:>9.2f}% {int(splits[i]):>8}")
print(f"\nModelo guardado en {MODEL_PATH} ({len(user_profiles)} perfiles de usuario listos)")