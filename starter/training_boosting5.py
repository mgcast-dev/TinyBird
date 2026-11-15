"""Entrenamiento y evaluación offline v9 con negativos difíciles y split temporal.

Lee modelo_lgbm8.pkl/features_data4.pkl sin modificarlos y crea exclusivamente
starter/models/modelo_lgbm9.pkl, features_data5.pkl y training_report5.json.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
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
MODELS.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS / "modelo_lgbm9.pkl"
DATA_PATH = MODELS / "features_data5.pkl"
REPORT_PATH = MODELS / "training_report5.json"
SESSION_LIMIT = 30000
VALIDATION_FRACTION = 0.20
PURCHASE_WEIGHT = 3.0
HARD_NEGATIVE_WEIGHT = 0.75
GLOBAL_NEGATIVE_WEIGHT = 0.40

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

# El módulo v4 solo carga los artefactos existentes; no se inicia su main/bg_worker.
import recommender4 as v4

FEATURE_NAMES = list(v4.FEATURE_NAMES)
if v4.LGB_MODEL is None or not v4.FEAT_DATA:
    raise RuntimeError("No se pudieron cargar los artefactos v8 requeridos para construir v9")
BASE_DATA = v4.FEAT_DATA
CATALOG = v4.CATALOG
PROD_STATS = v4.PROD_STATS
BASE_CO_MATRIX = {source: dict(targets) for source, targets in v4.CO_MATRIX.items()}
BASE_CO_TOP = {source: list(targets) for source, targets in v4.CO_TOP_LIST.items()}
CO_MATRIX = BASE_CO_MATRIX
CO_TOP = BASE_CO_TOP
DEFAULT_ACTIVE = set(v4.ACTIVE_IDS) - set(v4.UNAVAILABLE_SET)
GLOBAL_POPULAR = [pid for pid in v4.TOP_GLOBAL_HIST if pid in DEFAULT_ACTIVE]


def query(sql: str, timeout: int = 180) -> list[dict[str, Any]]:
    response = HTTP.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    if not response.ok:
        print("[RawTree SQL]", response.status_code, response.text[:1000])
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


print("1/6 Descargando el tramo más reciente de sesiones...")
sessions = query(f"""
SELECT request_id::String AS request_id, ts::DateTime64(3) AS ts,
       user_id::String AS user_id, context, recent_views, shown, propensity,
       clicked, purchased, revenue_eur::Float64 AS revenue_eur
FROM shop.history_sessions
WHERE clicked IS NOT NULL OR purchased IS NOT NULL OR rand() % 8 = 0
ORDER BY ts::DateTime64(3) DESC
LIMIT {SESSION_LIMIT}
""", timeout=240)
sessions = [row for row in sessions if parse_ts(row.get("ts")) is not None]
sessions.sort(key=lambda row: parse_ts(row["ts"]))
if len(sessions) < 20:
    raise RuntimeError(f"Insuficientes sesiones para split temporal: {len(sessions)}")
split_index = max(1, min(len(sessions) - 1, int(len(sessions) * (1. - VALIDATION_FRACTION))))
# No permite que sesiones con el mismo timestamp queden a ambos lados del corte.
while split_index < len(sessions) - 1 and parse_ts(sessions[split_index - 1]["ts"]) == parse_ts(sessions[split_index]["ts"]):
    split_index += 1
train_sessions, validation_sessions = sessions[:split_index], sessions[split_index:]
validation_start = parse_ts(validation_sessions[0]["ts"])
print(f"   Sesiones: train={len(train_sessions)} validation={len(validation_sessions)}; corte={validation_start.isoformat()}")

# Reutiliza únicamente la co-matriz base de v8 y expone hasta 35 resultados
# por visto para que validación e inferencia de v9 compartan el mismo pool.
CO_MATRIX = BASE_CO_MATRIX
CO_TOP = {
    source: [target for target, _ in sorted(targets.items(), key=lambda item: item[1], reverse=True)[:35]]
    for source, targets in CO_MATRIX.items()
}
v4.CO_MATRIX = CO_MATRIX
v4.CO_TOP_LIST = CO_TOP
print(f"   Co-matriz v8 cargada: {len(CO_MATRIX):,} productos origen")

cutoff_sql = validation_start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
best_rows = query(f"""
SELECT product_id::String AS product_id, count() AS purchases
FROM shop.history_events
WHERE event_type='purchase' AND ts::DateTime64(3) < toDateTime64('{cutoff_sql}', 3, 'UTC')
GROUP BY product_id
ORDER BY purchases DESC
LIMIT 5000
""", timeout=120)
VALIDATION_BEST = [str(row["product_id"]) for row in best_rows if str(row["product_id"]) in DEFAULT_ACTIVE]
if not VALIDATION_BEST:
    VALIDATION_BEST = GLOBAL_POPULAR

TOP_BY_SUBCAT: dict[str, list[str]] = defaultdict(list)
for pid in GLOBAL_POPULAR:
    subcat = str(CATALOG.get(pid, {}).get("subcategory") or "")
    if subcat and len(TOP_BY_SUBCAT[subcat]) < 40:
        TOP_BY_SUBCAT[subcat].append(pid)
TRENDING = sorted(
    (pid for pid in DEFAULT_ACTIVE),
    key=lambda pid: PROD_STATS.get(pid, {}).get("purchases_24h", 0.) * 10.
        + PROD_STATS.get(pid, {}).get("carts_24h", 0.) * 3.
        + PROD_STATS.get(pid, {}).get("views_24h", 0.),
    reverse=True,
)

# Profiles temporales creados durante el recorrido cronológico de sesiones. Las
# features de usuario de un ejemplo solo ven outcomes de sesiones anteriores.
PIT_USERS: dict[str, dict[str, Any]] = {}
PIT_CONTEXT_PRICES: dict[str, list[float]] = defaultdict(lambda: [0., 0.])
PIT_CONTEXT_SUBCATS: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
PIT_CONTEXT_TOTALS: dict[str, float] = defaultdict(float)
PIT_CONTEXT_PRODUCTS: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))


def ctx_keys(context: dict[str, Any]) -> list[str]:
    region, device = str(context.get("region", "")), str(context.get("device", ""))
    return [f"{region}|{device}", f"{region}|", f"|{device}"]


def empty_profile() -> dict[str, Any]:
    return {"subcat_weights": {}, "brand_weights": {}, "price_sum": 0., "price_weight": 0.,
            "items": {}, "event_count": 0}


def update_profile_event(user_id: str, product_id: str, kind: str, weight: float) -> None:
    if not user_id or not product_id:
        return
    product = CATALOG.get(product_id, {})
    subcat, brand = str(product.get("subcategory") or ""), str(product.get("brand") or "")
    price = float(product.get("price_eur") or 0.)
    profile = PIT_USERS.setdefault(user_id, empty_profile())
    if subcat:
        profile["subcat_weights"][subcat] = profile["subcat_weights"].get(subcat, 0.) + weight
    if brand:
        profile["brand_weights"][brand] = profile["brand_weights"].get(brand, 0.) + weight
    if price > 0:
        profile["price_sum"] += price * weight
        profile["price_weight"] += weight
    profile["event_count"] += 1
    state = profile["items"].setdefault(product_id, {"view": False, "cart": False, "purchase": False})
    state_key = {"view": "view", "add_to_cart": "cart", "purchase": "purchase"}.get(kind)
    if state_key:
        state[state_key] = True


def update_context_event(context: dict[str, Any], product_id: str, weight: float) -> None:
    product = CATALOG.get(product_id, {})
    subcat = str(product.get("subcategory") or "")
    price = float(product.get("price_eur") or 0.)
    for key in ctx_keys(context):
        if price > 0:
            PIT_CONTEXT_PRICES[key][0] += price
            PIT_CONTEXT_PRICES[key][1] += 1.
        if subcat:
            PIT_CONTEXT_SUBCATS[key][subcat] += weight
            PIT_CONTEXT_TOTALS[key] += weight
        PIT_CONTEXT_PRODUCTS[key][product_id] += weight


def activate_pit_profiles() -> None:
    v4.USER_PROFILES = PIT_USERS
    v4.CONTEXT_PRICES = PIT_CONTEXT_PRICES
    v4.CONTEXT_SUBCATS = PIT_CONTEXT_SUBCATS
    v4.CONTEXT_TOTALS = PIT_CONTEXT_TOTALS


def update_pit_after_session(session: dict[str, Any], viewed: list[str], clicked: str, purchased: str) -> None:
    uid = str(session.get("user_id") or "")
    context = unpack(session.get("context"), dict)
    for pid in viewed:
        update_profile_event(uid, pid, "view", 1.)
    if clicked:
        update_profile_event(uid, clicked, "add_to_cart", 2.)
        update_context_event(context, clicked, 2.)
    if purchased:
        update_profile_event(uid, purchased, "purchase", 5.)
        update_context_event(context, purchased, 5.)


def add_unique(target: list[str], seen: set[str], pid: Any, active: set[str]) -> bool:
    product_id = str(pid or "")
    if product_id and product_id in active and product_id not in seen and len(target) < 80:
        target.append(product_id)
        seen.add(product_id)
        return True
    return False


def inference_candidates(viewed: list[str], user_id: str, context: dict[str, Any],
                         active: set[str], best: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for pid in reversed(viewed[-2:]):
        add_unique(result, seen, pid, active)
    co_count = 0
    for pid in reversed(viewed):
        for candidate in CO_TOP.get(pid, []):
            co_count += int(add_unique(result, seen, candidate, active))
            if co_count >= 35 or len(result) >= 80:
                break
        if co_count >= 35 or len(result) >= 80:
            break
    profile = PIT_USERS.get(user_id, {})
    subcats: list[str] = []
    if viewed:
        last_subcat = str(CATALOG.get(viewed[-1], {}).get("subcategory") or "")
        if last_subcat:
            subcats.append(last_subcat)
    favorites = sorted(profile.get("subcat_weights", {}).items(), key=lambda item: item[1], reverse=True)[:3]
    subcats.extend(subcat for subcat, _ in favorites if subcat and subcat not in subcats)
    if not subcats:
        contextual: dict[str, float] = defaultdict(float)
        for key in ctx_keys(context):
            for subcat, score in PIT_CONTEXT_SUBCATS.get(key, {}).items():
                contextual[subcat] += float(score)
            if contextual:
                break
        subcats = [subcat for subcat, _ in sorted(contextual.items(), key=lambda item: item[1], reverse=True)[:3]]
    subcat_count = 0
    for subcat in subcats:
        for candidate in TOP_BY_SUBCAT.get(subcat, []):
            subcat_count += int(add_unique(result, seen, candidate, active))
            if subcat_count >= 20 or len(result) >= 80:
                break
        if subcat_count >= 20 or len(result) >= 80:
            break
    trend_count = 0
    for candidate in TRENDING:
        trend_count += int(add_unique(result, seen, candidate, active))
        if trend_count >= 15 or len(result) >= 80:
            break
    contextual_products: dict[str, float] = defaultdict(float)
    for key in ctx_keys(context):
        for product_id, score in PIT_CONTEXT_PRODUCTS.get(key, {}).items():
            contextual_products[product_id] += float(score)
        if contextual_products:
            break
    context_added = 0
    for product_id, _ in sorted(contextual_products.items(), key=lambda item: item[1], reverse=True):
        context_added += int(add_unique(result, seen, product_id, active))
        if context_added >= 10 or len(result) >= 80:
            break
    profile = PIT_USERS.get(user_id, {})
    abandoned = [pid for pid, state in profile.get("items", {}).items()
                 if state.get("cart") and not state.get("purchase")]
    abandoned.sort(key=lambda pid: PROD_STATS.get(pid, {}).get("purchases", 0.), reverse=True)
    for candidate in abandoned[:8]:
        add_unique(result, seen, candidate, active)
    previously_viewed = [
        pid for pid, state in profile.get("items", {}).items()
        if state.get("view") and not state.get("purchase") and not state.get("cart")
    ]
    previously_viewed.sort(key=lambda pid: PROD_STATS.get(pid, {}).get("purchases", 0.), reverse=True)
    for candidate in previously_viewed[:8]:
        add_unique(result, seen, candidate, active)
    for candidate in best:
        add_unique(result, seen, candidate, active)
        if len(result) >= 80:
            break
    return result


def training_candidates(shown: list[str], viewed: list[str], active: set[str]) -> tuple[list[str], dict[str, float]]:
    """Incluye impresiones y negativos difíciles de subcategoría/co-matriz/populares."""
    candidates: list[str] = []
    source_weight: dict[str, float] = {}
    seen: set[str] = set()
    for pid in shown:
        if add_unique(candidates, seen, pid, active):
            source_weight[pid] = 1.0
    last_subcat = str(CATALOG.get(viewed[-1], {}).get("subcategory") or "") if viewed else ""
    subcat_added = 0
    if last_subcat:
        for pid in TOP_BY_SUBCAT.get(last_subcat, []):
            if pid not in seen and add_unique(candidates, seen, pid, active):
                source_weight[pid] = HARD_NEGATIVE_WEIGHT
                subcat_added += 1
            if subcat_added >= 12 or len(candidates) >= 80:
                break
    co_added = 0
    for viewed_pid in reversed(viewed):
        for pid in CO_TOP.get(viewed_pid, []):
            if pid not in seen and add_unique(candidates, seen, pid, active):
                source_weight[pid] = HARD_NEGATIVE_WEIGHT
                co_added += 1
            if co_added >= 12 or len(candidates) >= 80:
                break
        if co_added >= 12 or len(candidates) >= 80:
            break
    global_added = 0
    for pid in GLOBAL_POPULAR:
        if pid not in seen and add_unique(candidates, seen, pid, active):
            source_weight[pid] = GLOBAL_NEGATIVE_WEIGHT
            global_added += 1
        if global_added >= 4 or len(candidates) >= 80:
            break
    return candidates, source_weight


def build_row(product_id: str, viewed: list[str], uid: str, context: dict[str, Any]) -> list[float]:
    return v4.build_feature_row(product_id, viewed, uid, context)


print("2/6 Construyendo ejemplos con negativos difíciles y perfiles cronológicos...")
all_train_rows: list[np.ndarray] = []
all_binary_labels: list[int] = []
all_rank_labels: list[int] = []
all_weights: list[float] = []
all_groups: list[int] = []
validation_feature_rows: list[np.ndarray] = []
validation_train_rows: list[np.ndarray] = []
validation_binary_labels: list[int] = []
validation_rank_labels: list[int] = []
validation_weights: list[float] = []
validation_groups: list[int] = []
validation_records: list[dict[str, Any]] = []
active_ids = set(DEFAULT_ACTIVE)
v4_recommendations: list[list[str]] = []
v4_inference_seconds = 0.0
validation_row_cursor = 0

for session_index, session in enumerate(sessions):
    ts = parse_ts(session["ts"])
    uid = str(session.get("user_id") or "")
    context = unpack(session.get("context"), dict)
    recent = unpack(session.get("recent_views"), list)
    viewed = [str(v["product_id"]) for v in recent if isinstance(v, dict) and v.get("product_id")]
    shown = [str(pid) for pid in unpack(session.get("shown"), list) if pid]
    props = unpack(session.get("propensity"), list)
    clicked = str(session.get("clicked") or "")
    purchased = str(session.get("purchased") or "")
    revenue = float(session.get("revenue_eur") or 0.)
    activate_pit_profiles()

    if session_index < split_index:
        candidates, source_weights = training_candidates(shown, viewed, active_ids)
        group_size = 0
        session_rows: list[list[float]] = []
        for candidate in candidates:
            shown_index = shown.index(candidate) if candidate in shown else -1
            if shown_index >= 0:
                try:
                    propensity = float(props[shown_index]) if shown_index < len(props) else .1
                except (TypeError, ValueError):
                    propensity = .1
                row_weight = min(1. / max(propensity, .02), 10.)
            else:
                row_weight = source_weights.get(candidate, HARD_NEGATIVE_WEIGHT)
            is_buy = candidate == purchased and bool(purchased)
            is_click = candidate == clicked and bool(clicked)
            if is_buy:
                row_weight = max(row_weight, 1.0) * PURCHASE_WEIGHT
            elif is_click:
                row_weight = max(row_weight, 1.0)
            session_rows.append(build_row(candidate, viewed, uid, context))
            all_binary_labels.append(int(is_buy or is_click))
            all_rank_labels.append(2 if is_buy else (1 if is_click else 0))
            all_weights.append(row_weight)
            group_size += 1
        if session_rows:
            all_train_rows.append(np.asarray(session_rows, dtype=np.float32))
            all_groups.append(group_size)
    else:
        build_start = time.perf_counter()
        candidates = inference_candidates(viewed, uid, context, active_ids, VALIDATION_BEST)
        start = validation_row_cursor
        val_features = [build_row(candidate, viewed, uid, context) for candidate in candidates]
        feature_build_ms = (time.perf_counter() - build_start) * 1000.
        if val_features:
            validation_feature_rows.append(np.asarray(val_features, dtype=np.float32))
        validation_row_cursor += len(val_features)
        baseline_start = time.perf_counter()
        # Mantiene el baseline v4 realmente intacto en la comparación.
        current_co, current_co_top = v4.CO_MATRIX, v4.CO_TOP_LIST
        v4.CO_MATRIX, v4.CO_TOP_LIST = BASE_CO_MATRIX, BASE_CO_TOP
        try:
            v4_recs = v4.recommend(recent, VALIDATION_BEST, user_id=uid, context=context)
        finally:
            v4.CO_MATRIX, v4.CO_TOP_LIST = current_co, current_co_top
        v4_inference_seconds += time.perf_counter() - baseline_start
        v4_recommendations.append(v4_recs)
        validation_records.append({
            "request_id": str(session.get("request_id") or ""),
            "candidates": candidates,
            "start": start,
            "end": start + len(val_features),
            "clicked": clicked,
            "purchased": purchased,
            "viewed": viewed,
            "revenue": revenue,
            "recent_views": recent,
            "user_id": uid,
            "context": context,
            "feature_build_ms": feature_build_ms,
        })
        # Guardamos también las impresiones y negativos difíciles de la parte
        # temporal de validación para el reentrenamiento final del modelo.
        final_candidates, final_source_weights = training_candidates(shown, viewed, active_ids)
        final_group_size = 0
        final_session_rows: list[list[float]] = []
        for candidate in final_candidates:
            shown_index = shown.index(candidate) if candidate in shown else -1
            if shown_index >= 0:
                try:
                    propensity = float(props[shown_index]) if shown_index < len(props) else .1
                except (TypeError, ValueError):
                    propensity = .1
                row_weight = min(1. / max(propensity, .02), 10.)
            else:
                row_weight = final_source_weights.get(candidate, HARD_NEGATIVE_WEIGHT)
            is_buy = candidate == purchased and bool(purchased)
            is_click = candidate == clicked and bool(clicked)
            if is_buy:
                row_weight = max(row_weight, 1.0) * PURCHASE_WEIGHT
            elif is_click:
                row_weight = max(row_weight, 1.0)
            final_session_rows.append(build_row(candidate, viewed, uid, context))
            validation_binary_labels.append(int(is_buy or is_click))
            validation_rank_labels.append(2 if is_buy else (1 if is_click else 0))
            validation_weights.append(row_weight)
            final_group_size += 1
        if final_session_rows:
            validation_train_rows.append(np.asarray(final_session_rows, dtype=np.float32))
            validation_groups.append(final_group_size)
    # Outcomes solo actualizan el perfil después de construir las features de sesión.
    update_pit_after_session(session, viewed, clicked, purchased)

# Separa la porción de validación de las filas concatenadas al final.
X_train = np.vstack(all_train_rows) if all_train_rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
train_end = len(X_train)
y_binary = np.asarray(all_binary_labels, dtype=np.int32)
y_rank = np.asarray(all_rank_labels, dtype=np.int32)
weights = np.asarray(all_weights, dtype=np.float32)
if X_train.ndim != 2 or X_train.shape[1] != len(FEATURE_NAMES) or not len(X_train):
    raise RuntimeError(f"Matriz de entrenamiento inválida: {X_train.shape}")
val_X = np.vstack(validation_feature_rows) if validation_feature_rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
if len(val_X) != sum(record["end"] - record["start"] for record in validation_records):
    raise RuntimeError("Las filas de validación no coinciden con los grupos")
print(f"   Train: {len(X_train):,} filas; validación: {len(val_X):,} filas; features={len(FEATURE_NAMES)}")


def metrics_for_predictions(records: list[dict[str, Any]], score_vector: np.ndarray) -> dict[str, float | int]:
    click_count = 0
    hits = 0
    novel_hits = 0
    novel_clicks = 0
    candidate_hits = 0
    novel_candidate_hits = 0
    revenue_hit = 0.0
    elapsed_requests = 0
    for record in records:
        clicked = record["clicked"]
        purchased = record["purchased"]
        candidates = record["candidates"]
        scores = score_vector[record["start"]:record["end"]]
        order = np.argsort(scores)[::-1][:10]
        recommended = [candidates[int(i)] for i in order]
        target = clicked or purchased
        if clicked:
            click_count += 1
            if clicked in candidates:
                candidate_hits += 1
                if clicked not in record["viewed"]:
                    novel_candidate_hits += 1
            if target in recommended:
                hits += 1
                if target not in record["viewed"]:
                    novel_hits += 1
            if target not in record["viewed"]:
                novel_clicks += 1
        if purchased and purchased in recommended:
            revenue_hit += float(record["revenue"])
        elapsed_requests += 1
    denominator = max(click_count, 1)
    return {
        "click_sessions": click_count,
        "hit_rate@10": hits / denominator,
        "novel_hit_rate@10": novel_hits / denominator,
        "novel_hit_rate_conditional": novel_hits / max(novel_clicks, 1),
        "candidate_recall@80": candidate_hits / denominator,
        "novel_candidate_recall@80": novel_candidate_hits / denominator,
        "novel_click_sessions": novel_clicks,
        "expected_revenue_per_request_eur": revenue_hit / max(elapsed_requests, 1),
        "requests": elapsed_requests,
    }


def predict_records(model: Any, records: list[dict[str, Any]]) -> np.ndarray:
    return model.booster_.predict(val_X)


def validation_metrics(model: Any) -> tuple[dict[str, float | int], np.ndarray]:
    scores = predict_records(model, validation_records)
    return metrics_for_predictions(validation_records, scores), scores


EXPERIMENTS = [
    ("binary", 31, .03, 20), ("binary", 31, .08, 100),
    ("binary", 63, .05, 50), ("binary", 63, .08, 20),
    ("binary", 127, .03, 100), ("binary", 127, .08, 50),
    ("lambdarank", 31, .03, 20), ("lambdarank", 31, .08, 100),
    ("lambdarank", 63, .05, 50), ("lambdarank", 127, .08, 100),
]
train_groups = []
rows_at = 0
for group in all_groups:
    if rows_at + group > train_end:
        break
    train_groups.append(group)
    rows_at += group

print("3/6 Probando objetivos y configuraciones de LightGBM en el split temporal...")
experiment_results: list[dict[str, Any]] = []
champion = None
champion_config: tuple[str, int, float, int] | None = None
champion_score = -1.0
champion_metrics: dict[str, float | int] = {}
for index, config in enumerate(EXPERIMENTS, start=1):
    objective, leaves, learning_rate, min_child = config
    shared_params = dict(
        n_estimators=180, learning_rate=learning_rate, num_leaves=leaves,
        min_child_samples=min_child, subsample=.8, subsample_freq=1,
        colsample_bytree=.8, random_state=42, n_jobs=-1, verbosity=-1,
    )
    if objective == "binary":
        candidate_model = lgb.LGBMClassifier(objective="binary", **shared_params)
        fit_y = y_binary
    else:
        candidate_model = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", label_gain=[0, 1, 3], **shared_params
        )
        fit_y = y_rank
    fit_kwargs: dict[str, Any] = {"sample_weight": weights}
    if objective == "lambdarank":
        fit_kwargs["group"] = train_groups
    candidate_model.fit(X_train, fit_y, **fit_kwargs)
    current_metrics, _ = validation_metrics(candidate_model)
    selection_score = (float(current_metrics["hit_rate@10"]) + float(current_metrics["novel_hit_rate@10"])) / 2.
    experiment_results.append({
        "objective": objective, "num_leaves": leaves, "learning_rate": learning_rate,
        "min_child_samples": min_child, "subsample": .8, "colsample_bytree": .8,
        **current_metrics, "selection_score": selection_score,
    })
    print(
        f"   [{index:02d}/{len(EXPERIMENTS)}] {objective:10s} leaves={leaves:3d} "
        f"lr={learning_rate:.2f} child={min_child:3d} hit={current_metrics['hit_rate@10']:.4f} "
        f"novel={current_metrics['novel_hit_rate@10']:.4f} score={selection_score:.4f}"
    )
    if selection_score > champion_score:
        champion_score, champion_config, champion = selection_score, config, candidate_model
        champion_metrics = current_metrics

if champion is None or champion_config is None:
    raise RuntimeError("Ninguna configuración de LightGBM se pudo entrenar")

print("4/6 Comparando v8 y v9 en las mismas sesiones de validación...")
v5_scores_parts: list[np.ndarray] = []
v5_inference_seconds = 0.0
for record in validation_records:
    t0 = time.perf_counter()
    if record["end"] > record["start"]:
        row_scores = champion.booster_.predict(val_X[record["start"]:record["end"]])
    else:
        row_scores = np.asarray([], dtype=np.float32)
    v5_inference_seconds += time.perf_counter() - t0
    v5_inference_seconds += record["feature_build_ms"] / 1000.
    v5_scores_parts.append(row_scores)

def metrics_for_lists(records: list[dict[str, Any]], rec_lists: list[list[str]]) -> dict[str, float | int]:
    clicks = hits = novel_hits = novel_clicks = 0
    revenue_hit = 0.
    for record, recs in zip(records, rec_lists):
        target = record["clicked"] or record["purchased"]
        if record["clicked"]:
            clicks += 1
            if target in recs:
                hits += 1
                if target not in record["viewed"]:
                    novel_hits += 1
            if target not in record["viewed"]:
                novel_clicks += 1
        if record["purchased"] and record["purchased"] in recs:
            revenue_hit += float(record["revenue"])
    return {
        "click_sessions": clicks,
        "hit_rate@10": hits / max(clicks, 1),
        "novel_hit_rate@10": novel_hits / max(clicks, 1),
        "novel_hit_rate_conditional": novel_hits / max(novel_clicks, 1),
        "novel_click_sessions": novel_clicks,
        "expected_revenue_per_request_eur": revenue_hit / max(len(records), 1),
        "requests": len(records),
    }

champion_recs = []
for record, scores in zip(validation_records, v5_scores_parts):
    order = np.argsort(scores)[::-1][:10]
    champion_recs.append([record["candidates"][int(i)] for i in order])
v4_metrics = metrics_for_lists(validation_records, v4_recommendations)
v5_metrics = metrics_for_lists(validation_records, champion_recs)
v5_metrics["candidate_recall@80"] = champion_metrics.get("candidate_recall@80", 0.)
v5_metrics["novel_candidate_recall@80"] = champion_metrics.get("novel_candidate_recall@80", 0.)
v4_metrics["mean_inference_ms_per_request"] = 1000. * v4_inference_seconds / max(len(validation_records), 1)
v5_metrics["mean_inference_ms_per_request"] = 1000. * v5_inference_seconds / max(len(validation_records), 1)

print("5/6 Reentrenando el ganador sobre train+validación y guardando solo artefactos v9...")
objective, leaves, learning_rate, min_child = champion_config
final_params = dict(
    n_estimators=180, learning_rate=learning_rate, num_leaves=leaves,
    min_child_samples=min_child, subsample=.8, subsample_freq=1,
    colsample_bytree=.8, random_state=42, n_jobs=-1, verbosity=-1,
)
X_val_train = np.vstack(validation_train_rows) if validation_train_rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
X_final = np.vstack((X_train, X_val_train))
y_final_binary = np.concatenate((y_binary, np.asarray(validation_binary_labels, dtype=np.int32)))
y_final_rank = np.concatenate((y_rank, np.asarray(validation_rank_labels, dtype=np.int32)))
weights_final = np.concatenate((weights, np.asarray(validation_weights, dtype=np.float32)))
groups_final = all_groups + validation_groups
if objective == "binary":
    final_model = lgb.LGBMClassifier(objective="binary", **final_params)
    final_model.fit(X_final, y_final_binary, sample_weight=weights_final)
else:
    final_model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", label_gain=[0, 1, 3], **final_params)
    final_model.fit(X_final, y_final_rank, group=groups_final, sample_weight=weights_final)

feature_data = dict(BASE_DATA)
feature_data.update({
    "feature_names": FEATURE_NAMES,
    "context_products": {key: dict(values) for key, values in PIT_CONTEXT_PRODUCTS.items()},
    "candidate_pool_size": 80,
    "training_metadata": {
        "temporal_split": True,
        "validation_start": validation_start.isoformat(),
        "train_sessions": len(train_sessions),
        "validation_sessions": len(validation_sessions),
        "selected_objective": objective,
        "selected_num_leaves": leaves,
        "selected_learning_rate": learning_rate,
        "selected_min_child_samples": min_child,
        "hard_negative_weights": {"subcat_co_matrix": HARD_NEGATIVE_WEIGHT, "global_popular": GLOBAL_NEGATIVE_WEIGHT},
    },
})
joblib.dump(final_model, MODEL_PATH)
joblib.dump(feature_data, DATA_PATH)

report = {
    "validation_start": validation_start.isoformat(),
    "train_sessions": len(train_sessions),
    "validation_sessions": len(validation_sessions),
    "selected_configuration": {
        "objective": objective, "num_leaves": leaves, "learning_rate": learning_rate,
        "min_child_samples": min_child, "subsample": .8, "colsample_bytree": .8,
    },
    "selection_rule": "maximiza la media de hit_rate@10 y novel_hit_rate@10",
    "experiments": experiment_results,
    "comparison": {"recommender4": v4_metrics, "recommender5_selected": v5_metrics},
    "expected_value_definition": "revenue_eur de compra cuyo producto está en top-10, dividido por peticiones de validación",
    "inference_timing": "CPU local; incluye generación de features/candidatos y booster.predict, excluye red",
    "feature_snapshot_note": "Las features de usuario se generaron cronológicamente desde el split; stats/co_matrix del artefacto v8 se reutilizan como snapshot compartido.",
}
REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

print("6/6 Informe offline")
print(f"{'modelo':<24} {'hit@10':>9} {'novel@10':>10} {'ms/petición':>13} {'EUR/petición':>14}")
for name, metrics in (("recommender4", v4_metrics), ("recommender5", v5_metrics)):
    print(f"{name:<24} {metrics['hit_rate@10']:>9.4f} {metrics['novel_hit_rate@10']:>10.4f} "
          f"{metrics['mean_inference_ms_per_request']:>13.3f} {metrics['expected_revenue_per_request_eur']:>14.4f}")
print(f"Ganador: {objective}, leaves={leaves}, lr={learning_rate}, min_child={min_child}")
print(f"Validación del ganador (antes del reentrenamiento final): {champion_metrics}")
print(f"Experimentos: {REPORT_PATH}")
print(f"Modelo v9: {MODEL_PATH}")
print(f"Features v9: {DATA_PATH}")
