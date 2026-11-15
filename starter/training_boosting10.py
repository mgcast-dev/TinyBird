"""Entrena LGBM 8 (28 features) con perfiles reconstruidos a tiempo de request.

No modifica modelo_lgbm7.pkl ni features_data7.pkl. Requiere que
shop.history_sessions incluya request_id, user_id y ts.
"""
from __future__ import annotations

import bisect
import math
import os
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines() if (ROOT / ".env").exists() else []:
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
BASELINE_PATH = MODELS_DIR / "modelo_lgbm7.pkl"
MAX_SESSIONS = int(os.environ.get("LGBM8_MAX_SESSIONS", "1000"))
HISTORY_DAYS = int(os.environ.get("LGBM8_HISTORY_DAYS", "1"))
COLD_START_AUGMENT_WEIGHT = float(os.environ.get("LGBM8_COLD_START_AUGMENT_WEIGHT", "1.0"))

FEATURE_NAMES = [
    "purchases", "conv_rate", "cart_rate", "price", "price_ratio",
    "same_subcat", "same_cat", "same_brand", "co_score_last", "co_score_total",
    "already_viewed", "slot", "purchases_24h", "views_24h", "carts_24h",
    "product_age_days", "recent_popularity",
    "user_subcategory_affinity", "user_brand_affinity", "price_fit",
    "user_purchase_rate", "user_bought_similar", "recent_subcategory_affinity",
    "candidate_seen_by_user", "days_since_last_interaction", "user_interaction_score",
    "candidate_price_percentile_user", "user_is_cold_start",
]


def query(sql: str, timeout: int = 120) -> list[dict[str, Any]]:
    response = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"RawTree HTTP {response.status_code}: {response.text[:1500]}")
    data = response.json().get("data")
    if not isinstance(data, list):
        raise RuntimeError("RawTree no devolviÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³ una lista en data")
    return data


def parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        raw = str(value).strip()
        try:
            dt = datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except (TypeError, ValueError):
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def parse_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


print("[LGBM8 1/7] Descargando sesiones con usuario y timestamp...")
# Esta uniÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n es necesaria para reconstruir el perfil en el instante de la request.
sessions_sql = f"""
SELECT
    request_id::String AS request_id,
    user_id::String AS user_id,
    ts AS request_ts,
    recent_views AS recent_views,
    shown AS shown,
    propensity AS propensity,
    clicked AS clicked,
    purchased AS purchased
FROM shop.history_sessions
WHERE (clicked IS NOT NULL OR cityHash64(request_id::String) % 8 = 0)
ORDER BY ts DESC
LIMIT {MAX_SESSIONS}
"""
try:
    raw_sessions = query(sessions_sql, timeout=180)
except Exception as exc:
    raise RuntimeError(
        "No se pudo enlazar shop.history_sessions.request_id con shop.requests.request_id. "
        "LGBM 8 necesita esa uniÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n para obtener user_id y ts; revisa el esquema RawTree. "
        f"Error original: {exc}"
    ) from exc

sessions: list[dict[str, Any]] = []
for row in raw_sessions:
    ts = parse_ts(row.get("request_ts"))
    if ts is None or row.get("request_id") is None:
        continue
    row["_ts"] = ts
    row["_uid"] = str(row.get("user_id") or "")
    sessions.append(row)
sessions.sort(key=lambda row: row["_ts"])
if len(sessions) < 30:
    raise RuntimeError(f"Solo hay {len(sessions)} requests vÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡lidas; hacen falta mÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡s para train/validation/test.")
print(f"  Requests muestreadas: {len(sessions)}")

# Agregado hist?rico peque?o para detectar usuarios activos antes de la ventana raw.
# El corte es anterior a la request m?s antigua del muestreo, as? que no filtra futuro.
sampled_users = sorted({row["_uid"] for row in sessions if row["_uid"]})
HISTORICAL_USERS: set[str] = set()
if sampled_users:
    prior_cutoff = sessions[0]["_ts"] - timedelta(days=HISTORY_DAYS)
    prior_cutoff_sql = prior_cutoff.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    prior_users_sql = f"""
SELECT user_id::String AS user_id, count() AS n_interactions
FROM shop.history_events
WHERE user_id IS NOT NULL
  AND ts::DateTime64(3) < toDateTime64('{prior_cutoff_sql}', 3)
  AND event_type::String IN ('view', 'click', 'add_to_cart', 'purchase')
GROUP BY user_id
"""
    try:
        sampled_user_set = set(sampled_users)
        HISTORICAL_USERS = {str(row["user_id"]) for row in query(prior_users_sql, timeout=180)
                            if str(row.get("user_id")) in sampled_user_set and numeric(row.get("n_interactions")) > 0}
    except Exception as exc:
        raise RuntimeError(f"No se pudo calcular el estado cold-start previo al muestreo: {exc}") from exc
print(f"  Usuarios con actividad antes de la ventana: {len(HISTORICAL_USERS)} / {len(sampled_users)}")

print("[LGBM8 2/7] Descargando catÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡logo y cambios temporales...")
base_rows = query("""
SELECT product_id::String AS product_id,
       category::String AS category,
       subcategory::String AS subcategory,
       brand::String AS brand,
       price_eur::Float64 AS price_eur,
       is_active::Bool AS is_active,
       created_at
FROM shop.catalog
""")
base_catalog = {str(r["product_id"]): dict(r) for r in base_rows if r.get("product_id") is not None}
try:
    update_rows = query("""
SELECT product_id::String AS product_id,
       category::String AS category,
       subcategory::String AS subcategory,
       brand::String AS brand,
       price_eur::Float64 AS price_eur,
       is_active::Bool AS is_active,
       ts
FROM shop.catalog_updates
ORDER BY ts
""", timeout=180)
except Exception as exc:
    raise RuntimeError(
        "No se pudieron leer las versiones temporales de shop.catalog_updates. "
        "El entrenamiento no usarÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡ silenciosamente el catÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡logo actual para requests antiguas. "
        f"Error original: {exc}"
    ) from exc

catalog_updates: dict[str, list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
for row in update_rows:
    ts = parse_ts(row.get("ts"))
    if ts and row.get("product_id") is not None:
        snapshot = {k: row.get(k) for k in ("category", "subcategory", "brand", "price_eur", "is_active")}
        catalog_updates[str(row["product_id"])].append((ts, snapshot))
for pid in catalog_updates:
    catalog_updates[pid].sort(key=lambda item: item[0])
    # Updates may contain partial values; fill missing fields only from an earlier update.
    previous: dict[str, Any] = {}
    fixed = []
    for ts, snapshot in catalog_updates[pid]:
        current = dict(previous)
        current.update({k: v for k, v in snapshot.items() if v is not None})
        fixed.append((ts, current))
        previous = current
    catalog_updates[pid] = fixed

catalog_times = {pid: [item[0] for item in updates] for pid, updates in catalog_updates.items()}

def catalog_at(pid: str, ts: datetime) -> dict[str, Any] | None:
    updates = catalog_updates.get(pid)
    if updates:
        idx = bisect.bisect_left(catalog_times[pid], ts) - 1
        if idx < 0:
            # No se atribuye un snapshot futuro a una fecha anterior.
            return None
        info = dict(updates[idx][1])
        created = parse_ts(base_catalog.get(pid, {}).get("created_at"))
        if created and created >= ts:
            return None
        info["created_at"] = created
        return info
    info = base_catalog.get(pid)
    if not info:
        return None
    created = parse_ts(info.get("created_at"))
    if created and created >= ts:
        return None
    return info

print("[LGBM8 3/7] Descargando eventos histÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³ricos ordenables por tiempo...")
bounds = query("SELECT min(ts) AS min_ts, max(ts) AS max_ts FROM shop.history_events", timeout=90)
if not bounds or not parse_ts(bounds[0].get("min_ts")) or not parse_ts(bounds[0].get("max_ts")):
    raise RuntimeError("shop.history_events est? vac?o o no devuelve min(ts)/max(ts).")
event_min = max(parse_ts(bounds[0]["min_ts"]), sessions[0]["_ts"] - timedelta(days=HISTORY_DAYS))
event_max = parse_ts(bounds[0]["max_ts"]) + timedelta(milliseconds=1)
print(f"  Ventana de eventos: {event_min.isoformat()} .. {event_max.isoformat()} ({HISTORY_DAYS} d?as previos a la primera request)")
total_event_count = 0

def fetch_event_range(start: datetime, end: datetime, depth: int = 0) -> list[dict[str, Any]]:
    start_sql = start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    end_sql = end.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    sql = f"""
SELECT user_id::String AS user_id,
       session_id::String AS session_id,
       product_id::String AS product_id,
       event_type::String AS event_type,
       ts
FROM shop.history_events
WHERE ts::DateTime64(3) >= toDateTime64('{start_sql}', 3)
  AND ts::DateTime64(3) < toDateTime64('{end_sql}', 3)
ORDER BY ts
"""
    try:
        return query(sql, timeout=240)
    except RuntimeError as exc:
        if "Limit for result exceeded" not in str(exc) or depth >= 16 or (end - start).total_seconds() <= 1:
            raise
        midpoint = start + (end - start) / 2
        print(f"  Intervalo grande; subdividiendo {start_sql} .. {end_sql}")
        return fetch_event_range(start, midpoint, depth + 1) + fetch_event_range(midpoint, end, depth + 1)

def iter_events():
    global total_event_count
    cursor = event_min
    chunk = timedelta(days=1)
    while cursor < event_max:
        next_cursor = min(cursor + chunk, event_max)
        raw_chunk = fetch_event_range(cursor, next_cursor)
        chunk_events = []
        for row in raw_chunk:
            ts = parse_ts(row.get("ts"))
            if ts is None or row.get("product_id") is None:
                continue
            kind = str(row.get("event_type", "")).lower()
            if kind not in {"view", "click", "add_to_cart", "purchase"}:
                continue
            row["_ts"] = ts
            row["_pid"] = str(row["product_id"])
            row["_uid"] = str(row.get("user_id") or "")
            row["_sid"] = str(row.get("session_id") or "")
            row["_kind"] = kind
            chunk_events.append(row)
        del raw_chunk
        chunk_events.sort(key=lambda row: row["_ts"])
        total_event_count += len(chunk_events)
        for event in chunk_events:
            yield event
        del chunk_events
        cursor = next_cursor
        print(f"  Eventos procesados/descargados: {total_event_count}")

n = len(sessions)
train_end = max(1, int(n * 0.70))
valid_end = max(train_end + 1, int(n * 0.85))
valid_end = min(valid_end, n - 1)
train_cut = sessions[train_end]["_ts"]
valid_cut = sessions[valid_end]["_ts"]
print(f"  Train hasta {train_cut.isoformat()} | validation hasta {valid_cut.isoformat()} | test posterior")

# Estado online que avanza en orden temporal. Eventos de mismo timestamp que una
# request se procesan despuÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©s, cumpliendo event_ts < request_ts.
PRODUCT = defaultdict(lambda: {"views": 0, "carts": 0, "purchases": 0})
WINDOW = defaultdict(lambda: {"views": 0, "carts": 0, "purchases": 0, "events": deque()})
PROFILES: dict[str, dict[str, Any]] = {}
CO: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
SESSION_EVENTS: dict[str, deque] = {}
USERS_WITH_HISTORY: set[str] = set(HISTORICAL_USERS)
SESSION_EVENT_QUEUE: deque[tuple[datetime, str, str, str]] = deque()
WINDOW_EVENTS: deque[tuple[datetime, str, str]] = deque()

def prune_window(cutoff: datetime) -> None:
    while WINDOW_EVENTS and WINDOW_EVENTS[0][0] < cutoff:
        old_ts, old_pid, old_kind = WINDOW_EVENTS.popleft()
        field = {"view": "views", "add_to_cart": "carts", "purchase": "purchases"}[old_kind]
        WINDOW[old_pid][field] -= 1



def prune_co(cutoff: datetime) -> None:
    score_by_kind = {"view": 1.0, "click": 0.0, "add_to_cart": 4.0, "purchase": 15.0}
    while SESSION_EVENT_QUEUE and SESSION_EVENT_QUEUE[0][0] < cutoff:
        old_ts, sid, old_pid, old_kind = SESSION_EVENT_QUEUE.popleft()
        history = SESSION_EVENTS.get(sid)
        if not history:
            continue
        if history and history[0] == (old_ts, old_pid, old_kind):
            history.popleft()
        for other_ts, other_pid, other_kind in history:
            if old_pid == other_pid:
                continue
            if old_kind == "view":
                CO[old_pid][other_pid] -= score_by_kind[other_kind]
                if CO[old_pid][other_pid] <= 1e-9:
                    CO[old_pid].pop(other_pid, None)
            if other_kind == "view":
                CO[other_pid][old_pid] -= score_by_kind[old_kind]
                if CO[other_pid][old_pid] <= 1e-9:
                    CO[other_pid].pop(old_pid, None)
        if not history:
            SESSION_EVENTS.pop(sid, None)


def profile_for(uid: str, ts: datetime) -> dict[str, Any] | None:
    if not uid:
        return None
    if uid not in PROFILES:
        PROFILES[uid] = {
            "view_history": deque(maxlen=500), "click_history": deque(maxlen=500),
            "cart_history": deque(maxlen=500), "purchase_history": deque(maxlen=500),
            "viewed_products": {}, "n_views": 0, "n_clicks": 0,
            "n_carts": 0, "n_purchases": 0,
        }
    return PROFILES[uid]


def process_event(ev: dict[str, Any]) -> None:
    ts, pid, kind = ev["_ts"], ev["_pid"], ev["_kind"]
    info = catalog_at(pid, ts) or {}
    counts = PRODUCT[pid]
    field = {"view": "views", "add_to_cart": "carts", "purchase": "purchases"}.get(kind)
    if field is not None:
        counts[field] += 1
        prune_window(ts - timedelta(hours=24))
        WINDOW[pid][field] += 1
        WINDOW_EVENTS.append((ts, pid, kind))

    uid = ev["_uid"]
    profile = profile_for(uid, ts)
    if uid:
        USERS_WITH_HISTORY.add(uid)
    if profile is not None:
        hist_key, n_key = {
            "view": ("view_history", "n_views"),
            "click": ("click_history", "n_clicks"),
            "add_to_cart": ("cart_history", "n_carts"),
            "purchase": ("purchase_history", "n_purchases"),
        }[kind]
        record = {"product_id": pid, "ts": ts, "event_type": kind,
                  "category": info.get("category"), "subcategory": info.get("subcategory"),
                  "brand": info.get("brand")}
        profile[hist_key].append(record)
        profile[n_key] += 1
        if kind == "view":
            profile["viewed_products"][pid] = min(profile["viewed_products"].get(pid, ts), ts)

    sid = ev["_sid"]
    if sid:
        prune_co(ts - timedelta(days=20))
        history = SESSION_EVENTS.setdefault(sid, deque())
        score_by_kind = {"view": 1.0, "click": 0.0, "add_to_cart": 4.0, "purchase": 15.0}
        for old_ts, old_pid, old_kind in history:
            if old_pid == pid:
                continue
            if old_kind == "view":
                CO[old_pid][pid] += score_by_kind[kind]
            if kind == "view":
                CO[pid][old_pid] += score_by_kind[old_kind]
        history.append((ts, pid, kind))
        SESSION_EVENT_QUEUE.append((ts, sid, pid, kind))


def profile_events_before(profile: dict[str, Any] | None, request_ts: datetime, uid: str) -> list[dict[str, Any]]:
    if profile is None:
        return []
    rows = []
    for key in ("view_history", "click_history", "cart_history", "purchase_history"):
        rows.extend(e for e in profile[key] if e["ts"] < request_ts)
    for ev in rows:
        if not ev["ts"] < request_ts:
            raise RuntimeError(
                f"DATA LEAKAGE usuario={uid} request_ts={request_ts.isoformat()} "
                f"event_ts={ev['ts'].isoformat()} tipo={ev['event_type']}"
            )
    return rows


def build_user_context(uid: str, request_ts: datetime) -> dict[str, Any]:
    profile = PROFILES.get(uid) if uid else None
    events = profile_events_before(profile, request_ts, uid)
    sub_counts: dict[str, int] = defaultdict(int)
    brand_counts: dict[str, int] = defaultdict(int)
    recent_sub_counts: dict[str, int] = defaultdict(int)
    bought_subcategories: set[str] = set()
    bought_brands: set[str] = set()
    clicks = buys = engagement = 0
    last_ts = None
    recent_n = 0
    recent_cutoff = request_ts - timedelta(days=7)
    weights_by_kind = {"view": 1, "click": 2, "add_to_cart": 4, "purchase": 8}
    for event in events:
        subcat, brand = event.get("subcategory"), event.get("brand")
        if subcat:
            sub_counts[str(subcat)] += 1
        if brand:
            brand_counts[str(brand)] += 1
        if event["event_type"] == "purchase":
            if subcat:
                bought_subcategories.add(str(subcat))
            if brand:
                bought_brands.add(str(brand))
            buys += 1
        elif event["event_type"] == "click":
            clicks += 1
        engagement += weights_by_kind[event["event_type"]]
        if last_ts is None or event["ts"] > last_ts:
            last_ts = event["ts"]
        if event["ts"] >= recent_cutoff:
            recent_n += 1
            if subcat:
                recent_sub_counts[str(subcat)] += 1
    return {
        "total": len(events), "sub_counts": sub_counts, "brand_counts": brand_counts,
        "recent_sub_counts": recent_sub_counts, "recent_n": recent_n,
        "bought_subcategories": bought_subcategories, "bought_brands": bought_brands,
        "clicks": clicks, "purchases": buys, "engagement": engagement,
        "last_ts": last_ts,
        "viewed_products": profile["viewed_products"] if profile else {},
    }


def build_row(pid: str, viewed_ids: list[str], ts: datetime, user_ctx: dict[str, Any], is_cold: bool) -> list[float] | None:
    candidate = catalog_at(pid, ts)
    if candidate is None or candidate.get("is_active") is False:
        return None
    price = numeric(candidate.get("price_eur"), 25.0)
    cat, subcat, brand = candidate.get("category"), candidate.get("subcategory"), candidate.get("brand")
    total = PRODUCT[pid]
    recent = WINDOW[pid]
    views, carts, purchases = total["views"], total["carts"], total["purchases"]
    v24, c24, p24 = recent["views"], recent["carts"], recent["purchases"]
    conv = (purchases + 1.0) / (views + 50.0)
    cart_rate = (carts + 1.0) / (views + 20.0)
    created = parse_ts(candidate.get("created_at"))
    age = max(0.0, (ts - created).total_seconds() / 86400.0) if created else 0.0
    recent_pop = p24 * 10.0 + c24 * 3.0 + v24

    same_subcat = same_cat = same_brand = co_total = co_last = 0.0
    viewed_prices = []
    for idx, viewed_pid in enumerate(reversed(viewed_ids)):
        viewed_info = catalog_at(str(viewed_pid), ts)
        if not viewed_info:
            continue
        weight = 1.0 / (1.0 + 0.5 * idx)
        if viewed_info.get("subcategory") == subcat: same_subcat += weight
        if viewed_info.get("category") == cat: same_cat += weight
        if viewed_info.get("brand") == brand: same_brand += weight
        pair = CO.get(str(viewed_pid), {}).get(pid, 0.0)
        co_total += weight * pair
        if idx == 0: co_last = pair
        viewed_prices.append(numeric(viewed_info.get("price_eur"), price))
    avg_price = sum(viewed_prices) / len(viewed_prices) if viewed_prices else price
    base = [purchases, conv, cart_rate, price, price / max(avg_price, 1.0),
            same_subcat, same_cat, same_brand, co_last, co_total,
            float(pid in [str(x) for x in viewed_ids]), 1.0, p24, v24, c24, age, recent_pop]

    denom = user_ctx["total"]
    sub_aff = user_ctx["sub_counts"].get(str(subcat), 0) / denom if denom and subcat else 0.0
    brand_aff = user_ctx["brand_counts"].get(str(brand), 0) / denom if denom and brand else 0.0
    # No existe un precio hist?rico de compra en el esquema usado: neutrales expl?citos.
    price_fit = price_percentile = 0.5
    purchase_rate = user_ctx["purchases"] / user_ctx["clicks"] if user_ctx["clicks"] else 0.0
    bought_similar = float((subcat and str(subcat) in user_ctx["bought_subcategories"]) or
                           (brand and str(brand) in user_ctx["bought_brands"]))
    recent_aff = (user_ctx["recent_sub_counts"].get(str(subcat), 0) / user_ctx["recent_n"]
                  if user_ctx["recent_n"] and subcat else 0.0)
    first_view = user_ctx["viewed_products"].get(pid)
    seen_before = float(first_view is not None and first_view < ts)
    last_interaction = user_ctx["last_ts"]
    since_days = max(0.0, (ts - last_interaction).total_seconds() / 86400.0) if last_interaction else 0.0
    row = base + [sub_aff, brand_aff, price_fit, purchase_rate, bought_similar,
                  recent_aff, seen_before, since_days, float(user_ctx["engagement"]),
                  price_percentile, float(is_cold)]
    if len(row) != 28:
        raise RuntimeError(f"Vector LGBM8 incorrecto para {pid}: {len(row)} features")
    return row

print("[LGBM8 4/7] Reconstruyendo features as-of request y labels...")
X: list[list[float]] = []
y: list[int] = []
weights: list[float] = []
request_groups: list[dict[str, Any]] = []
event_iter = iter(iter_events())
next_event = next(event_iter, None)
skipped_no_catalog = 0
for req_idx, req in enumerate(sessions):
    request_ts = req["_ts"]
    while next_event is not None and next_event["_ts"] < request_ts:
        process_event(next_event)
        next_event = next(event_iter, None)
    prune_window(request_ts - timedelta(hours=24))
    prune_co(request_ts - timedelta(days=20))
    uid = req["_uid"]
    user_ctx = build_user_context(uid, request_ts)
    cold = uid not in USERS_WITH_HISTORY
    recent_views = [v for v in parse_array(req.get("recent_views")) if isinstance(v, dict) and v.get("product_id") is not None]
    viewed_ids = [str(v["product_id"]) for v in recent_views]
    shown = [str(pid) for pid in parse_array(req.get("shown")) if pid is not None and str(pid)]
    props = parse_array(req.get("propensity"))
    clicked = str(req.get("clicked") or "")
    purchased = str(req.get("purchased") or "")
    start = len(X)
    candidates, labels, props_used = [], [], []
    for idx, pid in enumerate(shown):
        row = build_row(pid, viewed_ids, request_ts, user_ctx, cold)
        if row is None:
            skipped_no_catalog += 1
            continue
        is_buy, is_click = pid == purchased, pid == clicked
        label = int(is_buy or is_click)
        prop = numeric(props[idx], 0.1) if idx < len(props) else 0.1
        ips_weight = min(1.0 / max(prop, 0.02), 10.0)
        if is_buy:
            ips_weight *= 4.0
        if cold:
            ips_weight *= COLD_START_AUGMENT_WEIGHT
        X.append(row); y.append(label); weights.append(ips_weight)
        candidates.append(pid); labels.append(label); props_used.append(max(prop, 0.02))
    if candidates:
        request_groups.append({"idx": req_idx, "start": start, "end": len(X),
                               "candidates": candidates, "labels": labels,
                               "propensity": props_used, "cold": cold,
                               "clicked": clicked, "purchased": purchased})
if skipped_no_catalog:
    print(f"  Ejemplos omitidos por falta de snapshot disponible a tiempo: {skipped_no_catalog}")
if not X:
    raise RuntimeError("No quedaron ejemplos con catÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡logo histÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³rico disponible; no se entrenÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³ ningÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Âºn modelo.")
X_np = np.asarray(X, dtype=np.float32)
y_np = np.asarray(y, dtype=np.int32)
w_np = np.asarray(weights, dtype=np.float32)
if X_np.ndim != 2 or X_np.shape[1] != 28 or len(X_np) != len(y_np) or len(X_np) != len(w_np):
    raise RuntimeError(f"Dataset incompatible: X={X_np.shape}, y={y_np.shape}, weights={w_np.shape}")

# Split temporal se basa en timestamps originales de requests; filas no se mezclan.
row_train = np.zeros(len(X_np), dtype=bool)
row_val = np.zeros(len(X_np), dtype=bool)
row_test = np.zeros(len(X_np), dtype=bool)
for group in request_groups:
    ts = sessions[group["idx"]]["_ts"]
    mask = slice(group["start"], group["end"])
    if ts < train_cut: row_train[mask] = True
    elif ts < valid_cut: row_val[mask] = True
    else: row_test[mask] = True
if not row_train.any() or not row_val.any() or not row_test.any():
    raise RuntimeError("El split temporal dejÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³ train, validation o test vacÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­o; revisa el rango temporal.")
if len(np.unique(y_np[row_train])) < 2:
    raise RuntimeError("Train solo contiene una clase; hacen falta clicks/compras y negativos en el periodo de train.")
COLD_NEUTRAL_FEATURES = np.asarray([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0], dtype=np.float32)
X_cold_np = X_np.copy()
X_cold_np[:, 17:28] = COLD_NEUTRAL_FEATURES
cold_train = row_train & (X_np[:, 27] > 0.5)
cold_train_positive = int(np.sum(y_np[cold_train] == 1))
synthetic_cold_rows = int(row_train.sum())
print(f"  Matriz: {X_np.shape}; train={row_train.sum()}, validation={row_val.sum()}, test={row_test.sum()}")
print(f"  Cold-start observados en train: {int(cold_train.sum())} filas, {cold_train_positive} positivas")
print(f"  Augmentaci?n cold-start simulada: {synthetic_cold_rows} filas adicionales, peso x{COLD_START_AUGMENT_WEIGHT:g}")

print("[LGBM8 5/7] Entrenando con IPS y peso de compra heredados...")
model = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.05, num_leaves=31,
                           subsample_for_bin=50000, random_state=42, n_jobs=-1)
X_fit = np.concatenate([X_np[row_train], X_cold_np[row_train]], axis=0)
y_fit = np.concatenate([y_np[row_train], y_np[row_train]], axis=0)
w_fit = np.concatenate([w_np[row_train], w_np[row_train] * COLD_START_AUGMENT_WEIGHT], axis=0)
model.fit(X_fit, y_fit, sample_weight=w_fit)
if getattr(model, "n_features_in_", None) != 28:
    raise RuntimeError("El clasificador LGBM8 no quedÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³ con 28 features")

try:
    baseline = joblib.load(BASELINE_PATH)
    if getattr(baseline, "n_features_in_", None) != 17:
        raise RuntimeError("El artefacto de LGBM7 no tiene 17 features; no se modifica.")
except FileNotFoundError:
    baseline = None
    print("  Aviso: no se encontrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³ modelo_lgbm7.pkl; se omite comparaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n con baseline.")


def evaluate(model_obj: Any, width: int, groups: list[dict[str, Any]], mask: np.ndarray,
             cold_mode: bool = False) -> dict[str, float]:
    selected = [g for g in groups if mask[g["start"]:g["end"]].any()]
    if not selected:
        return {"requests": 0.0}
    hits10: list[float] = []
    hits1: list[float] = []
    cold_hits10: list[float] = []
    cold_hits1: list[float] = []
    warm_hits10: list[float] = []
    ips_click_rewards: list[float] = []
    snips_num = snips_den = 0.0
    recommended_ids: set[str] = set()
    candidate_ids: set[str] = set()
    diversity: list[float] = []
    latencies: list[float] = []
    candidate_counts: list[int] = []
    for g in selected:
        start, end = g["start"], g["end"]
        ix = np.arange(start, end)
        ix = ix[mask[ix]]
        if not len(ix):
            continue
        candidate_counts.append(len(ix))
        candidate_ids.update(g["candidates"])
        t0 = time.perf_counter()
        input_rows = X_cold_np[ix, :width] if cold_mode else X_np[ix, :width]
        scores = model_obj.predict_proba(input_rows)[:, 1]
        latencies.append((time.perf_counter() - t0) * 1000.0)
        order = np.argsort(scores)[::-1][:10]
        chosen_ids = [g["candidates"][int(j)] for j in order]
        chosen_labels = [g["labels"][int(j)] for j in order]
        chosen_props = [g["propensity"][int(j)] for j in order]
        hit10 = float(any(chosen_labels))
        hit1 = float(bool(chosen_labels) and chosen_labels[0] == 1)
        hits10.append(hit10)
        hits1.append(hit1)
        if cold_mode or g["cold"]:
            cold_hits10.append(hit10)
            cold_hits1.append(hit1)
        else:
            warm_hits10.append(hit10)
        recommended_ids.update(chosen_ids)
        cats = [catalog_at(pid, sessions[g["idx"]]["_ts"]) for pid in chosen_ids]
        diversity.append(float(len({c.get("category") for c in cats if c and c.get("category")})))
        click_pos = next((j for j, pid in enumerate(chosen_ids) if pid == g["clicked"]), None)
        click_weight = min(1.0 / max(chosen_props[click_pos], 0.02), 10.0) if click_pos is not None else 0.0
        ips_click_rewards.append(click_weight)
        snips_num += click_weight
        snips_den += sum(min(1.0 / max(prop, 0.02), 10.0) for prop in chosen_props)
    count = len(hits10)
    return {
        "requests": float(count),
        "mean_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "cold_start_requests": float(len(cold_hits10)),
        "hit_at_1": float(np.mean(hits1)) if hits1 else 0.0,
        "hit_at_10": float(np.mean(hits10)) if hits10 else 0.0,
        "ips_ctr": float(sum(ips_click_rewards) / count) if count else 0.0,
        "snips": float(snips_num / snips_den) if snips_den else 0.0,
        "cold_start_hit_at_1": float(np.mean(cold_hits1)) if cold_hits1 else 0.0,
        "cold_start_hit_at_10": float(np.mean(cold_hits10)) if cold_hits10 else 0.0,
        "warm_start_hit_at_10": float(np.mean(warm_hits10)) if warm_hits10 else 0.0,
        "catalog_coverage": float(len(recommended_ids) / len(candidate_ids)) if candidate_ids else 0.0,
        "mean_category_diversity_at_10": float(np.mean(diversity)) if diversity else 0.0,
        "p95_batch_predict_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def eval_groups(mask: np.ndarray) -> list[dict[str, Any]]:
    return [g for g in request_groups if mask[g["start"]:g["end"]].any()]

print("[LGBM8 6/7] Comparando LGBM7 y LGBM8 en validation/test...")
metrics: dict[str, Any] = {}
for split_name, mask in (("validation", row_val), ("test", row_test)):
    groups = eval_groups(mask)
    if baseline is not None:
        metrics[f"lgbm7_{split_name}"] = evaluate(baseline, 17, groups, mask)
    metrics[f"lgbm8_{split_name}"] = evaluate(model, 28, groups, mask)
    metrics[f"lgbm7_{split_name}_simulated_cold"] = evaluate(baseline, 17, groups, mask, cold_mode=True) if baseline is not None else {"requests": float(len(groups))}
    metrics[f"lgbm8_{split_name}_simulated_cold"] = evaluate(model, 28, groups, mask, cold_mode=True)

# Acumula todos los eventos para guardar seÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â±ales actuales de serving en el bundle.
while next_event is not None:
    process_event(next_event)
    next_event = next(event_iter, None)
now = datetime.now(timezone.utc)
prune_window(now - timedelta(hours=24))
prune_co(now - timedelta(days=20))
serving_catalog: dict[str, dict[str, Any]] = {}
for pid in base_catalog:
    latest = catalog_at(pid, now)
    if latest is not None:
        latest = dict(latest)
        latest["product_id"] = pid
        serving_catalog[pid] = latest
serving_stats = {}
for pid, counts in PRODUCT.items():
    w = WINDOW[pid]
    v, c, p = counts["views"], counts["carts"], counts["purchases"]
    v24, c24, p24 = w["views"], w["carts"], w["purchases"]
    created = parse_ts(serving_catalog.get(pid, {}).get("created_at"))
    serving_stats[pid] = {
        "views": float(v), "carts": float(c), "purchases": float(p),
        "conv_rate": (p + 1.0) / (v + 50.0), "cart_rate": (c + 1.0) / (v + 20.0),
        "views_24h": float(v24), "carts_24h": float(c24), "purchases_24h": float(p24),
        "recent_conv_rate": (p24 + 1.0) / (v24 + 20.0),
        "recent_popularity": float(p24 * 10 + c24 * 3 + v24),
        "product_age_days": max(0.0, (now - created).total_seconds() / 86400.0) if created else 0.0,
    }
for pid, info in serving_catalog.items():
    serving_stats.setdefault(pid, {"views": 0.0, "carts": 0.0, "purchases": 0.0,
        "conv_rate": 0.01, "cart_rate": 0.02, "views_24h": 0.0, "carts_24h": 0.0,
        "purchases_24h": 0.0, "recent_conv_rate": 0.01, "recent_popularity": 0.0,
        "product_age_days": 0.0})
serving_co = {a: dict(sorted(b.items(), key=lambda kv: kv[1], reverse=True)[:25]) for a, b in CO.items()}

joblib.dump(model, MODEL_PATH)
joblib.dump({
    "catalog": serving_catalog,
    "prod_stats": serving_stats,
    "co_matrix": serving_co,
    "feature_names": FEATURE_NAMES,
    "n_features": 28,
    "training_metadata": {
        "train_end_exclusive": train_cut.isoformat(),
        "validation_end_exclusive": valid_cut.isoformat(),
        "test_start": valid_cut.isoformat(),
        "request_count": len(sessions),
        "history_days_before_first_request": HISTORY_DAYS,
        "cold_start_augmentation": "duplicate train rows with personal features masked and user_is_cold_start=1",
        "cold_start_augmentation_weight": COLD_START_AUGMENT_WEIGHT,
        "cold_start_neutral_features": COLD_NEUTRAL_FEATURES.tolist(),
        "event_count": total_event_count,
        "timestamp_source": "shop.history_sessions.ts y shop.history_sessions.user_id",
        "cold_start_source": "sample-user historical aggregate before replay window plus strict event replay",
        "catalog_updates_used": True,
        "profile_history_limit_per_event_type": 500,
        "purchase_price_available": False,
        "metrics": metrics,
    },
}, FEATURES_PATH)

print("[LGBM8 7/7] Entrenamiento terminado")
print(f"Modelo: {MODEL_PATH}")
print(f"Features/estado: {FEATURES_PATH}")
print("Features: 28; artifacts LGBM7 no se han escrito.")
for name, values in metrics.items():
    print(name + ": " + ", ".join(f"{k}={v:.5f}" for k, v in values.items()))
