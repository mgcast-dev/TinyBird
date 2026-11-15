"""Recomendador en Tiempo Real con LightGBM (17 features + Refresco en Vivo de EstadÃƒÆ’Ã‚Â­sticas)."""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import requests

# =====================================================================
# 1. CONFIGURACIÃƒÆ’Ã¢â‚¬Å“N Y CARGA DEL MODELO DE 17 FEATURES DESDE DISCO
# =====================================================================
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

# Busca en starter/models/ y, si no estÃƒÆ’Ã‚Â¡ ahÃƒÆ’Ã‚Â­, en la carpeta actual
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "modelo_lgbm7.pkl" if (MODELS_DIR / "modelo_lgbm7.pkl").exists() else Path("modelo_lgbm7.pkl")
FEAT_PATH = MODELS_DIR / "features_data7.pkl" if (MODELS_DIR / "features_data7.pkl").exists() else Path("features_data7.pkl")

print(f"Cargando modelo LightGBM (17 features) desde {MODEL_PATH}...")
try:
    LGB_MODEL = joblib.load(MODEL_PATH)
    FEAT_DATA = joblib.load(FEAT_PATH)
    CATALOG: dict[str, Record] = FEAT_DATA["catalog"]
    PROD_STATS: dict[str, dict[str, float]] = FEAT_DATA["prod_stats"]
    CO_MATRIX: dict[str, dict[str, float]] = FEAT_DATA["co_matrix"]
    print("Ãƒâ€šÃ‚Â¡Modelo de 17 features cargado correctamente!")
except Exception as e:
    print(f"[AVISO CRÃƒÆ’Ã‚ÂTICO]: No se encontraron los archivos .pkl ({e}).")
    LGB_MODEL = None
    FEAT_DATA, CATALOG, PROD_STATS, CO_MATRIX = {}, {}, {}, {}

# Listas rÃƒÆ’Ã‚Â¡pidas de candidatos histÃƒÆ’Ã‚Â³ricos
TOP_GLOBAL_HIST = [
    pid for pid, _ in sorted(PROD_STATS.items(), key=lambda x: x[1].get("purchases", 0.0), reverse=True)
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

# Lista dinÃƒÆ’Ã‚Â¡mica con los productos mÃƒÆ’Ã‚Â¡s vendidos en vivo (se actualiza cada 30s)
TOP_TRENDING_LIVE: list[str] = []


# =====================================================================
# 2. ACTUALIZADOR EN SEGUNDO PLANO (Refresca Stock + Features en Vivo)
# =====================================================================
ACTIVE_IDS: set[str] = {pid for pid, c in CATALOG.items() if c.get("is_active", True)}
# ArtÃ­culos descubiertos despuÃ©s de cargar el snapshot.
NEW_PRODUCT_IDS: set[str] = set()
NEW_PRODUCT_FIRST_SEEN: dict[str, datetime] = {}
HERCULES_MODEL_NAME = "Hercules"

CATALOG_ACTIVE_SQL = """
SELECT product_id,
       argMax(category, v)    AS category,
       argMax(subcategory, v) AS subcategory,
       argMax(brand, v)       AS brand,
       argMax(price_eur, v)   AS price_eur,
       argMax(is_active, v)   AS is_active
FROM (
    SELECT product_id::String AS product_id, category::String AS category,
           subcategory::String AS subcategory, brand::String AS brand,
           price_eur::Float64 AS price_eur, is_active::Bool AS is_active, '' AS v
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String, category::String, subcategory::String, brand::String,
           price_eur::Float64, is_active::Bool, toString(ts)
    FROM shop.catalog_updates
)
GROUP BY product_id
"""

# Consulta a shop.events (eventos en vivo) para refrescar las variables de tendencia
LIVE_STATS_SQL = """
SELECT product_id::String AS product_id,
       countIf(event_type = 'view') AS views_live,
       countIf(event_type = 'add_to_cart') AS carts_live,
       countIf(event_type = 'purchase') AS purchases_live
FROM shop.events
WHERE ts >= now() - INTERVAL 24 HOUR
GROUP BY product_id
"""

NEW_REQUESTS_SQL = """
SELECT request_id, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 60
"""


def bg_worker() -> None:
    """Hilo en segundo plano: actualiza stock cada 2s y recalcula features en vivo cada 30s."""
    global ACTIVE_IDS, TOP_TRENDING_LIVE
    last_live = 0.0
    while True:
        try:
            # A. Cada 2 segundos: Actualizar stock, precios en vivo y productos nuevos
            r = http_bg.post(f"{API}/v1/query", json={"sql": CATALOG_ACTIVE_SQL}, timeout=8)
            if r.ok:
                rows = r.json()["data"]
                ACTIVE_IDS = {row["product_id"] for row in rows if row.get("is_active")}
                for row in rows:
                    pid = row["product_id"]
                    if pid not in CATALOG:
                        # Ãƒâ€šÃ‚Â¡Producto nuevo detectado en vivo! Lo aÃƒÆ’Ã‚Â±adimos al catÃƒÆ’Ã‚Â¡logo y a PROD_STATS
                        CATALOG[pid] = row
                        NEW_PRODUCT_IDS.add(pid)
                        NEW_PRODUCT_FIRST_SEEN[pid] = datetime.now(timezone.utc)
                        PROD_STATS[pid] = {
                            "views": 0.0, "carts": 0.0, "purchases": 0.0,
                            "conv_rate": 0.02, "cart_rate": 0.04,
                            "views_24h": 0.0, "carts_24h": 0.0, "purchases_24h": 0.0,
                            "recent_conv_rate": 0.02, "recent_popularity": 5.0,
                            "product_age_days": 0.0,  # Edad 0 dÃƒÆ’Ã‚Â­as = ReciÃƒÆ’Ã‚Â©n lanzado
                        }
                    else:
                        CATALOG[pid]["price_eur"] = row.get("price_eur", CATALOG[pid].get("price_eur", 25.0))

            # B. Cada 30 segundos: Recalcular las features de actividad reciente desde shop.events
            if time.time() - last_live >= 30.0:
                last_live = time.time()
                r_live = http_bg.post(f"{API}/v1/query", json={"sql": LIVE_STATS_SQL}, timeout=10)
                if r_live.ok:
                    live_rows = r_live.json()["data"]
                    trending_scores: list[tuple[float, str]] = []

                    for row in live_rows:
                        pid = row["product_id"]
                        v_live = float(row["views_live"])
                        c_live = float(row["carts_live"])
                        p_live = float(row["purchases_live"])

                        # Calculamos la popularidad y conversiÃƒÆ’Ã‚Â³n en vivo con la misma fÃƒÆ’Ã‚Â³rmula del entrenamiento
                        rec_pop = p_live * 10.0 + c_live * 3.0 + v_live * 1.0
                        live_conv = (p_live + 1.0) / (v_live + 20.0)
                        live_cart = (c_live + 1.0) / (v_live + 15.0)

                        if pid not in PROD_STATS:
                            PROD_STATS[pid] = {
                                "base_purchases": 0.0, "base_conv": live_conv, "base_cart": live_cart,
                                "views": v_live, "carts": c_live, "purchases": p_live,
                                "conv_rate": live_conv, "cart_rate": live_cart,
                                "views_24h": v_live, "carts_24h": c_live, "purchases_24h": p_live,
                                "recent_conv_rate": live_conv, "recent_popularity": rec_pop,
                                "product_age_days": 0.0,
                            }
                        else:
                            st = PROD_STATS[pid]
                            # Guardamos la foto original del .pkl la primera vez para no acumular de mÃƒÆ’Ã‚Â¡s cada 30s
                            if "base_purchases" not in st:
                                st["base_purchases"] = st["purchases"]
                                st["base_conv"] = st["conv_rate"]
                                st["base_cart"] = st["cart_rate"]

                            # 1. Variables antiguas actualizadas con lo ocurrido en vivo
                            st["purchases"] = st["base_purchases"] + p_live
                            st["conv_rate"] = 0.7 * st["base_conv"] + 0.3 * live_conv
                            st["cart_rate"] = 0.7 * st["base_cart"] + 0.3 * live_cart

                            # 2. Variables nuevas de tendencia 24h actualizadas en vivo
                            st["views_24h"] = v_live
                            st["carts_24h"] = c_live
                            st["purchases_24h"] = p_live
                            st["recent_popularity"] = rec_pop
                            st["recent_conv_rate"] = live_conv

                        trending_scores.append((rec_pop, pid))

                    # Guardamos el Top 15 de productos de moda ahora mismo para meterlos como candidatos
                    trending_scores.sort(reverse=True)
                    TOP_TRENDING_LIVE = [pid for _, pid in trending_scores[:15]]

        except Exception as e:
            print(f"[BG aviso]: {e}")
        time.sleep(2.0)


# =====================================================================
# 3. CONSTRUCCIÃƒÆ’Ã¢â‚¬Å“N DE LAS 17 FEATURES + SELECCIÃƒÆ’Ã¢â‚¬Å“N DE CANDIDATOS
# =====================================================================
def build_feature_row(cand_id: str, viewed_ids: list[str]) -> list[float]:
    """Construye las 17 variables exactas que espera modelo_lgbm7.pkl leyendo la RAM actualizada."""
    c_info = CATALOG.get(cand_id, {})
    st = PROD_STATS.get(cand_id, {
        "purchases": 0.0, "conv_rate": 0.01, "cart_rate": 0.02,
        "purchases_24h": 0.0, "views_24h": 0.0, "carts_24h": 0.0,
        "product_age_days": 0.0, "recent_popularity": 0.0,
    })

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

    return [
        st["purchases"],          # 0
        st["conv_rate"],          # 1
        st["cart_rate"],          # 2
        c_price,                  # 3
        price_ratio,              # 4
        same_subcat,              # 5
        same_cat,                 # 6
        same_brand,               # 7
        co_score_last,            # 8
        co_score_total,           # 9
        already_viewed,           # 10
        1.0,                      # 11: Slot 1 fijo para Position Debiasing
        st["purchases_24h"],      # 12: Ãƒâ€šÃ‚Â¡Actualizada cada 30s en vivo!
        st["views_24h"],          # 13: Ãƒâ€šÃ‚Â¡Actualizada cada 30s en vivo!
        st["carts_24h"],          # 14: Ãƒâ€šÃ‚Â¡Actualizada cada 30s en vivo!
        st["product_age_days"],   # 15: 0.0 si el producto acaba de salir en vivo
        st["recent_popularity"],  # 16: Ãƒâ€šÃ‚Â¡Actualizada cada 30s en vivo!
    ]


def get_candidates(viewed_ids: list[str], active: set[str], request_id: str = "") -> list[str]:
    """Selecciona hasta 40 candidatos combinando sesiÃƒÆ’Ã‚Â³n, co-compras, subcategorÃƒÆ’Ã‚Â­a y tendencias en vivo."""
    cands: list[str] = []
    seen_c: set[str] = set()

    def add_c(pid: str) -> None:
        if pid in active and pid not in seen_c:
            seen_c.add(pid)
            cands.append(pid)

    # 1. Re-targeting: los ÃƒÆ’Ã‚Âºltimos 2 productos que el cliente acaba de mirar en su sesiÃƒÆ’Ã‚Â³n
    for v_id in reversed(viewed_ids[-2:]):
        add_c(v_id)

    # 2. Co-Compras y misma SubcategorÃƒÆ’Ã‚Â­a de lo visto en la sesiÃƒÆ’Ã‚Â³n
    for v_id in reversed(viewed_ids):
        for cid in CO_TOP_LIST.get(v_id, [])[:10]:
            add_c(cid)
        sc = CATALOG.get(v_id, {}).get("subcategory", "")
        if sc:
            for cid in TOP_BY_SUBCAT.get(sc, [])[:8]:
                add_c(cid)
        if len(cands) >= 26:
            break

    # 3. Top 5 productos que son tendencia ahora mismo en vivo (shop.events)
    for cid in TOP_TRENDING_LIVE[:5]:
        add_c(cid)

    # 4. Rellenar hasta 40 con superventas histÃƒÆ’Ã‚Â³ricos
    # Pool pequeÃ±o de artÃ­culos descubiertos en caliente; el reranker controla su exposiciÃ³n.
    for cid in sorted(NEW_PRODUCT_IDS, key=lambda pid: hashlib.sha256(f"{request_id}:{pid}".encode()).digest())[:8]:
        add_c(cid)

    for cid in TOP_GLOBAL_HIST:
        add_c(cid)
        if len(cands) >= 40:
            break

    return cands


# =====================================================================
# 4. COMPATIBILIDAD CON OFFLINE_EVAL.IPYNB Y PREDICCIÃƒÆ’Ã¢â‚¬Å“N EN LOTE
# =====================================================================
def best_sellers(until: str | None = None) -> list[str]:
    return TOP_GLOBAL_HIST if TOP_GLOBAL_HIST else list(CATALOG.keys())


def recommend(recent_views: list[Record], best: list[str] | None = None, unavailable: set[str] = frozenset()) -> list[str]:
    viewed_ids = [
        v["product_id"] for v in (recent_views or [])
        if isinstance(v, dict) and "product_id" in v
    ]
    active = set(CATALOG.keys()) - set(unavailable)

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

    if len(top_10) < 10:
        fallback_list = best if best is not None else TOP_GLOBAL_HIST
        for p in fallback_list:
            if p not in top_10 and p not in unavailable:
                top_10.append(p)
            if len(top_10) == 10:
                break

    return top_10[:10]


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _reserve_new_product_slots(ranked: list[str], candidates: list[str], cold_user: bool,
                               request_id: str) -> list[str]:
    """Da exposiciÃ³n limitada a productos nuevos; mantiene el resto del ranking LGBM."""
    slots = 2 if cold_user else 1
    eligible = []
    for pid in candidates:
        if pid not in NEW_PRODUCT_IDS:
            continue
        first_seen = NEW_PRODUCT_FIRST_SEEN.get(pid)
        if first_seen is None or (datetime.now(timezone.utc) - first_seen).total_seconds() > 14 * 86400:
            continue
        item = CATALOG.get(pid, {})
        try:
            price = float(item.get("price_eur", 0) or 0)
        except (TypeError, ValueError):
            price = 0.0
        if price > 0 and (item.get("category") or item.get("subcategory")):
            eligible.append(pid)
    if not eligible:
        return ranked[:10]
    import hashlib
    eligible.sort(key=lambda pid: hashlib.sha256(f"{request_id}:{pid}".encode()).digest())
    explore = eligible[:slots]
    rest = [pid for pid in ranked if pid not in eligible]
    return (rest[:10 - len(explore)] + explore)[:10]

def _hercules_rank(model_ranked: list[str], candidates: list[str], popularity: list[str],
                    unavailable: set[str] = frozenset()) -> list[str]:
    """Keep the proven top 10 intact; let the model rank additional and fallback items."""
    anchor = []
    candidate_set = set(candidates)
    for pid in popularity:
        if pid in candidate_set and pid not in unavailable and pid not in anchor:
            anchor.append(pid)
        if len(anchor) >= 10:
            break
    if len(anchor) < 10:
        return model_ranked
    return anchor + [pid for pid in model_ranked if pid not in set(anchor)]


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
            "model_version": "v6-lgbm17-live",
        })

    return submissions


def main() -> None:
    threading.Thread(target=bg_worker, daemon=True).start()
    seen: set[str] = set()
    print("Ãƒâ€šÃ‚Â¡Recomendador v6-lgbm17-live activo y refrescando features cada 30s!")

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
                print(f"answered {len(answers)} (v6-lgbm17-live)")
        except Exception as e:
            print(f"error: {e}")
            time.sleep(0.5)

        time.sleep(0.25)


# =====================================================================
# LGBM 8: perfiles temporales en memoria y 10 features personalizadas
# LGBM 7 se mantiene en sus propios archivos modelo_lgbm7.pkl/features_data7.pkl.
# =====================================================================
from collections import deque
from datetime import timedelta
import math

LGB8_MODEL_PATH = MODELS_DIR / "modelo_lgbm8.pkl"
LGB8_FEAT_PATH = MODELS_DIR / "features_data8.pkl"
LGB8_MODEL = None
LGB8_FEAT_DATA: dict[str, Any] = {}

try:
    LGB8_MODEL = joblib.load(LGB8_MODEL_PATH)
    LGB8_FEAT_DATA = joblib.load(LGB8_FEAT_PATH)
    if getattr(LGB8_MODEL, "n_features_in_", None) != 28:
        raise ValueError("modelo_lgbm8.pkl debe aceptar exactamente 28 features")
    # El bundle de LGBM 8 se basa en el catÃƒÆ’Ã‚Â¡logo/estadÃƒÆ’Ã‚Â­sticas de LGBM 7.
    CATALOG = LGB8_FEAT_DATA.get("catalog", CATALOG)
    PROD_STATS = LGB8_FEAT_DATA.get("prod_stats", PROD_STATS)
    CO_MATRIX = LGB8_FEAT_DATA.get("co_matrix", CO_MATRIX)
except FileNotFoundError:
    print(f"[LGBM8] AÃƒÆ’Ã‚Âºn no existe el modelo separado en {LGB8_MODEL_PATH}; se usarÃƒÆ’Ã‚Â¡ fallback.")
except Exception as exc:
    print(f"[LGBM8] No se pudo cargar el modelo 8: {exc}")
    LGB8_MODEL = None

PROFILE_RECENT_LIMIT = 500
PROFILE_POLL_SECONDS = 10.0
PROFILE_EVENT_LOOKBACK_MINUTES = 1
PROFILE_EVENTS_SQL = f"""
SELECT user_id::String AS user_id,
       product_id::String AS product_id,
       event_type::String AS event_type,
       ts
FROM shop.events
WHERE ts >= now() - INTERVAL {PROFILE_EVENT_LOOKBACK_MINUTES} MINUTE
  AND user_id IS NOT NULL
  AND event_type IN ('view', 'click', 'add_to_cart', 'purchase')
ORDER BY ts
LIMIT 20000
"""
NEW_REQUESTS_SQL = """
SELECT request_id, user_id::String AS user_id, ts, recent_views, context, is_new_user::Bool AS is_new_user
FROM shop.requests
ORDER BY ts DESC
LIMIT 60
"""

# Agregados sin lÃƒÆ’Ã‚Â­mite temporal + cola acotada de eventos recientes.
USER_PROFILES: dict[str, dict[str, Any]] = {}
_EVENT_KEYS: set[tuple[str, str, str, str]] = set()
_EVENT_KEY_ORDER: deque[tuple[str, str, str, str]] = deque()
_EVENT_LOCK = threading.RLock()


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _empty_profile(now: datetime) -> dict[str, Any]:
    return {
        "first_seen": now,
        "last_seen": now,
        "n_requests": 0,
        "view_history": deque(maxlen=PROFILE_RECENT_LIMIT),
        "click_history": deque(maxlen=PROFILE_RECENT_LIMIT),
        "cart_history": deque(maxlen=PROFILE_RECENT_LIMIT),
        "purchase_history": deque(maxlen=PROFILE_RECENT_LIMIT),
        "n_views": 0,
        "n_clicks": 0,
        "n_carts": 0,
        "n_purchases": 0,
        "category_counts": defaultdict(float),
        "subcategory_counts": defaultdict(float),
        "brand_counts": defaultdict(float),
        "purchase_prices": deque(maxlen=PROFILE_RECENT_LIMIT),
        "viewed_products": {},
        "device": None,
        "region": None,
        "page": None,
        "interaction_level": "new",
        "last_interaction": None,
    }


def _get_profile(user_id: Any, now: datetime | None = None) -> dict[str, Any] | None:
    if user_id is None or str(user_id).strip() in {"", "None", "null"}:
        return None
    uid = str(user_id)
    current = now or datetime.now(timezone.utc)
    with _EVENT_LOCK:
        profile = USER_PROFILES.get(uid)
        if profile is None:
            profile = _empty_profile(current)
            USER_PROFILES[uid] = profile
        return profile


def _interaction_level(profile: dict[str, Any]) -> str:
    n = profile["n_views"] + profile["n_clicks"] + profile["n_carts"] + profile["n_purchases"]
    if n == 0:
        return "new"
    if n < 3:
        return "light"
    if n < 15:
        return "engaged"
    return "loyal"


def _record_user_event(row: Record) -> None:
    uid = row.get("user_id")
    pid = row.get("product_id")
    event_type = str(row.get("event_type", "")).lower()
    event_ts = _parse_ts(row.get("ts"))
    if uid is None or pid is None or event_ts is None:
        return
    kind_to_history = {
        "view": ("view_history", "n_views"),
        "click": ("click_history", "n_clicks"),
        "add_to_cart": ("cart_history", "n_carts"),
        "purchase": ("purchase_history", "n_purchases"),
    }
    if event_type not in kind_to_history:
        return
    uid, pid = str(uid), str(pid)
    key = (uid, pid, event_type, event_ts.isoformat())
    with _EVENT_LOCK:
        if key in _EVENT_KEYS:
            return
        _EVENT_KEYS.add(key)
        _EVENT_KEY_ORDER.append(key)
        if len(_EVENT_KEY_ORDER) > 50000:
            _EVENT_KEYS.discard(_EVENT_KEY_ORDER.popleft())
        profile = _get_profile(uid, event_ts)
        profile["first_seen"] = min(profile["first_seen"], event_ts)
        history_key, count_key = kind_to_history[event_type]
        ev: Record = {"product_id": pid, "ts": event_ts, "event_type": event_type}
        info = CATALOG.get(pid, {})
        for field in ("category", "subcategory", "brand"):
            ev[field] = info.get(field)
        profile[history_key].append(ev)
        profile[count_key] += 1
        profile["last_seen"] = max(profile["last_seen"], event_ts)
        profile["last_interaction"] = max(profile["last_interaction"] or event_ts, event_ts)
        if event_type == "view":
            profile["viewed_products"][pid] = min(profile["viewed_products"].get(pid, event_ts), event_ts)
        # Sin precio histÃƒÂ³rico disponible, las features de precio quedan neutrales.
        for field, counter in (("category", "category_counts"), ("subcategory", "subcategory_counts"), ("brand", "brand_counts")):
            value = info.get(field)
            if value:
                profile[counter][str(value)] += 1.0
        profile["interaction_level"] = _interaction_level(profile)


def _profile_event_worker() -> None:
    while True:
        try:
            rows = http_bg.post(
                f"{API}/v1/query", json={"sql": PROFILE_EVENTS_SQL}, timeout=8
            )
            rows.raise_for_status()
            for event in rows.json().get("data", []):
                _record_user_event(event)
        except Exception as exc:
            print(f"[LGBM8 perfil] {exc}")
        time.sleep(PROFILE_POLL_SECONDS)


def update_user_request(req: Record) -> None:
    request_ts = _parse_ts(req.get("ts")) or datetime.now(timezone.utc)
    profile = _get_profile(req.get("user_id"), request_ts)
    if profile is None:
        return
    profile["n_requests"] += 1
    profile["last_seen"] = max(profile["last_seen"], request_ts)
    context = req.get("context") or {}
    if isinstance(context, dict):
        for key, target in (("device", "device"), ("region", "region"), ("page", "page")):
            if context.get(key) is not None:
                profile[target] = context[key]


def _profile_events_before(profile: dict[str, Any], request_ts: datetime) -> list[Record]:
    # Cola acotada; la condiciÃƒÆ’Ã‚Â³n estricta excluye eventos simultÃƒÆ’Ã‚Â¡neos o futuros.
    all_recent = []
    for key in ("view_history", "click_history", "cart_history", "purchase_history"):
        all_recent.extend(ev for ev in profile[key] if ev["ts"] < request_ts)
    all_recent.sort(key=lambda ev: ev["ts"])
    for ev in all_recent:
        if not ev["ts"] < request_ts:
            raise ValueError(
                "Data leakage en perfil: "
                f"usuario={profile.get('user_id')}, request_ts={request_ts.isoformat()}, "
                f"event_ts={ev['ts'].isoformat()}, tipo={ev.get('event_type')}"
            )
    return all_recent


def _personal_features(cand_id: str, profile: dict[str, Any] | None,
                       request_ts: datetime) -> list[float]:
    candidate = CATALOG.get(cand_id, {})
    subcat = candidate.get("subcategory")
    brand = candidate.get("brand")
    try:
        price = float(candidate.get("price_eur", 25.0))
    except (TypeError, ValueError):
        price = 25.0
    if profile is None:
        return [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]

    events = _profile_events_before(profile, request_ts)
    total = len(events)
    subcat_n = sum(1 for e in events if subcat and e.get("subcategory") == subcat)
    brand_n = sum(1 for e in events if brand and e.get("brand") == brand)
    subcat_affinity = subcat_n / total if total else 0.0
    brand_affinity = brand_n / total if total else 0.0

    known_prices = [float(x["price"]) for x in profile["purchase_prices"] if x["ts"] < request_ts]
    if known_prices:
        median_price = float(np.median(known_prices))
        price_fit = math.exp(-abs(price - median_price) / max(median_price, 1.0))
        price_percentile = sum(1 for p in known_prices if p <= price) / len(known_prices)
    else:
        price_fit = 0.5
        price_percentile = 0.5

    clicks = sum(1 for e in events if e["event_type"] == "click")
    purchases = sum(1 for e in events if e["event_type"] == "purchase")
    purchase_rate = purchases / clicks if clicks else 0.0
    bought_similar = float(any(
        e["event_type"] == "purchase" and
        ((subcat and e.get("subcategory") == subcat) or (brand and e.get("brand") == brand))
        for e in events
    ))
    recent_cutoff = request_ts - timedelta(days=7)
    recent_events = [e for e in events if e["ts"] >= recent_cutoff]
    recent_subcat = sum(1 for e in recent_events if subcat and e.get("subcategory") == subcat)
    recent_affinity = recent_subcat / len(recent_events) if recent_events else 0.0
    prior_view_ts = profile["viewed_products"].get(cand_id)
    seen_before = float(prior_view_ts is not None and prior_view_ts < request_ts)
    last_event_ts = max((e["ts"] for e in events), default=None)
    days_since = max(0.0, (request_ts - last_event_ts).total_seconds() / 86400.0) if last_event_ts else 0.0
    engagement = (sum({"view": 1, "click": 2, "add_to_cart": 4, "purchase": 8}[e["event_type"]] for e in events))
    return [subcat_affinity, brand_affinity, price_fit, purchase_rate,
            bought_similar, recent_affinity, seen_before, days_since,
            float(engagement), price_percentile]


_build_lgbm7_feature_row = build_feature_row
def build_feature_row(cand_id: str, viewed_ids: list[str], user_id: Any = None,
                      request_ts: datetime | None = None, is_new_user: Any = None) -> list[float]:
    # Las primeras 17 posiciones conservan exactamente el orden de LGBM 7.
    base = _build_lgbm7_feature_row(cand_id, viewed_ids)


    ts = request_ts or datetime.now(timezone.utc)
    first_seen = NEW_PRODUCT_FIRST_SEEN.get(cand_id)
    if first_seen is not None:
        base[15] = max(0.0, (ts - first_seen).total_seconds() / 86400.0)
    profile = USER_PROFILES.get(str(user_id)) if user_id is not None else None
    personal = _personal_features(cand_id, profile, ts)
    explicit_new = _coerce_bool(is_new_user)
    profile_has_history = bool(profile and (profile["n_views"] + profile["n_clicks"] + profile["n_carts"] + profile["n_purchases"]))
    is_cold = explicit_new is True or not profile_has_history
    row = base + personal + [float(is_cold)]
    if len(row) != 28:
        raise RuntimeError(f"LGBM 8 requiere 28 features; se construyeron {len(row)}")
    return row




def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def predict_batch(requests_batch: list[Record]) -> list[Record]:
    active = ACTIVE_IDS if ACTIVE_IDS else set(CATALOG.keys())
    model = LGB8_MODEL
    if model is None:
        # Fallback funcional: sirve el baseline sin alterar sus artefactos.
        model = LGB_MODEL
        if model is None:
            return []
    rows: list[list[float]] = []
    slices: list[tuple[str, list[str], int, int]] = []
    for req in requests_batch:
        update_user_request(req)
        request_ts = _parse_ts(req.get("ts")) or datetime.now(timezone.utc)
        user_id = req.get("user_id")
        viewed = [v["product_id"] for v in (req.get("recent_views") or [])
                  if isinstance(v, dict) and "product_id" in v]
        cands = get_candidates(viewed, active, str(req.get("request_id", "")))
        for anchor_id in TOP_GLOBAL_HIST[:10]:
            if anchor_id not in cands:
                cands.append(anchor_id)
        start = len(rows)
        for cid in cands:
            if model is LGB8_MODEL:
                rows.append(build_feature_row(cid, viewed, user_id, request_ts, req.get("is_new_user")))
            else:
                rows.append(_build_lgbm7_feature_row(cid, viewed))
        slices.append((str(req["request_id"]), cands, start, len(rows)))
    if not rows:
        return []
    scores = model.predict_proba(np.asarray(rows, dtype=np.float32))[:, 1]
    submissions = []
    for rid, cands, start, end in slices:
        order = np.argsort(scores[start:end])[::-1]
        ranked = [cands[i] for i in order]
        cold_user = bool(rows[start][-1]) if model is LGB8_MODEL and end > start else False
        ranked = _hercules_rank(ranked, cands, TOP_GLOBAL_HIST)
        ranked = _reserve_new_product_slots(ranked, cands, cold_user, rid)
        submissions.append({
            "request_id": rid,
            "items": ranked,
            "model_version": f"{HERCULES_MODEL_NAME}-v1" if model is LGB8_MODEL else "v7-lgbm17-fallback",
        })
    return submissions


def main() -> None:
    threading.Thread(target=bg_worker, daemon=True).start()
    threading.Thread(target=_profile_event_worker, daemon=True).start()
    seen: set[str] = set()
    print("Recomendador Hercules activo (LGBM8 + popularidad + personalizaci?n).")
    while True:
        try:
            if len(seen) > 100_000:
                seen.clear()
            response = http_fast.post(f"{API}/v1/query", json={"sql": NEW_REQUESTS_SQL}, timeout=3)
            response.raise_for_status()
            fresh = []
            for req in response.json().get("data", []):
                if req.get("request_id") not in seen:
                    seen.add(req["request_id"])
                    fresh.append(req)
            if fresh:
                answers = predict_batch(fresh)
                if answers:
                    http_fast.post(f"{API}/v1/tables/submissions", params={"database": TEAM},
                                   json=answers, timeout=4).raise_for_status()
                    print(f"answered {len(answers)} (LGBM8 experimental)")
        except Exception as exc:
            print(f"error: {exc}")
            time.sleep(0.5)
        time.sleep(0.25)








LGB8_FEATURE_NAMES = [
    "purchases", "conv_rate", "cart_rate", "price", "price_ratio",
    "same_subcat", "same_cat", "same_brand", "co_score_last", "co_score_total",
    "already_viewed", "slot", "purchases_24h", "views_24h", "carts_24h",
    "product_age_days", "recent_popularity",
    "user_subcategory_affinity", "user_brand_affinity", "price_fit",
    "user_purchase_rate", "user_bought_similar", "recent_subcategory_affinity",
    "candidate_seen_by_user", "days_since_last_interaction", "user_interaction_score",
    "candidate_price_percentile_user", "user_is_cold_start",
]


def recommend(recent_views: list[Record], best: list[str] | None = None,
              unavailable: set[str] = frozenset(), user_id: Any = None,
              request_ts: Any = None, is_new_user: Any = None) -> list[str]:
    viewed = [v["product_id"] for v in (recent_views or [])
              if isinstance(v, dict) and "product_id" in v]
    active = set(CATALOG.keys()) - set(unavailable)
    popularity = best if best is not None else TOP_GLOBAL_HIST
    cands = get_candidates(viewed, active, str(user_id or "offline"))
    for pid in popularity[:100]:
        if pid not in unavailable and pid not in cands:
            cands.append(pid)
    model = LGB8_MODEL if LGB8_MODEL is not None else LGB_MODEL
    if model is None or not cands:
        return [pid for pid in popularity if pid not in unavailable][:10]
    if model is LGB8_MODEL:
        ts = _parse_ts(request_ts) or datetime.now(timezone.utc)
        matrix = [build_feature_row(pid, viewed, user_id, ts, is_new_user) for pid in cands]
    else:
        matrix = [_build_lgbm7_feature_row(pid, viewed) for pid in cands]
    probs = model.predict_proba(np.asarray(matrix, dtype=np.float32))[:, 1]
    model_ranked = [cands[i] for i in np.argsort(probs)[::-1] if cands[i] not in unavailable]
    cold_user = bool(matrix[0][-1]) if model is LGB8_MODEL and matrix else False
    ranked = _hercules_rank(model_ranked, cands, popularity, unavailable)
    ranked = _reserve_new_product_slots(ranked, cands, cold_user, str(user_id or "offline"))
    for pid in popularity:
        if len(ranked) >= 10:
            break
        if pid not in unavailable and pid not in ranked:
            ranked.append(pid)
    return ranked[:10]

if __name__ == "__main__":
    main()
