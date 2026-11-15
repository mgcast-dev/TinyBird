"""Recomendador LightGBM v5: candidatos + ranking, respuestas validadas y envío con reintento.

Diseño: responder SIEMPRE a tiempo con 10 productos válidos. El modelo no decide
nada de eso; lo decide la capa de validación y envío.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import requests

from features import (N_FEATURES, build_feature_row, build_state,
                      get_candidates, viewed_from_requests)

# =====================================================================
# 1. CONFIGURACIÓN Y CARGA
# =====================================================================
ROOT = Path(__file__).resolve().parents[1]
for line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ.get("TEAM_ID", "")
HEADERS = {"Authorization": f"Bearer {os.environ.get('RAWTREE_API_KEY', '')}"}

http_fast = requests.Session()
http_fast.headers.update(HEADERS)
http_bg = requests.Session()
http_bg.headers.update(HEADERS)

Record = dict[str, Any]
MODEL_VERSION = "v6.2-lgbm-clean"
DEADLINE_S = 3.0      # plazo desde el ts de la petición
SAFETY_S = 0.5        # no intentamos responder si ya no llegamos
BATCH = 20            # tamaño de lote para el push
MODELS_DIR = Path(__file__).resolve().parent / "models"
SUFFIX = os.environ.get("MODEL_SUFFIX", "")
LGB_MODEL = joblib.load(MODELS_DIR / f"modelo_lgbm{SUFFIX}.pkl")
FEAT = joblib.load(MODELS_DIR / f"features_data{SUFFIX}.pkl")
assert FEAT.get("n_features") == N_FEATURES, "features_data no coincide con features.py"
assert LGB_MODEL.n_features_in_ == N_FEATURES, "el modelo no coincide con features.py"

STATE = build_state(FEAT["catalog"], FEAT["prod_stats"], FEAT["co_matrix"])
CATALOG: dict[str, Record] = STATE["catalog"]
ACTIVE: set[str] = {pid for pid, c in CATALOG.items() if c.get("is_active", True)}
OUT: set[str] = set()  # productos agotados ahora mismo

# =====================================================================
# 2. SQL
# =====================================================================
CATALOG_ACTIVE_SQL = """
SELECT product_id,
       argMax(category, v)    AS category,
       argMax(subcategory, v) AS subcategory,
       argMax(brand, v)       AS brand,
       argMax(price_eur, v)   AS price_eur,
       argMax(is_active, v)   AS is_active
FROM (
    SELECT product_id::String AS product_id,
           category::String AS category, subcategory::String AS subcategory,
           brand::String AS brand, price_eur::Float64 AS price_eur,
           is_active::Bool AS is_active, '' AS v
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String, category::String, subcategory::String,
           brand::String, price_eur::Float64, is_active::Bool, toString(ts)
    FROM shop.catalog_updates
)
GROUP BY product_id
"""

OUT_OF_STOCK_SQL = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'
"""

NEW_REQUESTS_SQL = """
SELECT request_id, ts, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 100
"""

# =====================================================================
# 3. HILO DE FONDO: stock cada 1 s, catálogo cada 10 s
# =====================================================================
def bg_worker() -> None:
    global ACTIVE, OUT
    last_catalog = 0.0
    while True:
        try:
            r = http_bg.post(f"{API}/v1/query", json={"sql": OUT_OF_STOCK_SQL}, timeout=3)
            if r.ok:
                OUT = {row["product_id"] for row in r.json()["data"]}

            if time.time() - last_catalog >= 10.0:
                last_catalog = time.time()
                r = http_bg.post(f"{API}/v1/query", json={"sql": CATALOG_ACTIVE_SQL}, timeout=8)
                if r.ok:
                    rows = r.json()["data"]
                    ACTIVE = {row["product_id"] for row in rows if row.get("is_active")}
                    for row in rows:
                        pid = row["product_id"]
                        if pid not in CATALOG:          # producto nuevo del evento
                            CATALOG[pid] = row
                        else:
                            CATALOG[pid]["price_eur"] = float(
                                row.get("price_eur", CATALOG[pid].get("price_eur", 25.0)))
        except Exception as e:
            print(f"[BG aviso]: {e}")
        time.sleep(1.0)

# =====================================================================
# 4. VALIDACIÓN Y MODELO
# =====================================================================
FALLBACK = STATE["top_global"][:200]


def finalize(items: list[str]) -> list[str]:
    """10 productos distintos, activos y en stock, leyendo el estado en el momento del envío."""
    out: list[str] = []
    used: set[str] = set()
    for pid in list(items) + FALLBACK:
        if pid in ACTIVE and pid not in OUT and pid not in used:
            used.add(pid)
            out.append(pid)
            if len(out) == 10:
                break
    return out


def predict_batch(batch) -> list[Record]:
    out: list[Record] = []
    for req in batch:
        rid = req["request_id"]
        viewed = viewed_from_requests(req.get("recent_views"))
        cands = get_candidates(STATE, viewed, ACTIVE) if viewed else []
        if not cands:
            items = finalize(FALLBACK)
        else:
            X = np.nan_to_num(np.array(
                [build_feature_row(STATE, c, viewed) for c in cands], dtype=np.float32))
            p = LGB_MODEL.predict_proba(X)[:, 1]
            items = finalize([cands[i] for i in np.argsort(p)[::-1]])
        out.append({"request_id": rid, "items": items, "model_version": MODEL_VERSION})
    return out

# =====================================================================
# 5. ENVÍO Y TIEMPOS
# =====================================================================
def age_s(ts: Any) -> float:
    """Segundos transcurridos desde el ts de la petición (ISO 8601, UTC)."""
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return time.time() - dt.timestamp()
    except ValueError:
        return 0.0


def push_with_retry(answers: list[Record]) -> bool:
    for attempt in range(2):
        try:
            http_fast.post(
                f"{API}/v1/tables/submissions",
                params={"database": TEAM},
                json=answers,
                timeout=2,
            ).raise_for_status()
            return True
        except Exception as e:
            print(f"push intento {attempt + 1} fallido: {e}")
    return False

# =====================================================================
# 6. BUCLE PRINCIPAL
# =====================================================================
def main() -> None:
    threading.Thread(target=bg_worker, daemon=True).start()
    seen: set[str] = set()
    lat_ms: list[float] = []
    print(f"Recomendador {MODEL_VERSION} activo.")

    if not TEAM:
        raise SystemExit("Falta TEAM_ID en .env")

    while True:
        try:
            if len(seen) > 100_000:
                seen.clear()

            r = http_fast.post(f"{API}/v1/query", json={"sql": NEW_REQUESTS_SQL}, timeout=1.5)
            r.raise_for_status()
            fresh = [q for q in r.json()["data"] if q["request_id"] not in seen]

            # Las más antiguas tienen el plazo más cercano: se responden primero
            fresh.sort(key=lambda q: str(q.get("ts")))

            # Si ya no llegamos al plazo, no gastamos CPU en ellas
            for q in fresh:
                if age_s(q.get("ts")) > DEADLINE_S - SAFETY_S:
                    seen.add(q["request_id"])
            live = [q for q in fresh if q["request_id"] not in seen]

            for i in range(0, len(live), BATCH):
                chunk = live[i:i + BATCH]
                answers = predict_batch(chunk)
                if push_with_retry(answers):
                    seen.update(q["request_id"] for q in chunk)
                    for q in chunk:
                        lat_ms.append(age_s(q.get("ts")) * 1000)
                    if len(lat_ms) >= 200:
                        lat_ms.sort()
                        print(f"latencia p50={lat_ms[len(lat_ms)//2]:.0f} ms  "
                              f"p95={lat_ms[int(len(lat_ms)*0.95)]:.0f} ms")
                        lat_ms.clear()
        except Exception as e:
            print(f"error: {e}")
            time.sleep(0.5)

        time.sleep(0.1)


if __name__ == "__main__":
    main()