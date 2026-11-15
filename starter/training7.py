"""Entrenamiento v7: dos modelos LightGBM (clic y compra | clic) con validación temporal.

    python starter/training7.py                    # modelo final (todos los datos hasta ahora)
    python starter/training7.py --cut 2026-11-07   # modelo para evaluar offline (solo datos < corte)

Con --cut los ficheros llevan sufijo _cut: models/v7_model_cut.pkl y models/v7_state_cut.pkl.
No sobrescribe nada de la versión anterior (modelo_lgbm*.pkl, features_data*.pkl).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import requests
from sklearn.metrics import roc_auc_score

from features7 import (CAT_IDX, FEATURE_NAMES, N_FEATURES, SLOT_IDX, VERSION, build_examples,
                       build_state, epoch_of, hourly_from_rows)

parser = argparse.ArgumentParser()
parser.add_argument("--cut", default="", help="YYYY-MM-DD: usar solo datos anteriores")
parser.add_argument("--sessions", type=int, default=60000, help="sesiones de entrenamiento")
parser.add_argument("--days", type=int, default=30, help="ventana de sesiones (días antes del corte)")
parser.add_argument("--drop", type=float, default=0.15, help="fracción de filas con producto 'frío' simulado")
args = parser.parse_args()

CUT = args.cut
SUFFIX = "_cut" if CUT else ""
NOW = f"toDateTime('{CUT}')" if CUT else "now()"

HERE = Path(__file__).resolve().parent
for env in (HERE.parent / ".env", HERE / ".env", Path(".env")):
    for line in env.read_text().splitlines() if env.exists() else []:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
API_KEY = os.environ.get("RAWTREE_API_KEY", "")
if not API_KEY:
    raise SystemExit(f"Falta RAWTREE_API_KEY (busqué .env en {HERE.parent})")
http = requests.Session()
http.headers["Authorization"] = f"Bearer {API_KEY}"


def query(sql: str, timeout: int = 120) -> list[dict]:
    last = None
    for attempt in range(2):
        try:
            r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
            r.raise_for_status()
            return r.json()["data"]
        except Exception as e:                      # un reintento por si es un fallo de red puntual
            last = e
            print(f"   (intento {attempt + 1} fallido: {e})", flush=True)
    raise last


T0 = time.time()
print(f"Corte temporal: {CUT or 'ninguno (modelo final)'}  |  {VERSION}", flush=True)

# 1. Catálogo y estadísticas por producto (solo datos anteriores al corte)
print("1/5 Catálogo y estadísticas por producto...", flush=True)
cat_rows = query("""
SELECT product_id::String AS product_id, category::String AS category,
       subcategory::String AS subcategory, brand::String AS brand, price_eur::Float64 AS price_eur
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
    prod_stats[r["product_id"]] = {"views": v, "carts": c, "purchases": p,
                                   "conv_rate": (p + 1.0) / (v + 50.0),
                                   "cart_rate": (c + 1.0) / (v + 20.0)}

# 2. Co-visitas y co-compras de los 20 días anteriores al corte
print("2/5 Co-visitas y co-compras...", flush=True)
co_rows = query(f"""
SELECT a.product_id AS visto, b.product_id AS recomendado,
       countIf(b.event_type = 'purchase') * 15
       + countIf(b.event_type = 'add_to_cart') * 4
       + countIf(b.event_type = 'view') AS score
FROM (
    SELECT session_id::String AS session_id, product_id::String AS product_id
    FROM shop.history_events
    WHERE event_type = 'view' AND ts::DateTime64(3) < {NOW} AND ts::DateTime64(3) > {NOW} - INTERVAL 20 DAY
) AS a
INNER JOIN (
    SELECT session_id::String AS session_id, product_id::String AS product_id, event_type::String AS event_type
    FROM shop.history_events
    WHERE ts::DateTime64(3) < {NOW} AND ts::DateTime64(3) > {NOW} - INTERVAL 20 DAY
) AS b ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY visto, recomendado
ORDER BY score DESC
LIMIT 25 BY visto
""", timeout=240)
co_matrix: dict[str, dict[str, float]] = defaultdict(dict)
for r in co_rows:
    co_matrix[r["visto"]][r["recomendado"]] = float(r["score"])
co_matrix = dict(co_matrix)

# 3. Vistas por subcategoría y hora (para la tendencia punto-en-el-tiempo)
print("3/5 Serie horaria de vistas por subcategoría...", flush=True)
try:
    hv_rows = query(f"""
    SELECT c.subcategory::String AS sub,
           toUnixTimestamp(toStartOfHour(e.ts::DateTime64(3))) AS h,
           count() AS v
    FROM shop.history_events AS e
    INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
    WHERE e.event_type = 'view' AND e.ts::DateTime64(3) < {NOW}
      AND e.ts::DateTime64(3) >= {NOW} - INTERVAL {args.days + 3} DAY
    GROUP BY sub, h
    """, timeout=240)
    hv = hourly_from_rows(hv_rows)
    print(f"   {len(hv_rows)} celdas (subcategoría, hora)", flush=True)
except Exception as e:                              # sin serie horaria: sub_lift queda en 0
    print(f"   AVISO: sin tendencia ({e}). sub_lift será 0.", flush=True)
    hv = {}

# 4. Sesiones de entrenamiento: las más recientes antes del corte, muestreadas al azar
print("4/5 Sesiones de shop.history_sessions...", flush=True)
sessions = query(f"""
SELECT ts, context, recent_views, shown, propensity, clicked, purchased
FROM shop.history_sessions
WHERE ts::DateTime64(3) < {NOW} AND ts::DateTime64(3) >= {NOW} - INTERVAL {args.days} DAY
  AND ((clicked IS NOT NULL AND clicked != '') OR rand() % 8 = 0)
ORDER BY rand()
LIMIT {args.sessions}
""", timeout=300)
sessions.sort(key=lambda s: epoch_of(s.get("ts"), 0.0))        # orden temporal para validar
regions = sorted({(s.get("context") or {}).get("region") for s in sessions
                  if isinstance(s.get("context"), dict) and (s.get("context") or {}).get("region")})
region_index = {r: float(i) for i, r in enumerate(regions)}
print(f"   {len(sessions)} sesiones, {len(regions)} regiones", flush=True)

# 5. Ejemplos y modelos
print("5/5 Features y entrenamiento...", flush=True)
state = build_state(catalog, prod_stats, co_matrix,
                    extra={"hv": hv, "region_index": region_index})
data = build_examples(state, sessions, np.random.default_rng(42), drop_p=args.drop)
X, y_click, y_buy, w = data["X"], data["y_click"], data["y_buy"], data["w"]
is_click_row, sess = data["is_click_row"], data["sess"]
assert X.shape[1] == N_FEATURES
print(f"   filas={len(y_click)}  clics={int(y_click.sum())}  compras={int(y_buy.sum())}  "
      f"CTR={y_click.mean():.4f}", flush=True)

split = int(len(sessions) * 0.85)                   # el 15 % más reciente valida
tr, va = sess < split, sess >= split


def fit(Xa, ya, wa, n_estimators=400, leaves=31, min_child=40, evals=None):
    clf = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=leaves,
                             min_child_samples=min_child, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1)
    kw = {"categorical_feature": CAT_IDX}
    if evals is not None:
        Xv, yv, wv = evals
        kw.update(eval_set=[(Xv, yv)], eval_sample_weight=[wv],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    clf.fit(Xa, ya, sample_weight=wa, **kw)
    return clf


def best_iter(clf, default):
    return int(getattr(clf, "best_iteration_", None) or default)


def safe_auc(y, s, wt=None):
    return roc_auc_score(y, s, sample_weight=wt) if 0 < y.sum() < len(y) else float("nan")


# --- modelo de clic
m_click = fit(X[tr], y_click[tr], w[tr], evals=(X[va], y_click[va], w[va]))
n_click = best_iter(m_click, 400)
print(f"   clic:   árboles={n_click}  AUC(val)={safe_auc(y_click[va], m_click.predict_proba(X[va])[:, 1], w[va]):.4f}",
      flush=True)

# --- modelo de compra | clic (solo filas con clic)
trb, vab = tr & is_click_row, va & is_click_row
m_buy = None
n_buy = 0
if y_buy[trb].sum() >= 200 and y_buy[vab].sum() >= 20:
    m_buy = fit(X[trb], y_buy[trb], w[trb], n_estimators=300, leaves=15, min_child=30,
                evals=(X[vab], y_buy[vab], w[vab]))
    n_buy = best_iter(m_buy, 300)
    print(f"   compra: árboles={n_buy}  AUC(val, solo clics)="
          f"{safe_auc(y_buy[vab], m_buy.predict_proba(X[vab])[:, 1], w[vab]):.4f}", flush=True)
else:
    print("   compra: pocos positivos, se usa solo el modelo de clic", flush=True)

# --- exponente de compra: lo que se sirve es p(clic) * p(compra|clic)^g, a slot 0
buy_exp = 0.0
if m_buy is not None:
    Xv0 = X[va].copy()
    Xv0[:, SLOT_IDX] = 0.0
    pc, pb = m_click.predict_proba(Xv0)[:, 1], m_buy.predict_proba(Xv0)[:, 1]
    best_auc = -1.0
    for g in (0.0, 0.5, 1.0):
        auc = safe_auc(y_buy[va], pc * np.power(pb, g), w[va])
        print(f"   AUC compra (val) con g={g}: {auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, buy_exp = auc, g
    print(f"   -> exponente de compra elegido: {buy_exp}", flush=True)
    if buy_exp == 0.0:
        m_buy = None

# --- reentrenar con todos los datos (más recientes incluidos) y guardar
print("   reentrenando con todos los datos...", flush=True)
final_click = fit(X, y_click, w, n_estimators=int(n_click * 1.15) + 1)
final_buy = None
if m_buy is not None:
    allb = is_click_row
    final_buy = fit(X[allb], y_buy[allb], w[allb], n_estimators=int(n_buy * 1.15) + 1, leaves=15, min_child=30)

try:
    gain = final_click.booster_.feature_importance(importance_type="gain")
    top = sorted(zip(FEATURE_NAMES, gain), key=lambda x: -x[1])[:12]
    print("   features más importantes (clic):", ", ".join(f"{n}={g / sum(gain):.0%}" for n, g in top), flush=True)
except Exception:
    pass

MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump({"click": final_click, "buy": final_buy, "buy_exp": buy_exp, "version": VERSION},
            MODELS_DIR / f"v7_model{SUFFIX}.pkl")
joblib.dump({"catalog": catalog, "prod_stats": prod_stats, "co_matrix": co_matrix, "hv": hv,
             "region_index": region_index, "cat_index": state["cat_index"],
             "n_features": N_FEATURES, "feature_names": FEATURE_NAMES, "version": VERSION,
             "cut": CUT, "trained_at": time.time()},
            MODELS_DIR / f"v7_state{SUFFIX}.pkl")
print(f"¡Listo en {time.time() - T0:.0f} s! Guardado en {MODELS_DIR} (sufijo '{SUFFIX}')", flush=True)
