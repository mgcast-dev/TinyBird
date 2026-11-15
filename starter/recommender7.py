"""Recomendador v7: ranking p(clic)·p(compra|clic) con contexto, tendencias en vivo y productos nuevos.

    python starter/recommender7.py

Diseño: responder SIEMPRE, a tiempo y con 10 productos válidos.
  * 2 hilos de pull (desfasados), 1 hilo que calcula y 4 hilos de push en paralelo
  * hilos de fondo: stock (1 s), catálogo (10 s o al ver un producto desconocido) y tendencia (30 s)
  * finalize() valida: 10 productos distintos, activos y en stock, leyendo el estado al enviar
El notebook importa este módulo sin arrancar nada (solo main() lanza los hilos).
"""
from __future__ import annotations

import itertools
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import requests

from features7 import (FEATURE_NAMES, N_FEATURES, VERSION, build_state, compute_live_lift,
                       epoch_of, make_ctx, rank_request, viewed_from_requests)

# =====================================================================
# 1. CONFIGURACIÓN Y CARGA
# =====================================================================
HERE = Path(__file__).resolve().parent
for _env in (HERE.parent / ".env", HERE / ".env", Path(".env")):
    for _line in _env.read_text().splitlines() if _env.exists() else []:
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ.get("TEAM_ID", "")
HEADERS = {"Authorization": f"Bearer {os.environ.get('RAWTREE_API_KEY', '')}"}

Record = dict[str, Any]
MODEL_VERSION = VERSION
DEADLINE_S = 3.0          # plazo desde el ts de la petición
STALE_WAIT_S = 2.0        # no calculamos una petición que lleva más de esto esperando en NUESTRA cola
RETRY_MARGIN_S = 0.6      # no reintentamos un push si ya no llegaríamos
POLL_S = 0.3              # periodo de cada hilo de pull (hay dos, desfasados)
BATCH = 25
PUSH_THREADS = 4        # varios push en paralelo: con 3 peticiones/s y 0,4 s de ida y vuelta, uno solo se satura

MODELS_DIR = HERE / "models"
SUFFIX = os.environ.get("MODEL_SUFFIX", "")
MODELS = joblib.load(MODELS_DIR / f"v7_model{SUFFIX}.pkl")
FEAT = joblib.load(MODELS_DIR / f"v7_state{SUFFIX}.pkl")
assert FEAT.get("n_features") == N_FEATURES and list(FEAT.get("feature_names", [])) == FEATURE_NAMES, \
    "el estado guardado no coincide con features7.py: vuelve a entrenar"
assert MODELS["click"].n_features_in_ == N_FEATURES, "el modelo no coincide con features7.py: vuelve a entrenar"

STATE = build_state(FEAT["catalog"], FEAT["prod_stats"], FEAT["co_matrix"], extra=FEAT)
CATALOG: dict[str, Record] = STATE["catalog"]
ACTIVE: set[str] = {pid for pid, c in CATALOG.items() if c.get("is_active", True)}
OUT: set[str] = set()                 # productos agotados ahora mismo
LIVE_LIFT: dict[str, float] = {}      # tendencia por subcategoría (vacío = neutra)
FALLBACK: list[str] = list(STATE["top_global"][:200])

READY = threading.Event()             # primera carga del catálogo hecha
NEED_CATALOG = threading.Event()      # alguien vio un producto que no conocemos

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

# vistas por producto: última hora y las 24 h anteriores (la tendencia por subcategoría se agrega aquí)
LIVE_SQL = """
SELECT product_id::String AS product_id,
       countIf(ts::DateTime64(3) > now() - INTERVAL 1 HOUR) AS v1,
       countIf(ts::DateTime64(3) <= now() - INTERVAL 1 HOUR) AS v24
FROM shop.events
WHERE event_type::String = 'view' AND ts::DateTime64(3) > now() - INTERVAL 25 HOUR
GROUP BY product_id
"""

NEW_REQUESTS_SQL = """
SELECT request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 40
"""


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _log(msg: str) -> None:
    print(msg, flush=True)


# =====================================================================
# 3. HILOS DE FONDO: stock, catálogo, tendencia
# =====================================================================
def stock_loop() -> None:
    global OUT
    sess = _session()
    while True:
        t0 = time.time()
        try:
            r = sess.post(f"{API}/v1/query", json={"sql": OUT_OF_STOCK_SQL}, timeout=(1.0, 3.0))
            if r.ok:
                OUT = {row["product_id"] for row in r.json()["data"]}
        except Exception as e:
            _log(f"[stock] {e}")
        time.sleep(max(0.2, 1.0 - (time.time() - t0)))


def refresh_catalog(sess: requests.Session) -> None:
    global ACTIVE
    r = sess.post(f"{API}/v1/query", json={"sql": CATALOG_ACTIVE_SQL}, timeout=(2.0, 10.0))
    r.raise_for_status()
    rows = r.json()["data"]
    if not rows:
        return
    active = set()
    for row in rows:
        pid = str(row["product_id"])
        if row.get("is_active"):
            active.add(pid)
        if pid not in CATALOG:                           # producto nuevo del evento
            CATALOG[pid] = row
            STATE["new_products"].append(pid)
            sub = row.get("subcategory")
            if sub:
                STATE["cold_by_subcat"].setdefault(sub, []).append(pid)
        else:
            try:
                CATALOG[pid]["price_eur"] = float(row.get("price_eur") or CATALOG[pid].get("price_eur") or 25.0)
            except (TypeError, ValueError):
                pass
    ACTIVE = active
    READY.set()


def catalog_loop() -> None:
    sess = _session()
    last = 0.0
    while True:
        now = time.time()
        if now - last >= 10.0 or (NEED_CATALOG.is_set() and now - last >= 2.0):
            NEED_CATALOG.clear()
            last = now
            try:
                refresh_catalog(sess)
            except Exception as e:
                _log(f"[catálogo] {e}")
                if not READY.is_set():
                    sess = _session()
        time.sleep(0.5)


def live_loop() -> None:
    global LIVE_LIFT
    sess = _session()
    while True:
        t0 = time.time()
        try:
            r = sess.post(f"{API}/v1/query", json={"sql": LIVE_SQL}, timeout=(2.0, 20.0))
            r.raise_for_status()
            lift = compute_live_lift(CATALOG, r.json()["data"])
            if lift:
                LIVE_LIFT = lift
        except Exception as e:
            _log(f"[tendencia] {e}")
        time.sleep(max(1.0, 30.0 - (time.time() - t0)))


# =====================================================================
# 4. VALIDACIÓN Y PREDICCIÓN
# =====================================================================
def _all_products():
    yield from list(CATALOG)


def finalize(items: list[str]) -> list[str]:
    """10 productos distintos, activos y en stock, leyendo el estado en el momento del envío."""
    out: list[str] = []
    used: set[str] = set()
    for pid in itertools.chain(items, FALLBACK, _all_products()):
        if pid in ACTIVE and pid not in OUT and pid not in used:
            used.add(pid)
            out.append(pid)
            if len(out) == 10:
                break
    return out


def predict_one(req: Record) -> Record:
    viewed = viewed_from_requests(req.get("recent_views"))
    if any(v not in CATALOG for v in viewed):
        NEED_CATALOG.set()
    try:
        ctx = make_ctx(req.get("context"), req.get("ts"), live_lift=LIVE_LIFT)
        ranked = rank_request(STATE, MODELS, ctx, viewed, ACTIVE, OUT)
    except Exception as e:                           # el modelo nunca debe dejarnos sin respuesta
        _log(f"[predict] {type(e).__name__}: {e}")
        ranked = []
    return {"request_id": req["request_id"], "items": finalize(ranked), "model_version": MODEL_VERSION}


# =====================================================================
# 5. PIPELINE: pull -> predict -> push
# =====================================================================
class Seen:
    """Conjunto acotado de request_id ya encolados."""

    def __init__(self, n: int = 30000):
        self._s: set[str] = set()
        self._d: deque[str] = deque()
        self._n = n
        self._lock = threading.Lock()

    def add_if_new(self, key: str) -> bool:
        with self._lock:
            if key in self._s:
                return False
            self._s.add(key)
            self._d.append(key)
            if len(self._d) > self._n:
                self._s.discard(self._d.popleft())
            return True


SEEN = Seen()
REQ_Q: queue.Queue = queue.Queue()        # (momento_de_llegada, petición)
ANS_Q: queue.Queue = queue.Queue()        # lista de (respuesta, ts_petición)
STATS = {"pulled": 0, "answered": 0, "stale": 0, "push_fail": 0, "pull_fail": 0}
LAT_MS: list[float] = []
STATS_LOCK = threading.Lock()


def poll_loop(offset: float) -> None:
    time.sleep(offset)
    sess = _session()
    fails = 0
    while True:
        t0 = time.time()
        try:
            r = sess.post(f"{API}/v1/query", json={"sql": NEW_REQUESTS_SQL}, timeout=(1.0, 2.5))
            r.raise_for_status()
            now = time.time()
            for q in r.json()["data"]:
                rid = q.get("request_id")
                if rid and SEEN.add_if_new(str(rid)):
                    REQ_Q.put((now, q))
                    with STATS_LOCK:
                        STATS["pulled"] += 1
            fails = 0
        except Exception as e:
            fails += 1
            with STATS_LOCK:
                STATS["pull_fail"] += 1
            if fails % 10 == 1:
                _log(f"[pull] {type(e).__name__}: {e}")
            if fails >= 3:
                sess = _session()
            time.sleep(0.3)
        time.sleep(max(0.0, POLL_S - (time.time() - t0)))


def worker_loop() -> None:
    while True:
        try:
            batch = [REQ_Q.get(timeout=0.5)]
        except queue.Empty:
            continue
        while len(batch) < BATCH:
            try:
                batch.append(REQ_Q.get_nowait())
            except queue.Empty:
                break
        batch.sort(key=lambda x: x[0])
        answers = []
        for arrived, req in batch:
            if time.time() - arrived > STALE_WAIT_S:         # ya no llegaríamos: no gastamos CPU
                with STATS_LOCK:
                    STATS["stale"] += 1
                continue
            try:
                answers.append((predict_one(req), req.get("ts")))
            except Exception as e:
                _log(f"[worker] {type(e).__name__}: {e}")
        if answers:
            ANS_Q.put(answers)


def push_batch(sess: requests.Session, answers: list[Record]) -> None:
    sess.post(f"{API}/v1/tables/submissions", params={"database": TEAM}, json=answers,
              timeout=(1.0, 2.5)).raise_for_status()


def push_loop() -> None:
    sess = _session()
    while True:
        group = list(ANS_Q.get())
        while len(group) < 100:
            try:
                group.extend(ANS_Q.get_nowait())
            except queue.Empty:
                break
        answers = [a for a, _ in group]
        oldest = min(epoch_of(ts) for _, ts in group)
        ok = False
        for attempt in range(2):
            try:
                push_batch(sess, answers)
                ok = True
                break
            except Exception as e:
                _log(f"[push] intento {attempt + 1}: {type(e).__name__}: {e}")
                sess = _session()
                if time.time() - oldest > DEADLINE_S - RETRY_MARGIN_S:
                    break
        done = time.time()
        with STATS_LOCK:
            if not ok:
                STATS["push_fail"] += len(group)
                continue
            STATS["answered"] += len(group)
            LAT_MS.extend((done - epoch_of(ts)) * 1000 for _, ts in group)
            if len(LAT_MS) >= 200:
                lat = sorted(LAT_MS)
                LAT_MS.clear()
                _log(f"latencia p50={lat[len(lat) // 2]:.0f} ms  p95={lat[int(len(lat) * 0.95)]:.0f} ms  "
                     f"máx={lat[-1]:.0f} ms | {dict(STATS)}")
                if lat[len(lat) // 2] < 0 or lat[len(lat) // 2] > 4000:
                    _log("   AVISO: latencias raras; revisa que el reloj del PC esté sincronizado")


# =====================================================================
# 6. MAIN
# =====================================================================
def main() -> None:
    if not TEAM:
        raise SystemExit("Falta TEAM_ID en .env")
    if not os.environ.get("RAWTREE_API_KEY"):
        raise SystemExit("Falta RAWTREE_API_KEY en .env")
    _log(f"Recomendador {MODEL_VERSION} (modelo '{SUFFIX or 'final'}') arrancando para {TEAM}...")
    for fn in (stock_loop, catalog_loop, live_loop):
        threading.Thread(target=fn, daemon=True).start()
    if not READY.wait(timeout=30):
        _log("AVISO: catálogo en vivo no cargado todavía; sigo con el del entrenamiento")
    threads = [(poll_loop, (0.0,)), (poll_loop, (POLL_S / 2,)), (worker_loop, ())]
    threads += [(push_loop, ())] * PUSH_THREADS
    for fn, args in threads:
        threading.Thread(target=fn, args=args, daemon=True).start()
    _log(f"Recomendador {MODEL_VERSION} activo. {len(CATALOG)} productos, {len(ACTIVE)} activos, "
         f"{len(STATE['new_products'])} nuevos.")
    while True:
        time.sleep(30)
        top = sorted(LIVE_LIFT.items(), key=lambda x: -x[1])[:3]
        _log(f"[estado] {dict(STATS)} | en cola={REQ_Q.qsize()} | tendencia: "
             + ", ".join(f"{s}={l:+.2f}" for s, l in top))


if __name__ == "__main__":
    main()
