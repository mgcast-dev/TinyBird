"""Recomendador v2: Popularidad en Vivo + Tendencias (15m vs 3h) + Afinidad de Sesión.

Este script sustituye al baseline original mantendo la misma arquitectura rápida:
1. Al arrancar: Calcula las ventas históricas una sola vez (no cambian).
2. Cada 30 segundos: Ejecuta SQL en RawTree para actualizar el catálogo activo (con stock)
   y calcular para cada producto su Popularidad (última hora) y Tendencia (últimos 15 min).
3. Cada 1 segundo: Revisa en `shop.catalog_updates` si algún producto se acaba de agotar.
4. Cada 0.3 segundos: Lee los clientes nuevos de `shop.requests`, calcula en memoria
   su afinidad con las categorías/subcategorías/marcas que acaban de ver en `recent_views`,
   combina las 4 señales normalizadas (0 a 1) y envía el Top 10 en menos de 1 segundo.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

# =====================================================================
# CONFIGURACIÓN Y CONEXIÓN A RAWTREE
# =====================================================================
for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"

Record = dict[str, Any]


# =====================================================================
# CONSULTAS SQL VERIFICADAS PARA RAWTREE
# =====================================================================

# 1. Ventas históricas (Se ejecuta SOLO UNA VEZ al arrancar el programa)
HIST_SALES_SQL = """
SELECT product_id::String AS product_id,
       count() AS n
FROM shop.history_events
WHERE event_type = 'purchase'
GROUP BY product_id
"""

# 2. Catálogo actual combinando shop.catalog + shop.catalog_updates (Exactamente Query 1 de queries.sql)
CATALOG_SQL = """
SELECT product_id,
       argMax(category, v)    AS category,
       argMax(subcategory, v) AS subcategory,
       argMax(brand, v)       AS brand,
       argMax(price_eur, v)   AS price_eur,
       argMax(is_active, v)   AS is_active
FROM (
    SELECT product_id::String AS product_id,
           category::String AS category,
           subcategory::String AS subcategory,
           brand::String AS brand,
           price_eur::Float64 AS price_eur,
           is_active::Bool AS is_active,
           '' AS v
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String,
           category::String,
           subcategory::String,
           brand::String,
           price_eur::Float64,
           is_active::Bool,
           toString(ts)
    FROM shop.catalog_updates
)
GROUP BY product_id
"""

# 3. Señales en vivo de los últimos 195 minutos (3 horas y 15 minutos) en shop.events
#    - popularity_score: 1 pto por vista, 5 por carrito, 10 por compra en la última hora
#    - views_15m: vistas en los últimos 15 minutos
#    - views_prev_3h: vistas en las 3 horas anteriores (para detectar subidas repentinas)
LIVE_SIGNALS_SQL = """
SELECT product_id::String AS product_id,
       countIf(ts > now() - INTERVAL 1 HOUR AND event_type = 'view')
       + 5 * countIf(ts > now() - INTERVAL 1 HOUR AND event_type = 'add_to_cart')
       + 10 * countIf(ts > now() - INTERVAL 1 HOUR AND event_type = 'purchase') AS popularity_score,
       countIf(ts > now() - INTERVAL 15 MINUTE AND event_type = 'view')          AS views_15m,
       countIf(ts <= now() - INTERVAL 15 MINUTE AND event_type = 'view')         AS views_prev_3h
FROM shop.events
WHERE ts > now() - INTERVAL 195 MINUTE
GROUP BY product_id
"""

# 4. Peticiones de recomendación de clientes en tiempo real
NEW_REQUESTS_SQL = """
SELECT request_id, user_id, is_new_user, recent_views, context
FROM shop.requests
ORDER BY ts DESC
LIMIT 50
"""

# 5. Control rápido de productos agotados (Idéntica al recommender.py original)
OUT_OF_STOCK_SQL = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'
"""


# =====================================================================
# FUNCIONES DE CARGA Y ACTUALIZACIÓN DE DATOS
# =====================================================================

def query(sql: str) -> list[Record]:
    """Envía una consulta SQL a RawTree y devuelve las filas como lista de diccionarios."""
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=10)
    r.raise_for_status()
    return r.json()["data"]


def load_hist_sales() -> dict[str, int]:
    """Carga el número de compras históricas por producto una sola vez."""
    return {r["product_id"]: int(r["n"]) for r in query(HIST_SALES_SQL)}


def update_product_signals(hist_sales: dict[str, int]) -> dict[str, Record]:
    """Cruza en Python el catálogo activo con las señales en vivo y las normaliza entre 0 y 1."""
    # 1. Leemos catálogo actualizado y nos quedamos solo con los productos activos (is_active == True)
    catalog_rows = query(CATALOG_SQL)
    active_catalog = {r["product_id"]: r for r in catalog_rows if r.get("is_active")}

    # 2. Leemos señales en vivo de shop.events (si la tabla está vacía o falla, usamos dict vacío)
    try:
        live_rows = {r["product_id"]: r for r in query(LIVE_SIGNALS_SQL)}
    except Exception as e:
        print(f"Aviso en live_signals (usando histórico): {e}")
        live_rows = {}

    # 3. Calculamos popularity_score y trending_score brutos para cada producto activo
    products: dict[str, Record] = {}
    max_pop = 1.0
    max_trend = 1.0

    for pid, cat_info in active_catalog.items():
        live = live_rows.get(pid, {})
        live_pop = float(live.get("popularity_score", 0.0))
        views_15m = float(live.get("views_15m", 0.0))
        views_prev_3h = float(live.get("views_prev_3h", 0.0))

        # Sumamos una pequeña fracción de las ventas históricas (x0.02) para desempatar productos sin eventos aún
        pop_score = live_pop + 0.02 * float(hist_sales.get(pid, 0))

        # Fórmula de Trending (Bloque 4 de vuestra query): ratio últimos 15m vs media de 15m en las 3h previas
        avg_prev_15m = max(views_prev_3h / 12.0, 1.0)
        trend_score = min(views_15m / avg_prev_15m, 20.0)

        if pop_score > max_pop:
            max_pop = pop_score
        if trend_score > max_trend:
            max_trend = trend_score

        products[pid] = {
            "product_id": pid,
            "category": str(cat_info.get("category", "")),
            "subcategory": str(cat_info.get("subcategory", "")),
            "brand": str(cat_info.get("brand", "")),
            "price_eur": float(cat_info.get("price_eur", 0.0)),
            "pop_raw": pop_score,
            "trend_raw": trend_score,
        }

    # 4. Normalización (Bloque 8 de vuestra query): dividimos entre el máximo para tener escala [0.0, 1.0]
    for p in products.values():
        p["popularity_norm"] = p["pop_raw"] / max_pop
        p["trending_norm"] = p["trend_raw"] / max_trend

    return products


def out_of_stock() -> set[str]:
    """Devuelve el conjunto de IDs de productos agotados en este segundo."""
    return {r["product_id"] for r in query(OUT_OF_STOCK_SQL)}


def pull(seen: set[str]) -> list[Record]:
    """Descarga las peticiones recientes y filtra las que aún no hemos respondido."""
    fresh = []
    for req in query(NEW_REQUESTS_SQL):
        if req["request_id"] not in seen:
            seen.add(req["request_id"])
            fresh.append(req)
    return fresh


# =====================================================================
# LÓGICA DEL MODELO DE RECOMENDACIÓN (CÁLCULO EN MEMORIA < 1 ms)
# =====================================================================

def recommend(
    recent_views: list[Record],
    products: dict[str, Record],
    top_base: list[str],
    unavailable: set[str] = frozenset(),
) -> list[str]:
    """Elige los 10 mejores productos para un cliente aplicando vuestra fórmula de 4 señales:
       Score = 0.35 * Popularidad + 0.25 * Tendencia + 0.25 * Subcategoría + 0.15 * (Categoría + Marca)
    """
    # Extraemos los IDs de los productos que el usuario acaba de ver en esta sesión
    viewed_ids = [
        v["product_id"]
        for v in (recent_views or [])
        if isinstance(v, dict) and "product_id" in v
    ]

    # CASO 1: El usuario acaba de entrar y no ha visto ningún producto todavía (`recent_views` vacío).
    # Le devolvemos directamente el Top 10 que combina Popularidad (0.35) + Tendencia (0.25) en vivo.
    if not viewed_ids:
        return [pid for pid in top_base if pid not in unavailable][:10]

    # CASO 2: El usuario SÍ ha visto productos en esta sesión.
    # Calculamos su afinidad por subcategoría, categoría y marca (Bloques 5 y 6 de vuestra query).
    subcat_scores: dict[str, float] = {}
    cat_scores: dict[str, float] = {}
    brand_scores: dict[str, float] = {}
    viewed_set = set(viewed_ids)

    # Recorremos los productos vistos: cuanto más reciente es la vista (mayor índice `idx`), más peso le damos
    for idx, pid in enumerate(viewed_ids):
        p_info = products.get(pid)
        if not p_info:
            continue
        weight = 1.0 + 0.5 * idx  # El último producto visto pesa más que el primero
        subcat = p_info["subcategory"]
        cat = p_info["category"]
        brand = p_info["brand"]

        subcat_scores[subcat] = subcat_scores.get(subcat, 0.0) + weight
        cat_scores[cat] = cat_scores.get(cat, 0.0) + weight
        brand_scores[brand] = brand_scores.get(brand, 0.0) + weight

    # Si ninguno de los productos vistos estaba en nuestro catálogo, usamos el ranking base
    if not subcat_scores:
        return [pid for pid in top_base if pid not in unavailable][:10]

    # Valores máximos para normalizar la afinidad del usuario entre 0.0 y 1.0 (Bloque 8)
    max_subcat = max(subcat_scores.values(), default=1.0)
    max_cat = max(cat_scores.values(), default=1.0)
    max_brand = max(brand_scores.values(), default=1.0)

    # Puntuamos todos los productos activos del catálogo (Bloque 9: Score Final)
    scored_candidates: list[tuple[float, str]] = []
    for pid, p in products.items():
        if pid in unavailable:
            continue

        # Señales globales en vivo (ya normalizadas entre 0 y 1)
        popularity_norm = p["popularity_norm"]
        trending_norm = p["trending_norm"]

        # Señales personalizadas del usuario (normalizadas entre 0 y 1)
        user_subcat_norm = subcat_scores.get(p["subcategory"], 0.0) / max_subcat
        cat_norm = cat_scores.get(p["category"], 0.0) / max_cat
        brand_norm = brand_scores.get(p["brand"], 0.0) / max_brand
        affinity_norm = 0.6 * cat_norm + 0.4 * brand_norm

        # Vuestra fórmula de pesos exacta: 35% pop + 25% trend + 25% subcategoría + 15% (categoría/marca)
        final_score = (
            0.35 * popularity_norm
            + 0.25 * trending_norm
            + 0.25 * user_subcat_norm
            + 0.15 * affinity_norm
        )

        # Evitamos recomendar exactamente el mismo producto que ya está mirando en la página actual
        if pid in viewed_set:
            final_score -= 0.05

        scored_candidates.append((final_score, pid))

    # Ordenamos de mayor a menor puntuación (Bloque 10: ORDER BY final_score DESC LIMIT 10)
    scored_candidates.sort(reverse=True)
    return [pid for _, pid in scored_candidates[:10]]


def predict(
    requests_batch: list[Record],
    products: dict[str, Record],
    top_base: list[str],
    unavailable: set[str],
) -> list[Record]:
    """Genera la lista de envíos para insertar en team_NN.submissions."""
    return [
        {
            "request_id": req["request_id"],
            "items": recommend(req.get("recent_views", []), products, top_base, unavailable),
            "model_version": "v2-sql-signals",
        }
        for req in requests_batch
    ]


def push(answers: list[Record]) -> None:
    """Envía el lote de recomendaciones a la tabla team_NN.submissions."""
    http.post(
        f"{API}/v1/tables/submissions",
        params={"database": TEAM},
        json=answers,
        timeout=10,
    ).raise_for_status()
    print(f"answered {len(answers)} (v2-sql-signals)")


# =====================================================================
# BUCLE PRINCIPAL EN TIEMPO REAL
# =====================================================================

def main() -> None:
    hist_sales: dict[str, int] = {}
    products: dict[str, Record] = {}
    top_base: list[str] = []
    seen = set[str]()
    unavailable: set[str] = set()

    signals_checked = 0.0
    stock_checked = 0.0

    while True:
        try:
            # 1. Al arrancar, cargamos las compras históricas una sola vez
            if not hist_sales:
                hist_sales = load_hist_sales()
                print(f"Histórico cargado: {len(hist_sales)} productos con ventas.")

            now = time.time()

            # 2. Cada 30 segundos, refrescamos el catálogo activo y las señales en vivo (popularidad y tendencia)
            if not products or (now - signals_checked >= 30.0):
                signals_checked = now
                try:
                    products = update_product_signals(hist_sales)
                    # Ranking precalculado para usuarios nuevos que entran sin `recent_views`
                    top_base = [
                        pid
                        for _, pid in sorted(
                            (
                                (0.35 * p["popularity_norm"] + 0.25 * p["trending_norm"], pid)
                                for pid, p in products.items()
                            ),
                            reverse=True,
                        )
                    ]
                    print(f"Señales actualizadas: {len(products)} productos activos en memoria.")
                except Exception as e:
                    print(f"Error actualizando señales (se mantiene versión anterior): {e}")

            # 3. Cada 1 segundo, comprobamos si algún producto se ha quedado sin stock
            if now - stock_checked >= 1.0:
                stock_checked = now
                try:
                    unavailable = out_of_stock()
                except Exception as e:
                    print(f"Error comprobando stock: {e}")

            # 4. Cada 0.3 segundos, leemos clientes nuevos y respondemos en lote
            if products:
                answers = predict(pull(seen), products, top_base, unavailable)
                if answers:
                    push(answers)

        except Exception as e:
            print(f"Error en bucle principal: {e}")
            time.sleep(2)

        time.sleep(0.3)


if __name__ == "__main__":
    main()