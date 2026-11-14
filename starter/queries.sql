-- SQL examples for the RawTree challenge. Run them with the CLI, the HTTP API or the MCP server:
--   rtree query --api-key $RAWTREE_API_KEY --database $TEAM_ID "<sql>"
--   curl -s https://api.rawtree.com/v1/query -H "Authorization: Bearer $RAWTREE_API_KEY" -d '{"sql":"<sql>"}'
-- Replace team_NN with your team database.
--
-- RawTree is schemaless: every column is Dynamic (each value keeps its own type). Filters, ORDER BY and most
-- functions just work; JOIN keys, argMax/argMin and some aggregates need a cast: product_id::String, ts::DateTime64(3).


-- 1. Current catalogue: shop.catalog with the latest shop.catalog_updates record of each product on top
SELECT product_id,
       argMax(category, v)    AS category,
       argMax(subcategory, v) AS subcategory,
       argMax(price_eur, v)   AS price_eur,
       argMax(is_active, v)   AS is_active
FROM (
    SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
           price_eur::Float64 AS price_eur, is_active::Bool AS is_active, '' AS v
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String, category::String, subcategory::String, price_eur::Float64, is_active::Bool, toString(ts)
    FROM shop.catalog_updates
)
GROUP BY product_id;


-- 2. Popularity per category in the last hour (live organic activity)
SELECT c.category::String AS category, e.product_id::String AS product_id,
       countIf(e.event_type = 'view')     AS views,
       countIf(e.event_type = 'purchase') AS purchases
FROM shop.events AS e
INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
WHERE e.ts > now() - INTERVAL 1 HOUR
GROUP BY category, product_id
ORDER BY purchases DESC, views DESC
LIMIT 5 BY category;


-- 3. What is trending right now? Views per subcategory, last 15 minutes vs the 3 hours before
SELECT c.subcategory::String AS subcategory,
       countIf(e.ts > now() - INTERVAL 15 MINUTE)        AS views_15m,
       countIf(e.ts <= now() - INTERVAL 15 MINUTE) / 12  AS avg_views_per_15m_before,
       views_15m / greatest(avg_views_per_15m_before, 1) AS lift
FROM shop.events AS e
INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
WHERE e.event_type = 'view' AND e.ts > now() - INTERVAL 195 MINUTE
GROUP BY subcategory
ORDER BY lift DESC
LIMIT 10;


-- 4. Co-views: products viewed in the same session as a given product (history)
SELECT b.product_id AS also_viewed, count() AS sessions
FROM (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events WHERE event_type = 'view') AS a
INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events WHERE event_type = 'view') AS b
    ON a.session_id = b.session_id
WHERE a.product_id = 'p_xxxxxx' AND b.product_id != a.product_id
GROUP BY also_viewed
ORDER BY sessions DESC
LIMIT 20;


-- 5. Your feedback: conversion and revenue per hour, split by who served the shopper
-- (clicked / purchased are '' when nothing happened)
SELECT toStartOfHour(ts::DateTime64(3)) AS hour,
       served_by::String                   AS served_by,
       count()                             AS requests,
       100 * countIf(clicked != '') / count()   AS ctr_pct,
       100 * countIf(purchased != '') / count() AS conversion_pct,
       sum(revenue_eur::Float64) / count()      AS revenue_per_request
FROM team_31.feedback
WHERE request_id != '__init__'
GROUP BY hour, served_by
ORDER BY hour DESC, served_by;


-- 6. Reliability: why did the fallback serve your shoppers, and how fast are you?
SELECT fallback_reason::String              AS fallback_reason,
       count()                              AS requests,
       quantile(0.5)(latency_ms::Float64)   AS p50_ms,
       quantile(0.95)(latency_ms::Float64)  AS p95_ms
FROM team_31.feedback
WHERE ts > now() - INTERVAL 30 MINUTE AND request_id != '__init__'
GROUP BY fallback_reason;


-- 7. Which slots get clicked? (position bias in the old system's log)
SELECT indexOf(shown, clicked) AS slot, count() AS clicks
FROM shop.history_sessions
WHERE clicked IS NOT NULL
GROUP BY slot
ORDER BY slot;







WITH

-- ============================================================
-- 1. REQUEST QUE QUEREMOS RECOMENDAR
-- ============================================================

request AS (
    SELECT
        request_id::String AS request_id,
        user_id::String AS user_id,
        is_new_user::Bool AS is_new_user,
        recent_views,
        context
    FROM shop.requests
    LIMIT 1
),


-- ============================================================
-- 2. CATÁLOGO ACTIVO
-- ============================================================

catalog AS (
    SELECT
        product_id::String AS product_id,
        category::String AS category,
        subcategory::String AS subcategory,
        brand::String AS brand,
        price_eur::Float64 AS price_eur
    FROM shop.catalog
    WHERE is_active::Bool = true
),


-- ============================================================
-- 3. TODAS LAS SEÑALES LIVE EN UNA SOLA PASADA
--
-- 195 minutos permiten calcular:
--   - popularidad última hora
--   - tendencia últimos 15m
-- ============================================================

live_signals AS (
    SELECT
        e.product_id::String AS product_id,

        countIf(
            e.ts > now() - INTERVAL 1 HOUR
            AND e.event_type = 'view'
        )
        +
        5 * countIf(
            e.ts > now() - INTERVAL 1 HOUR
            AND e.event_type = 'add_to_cart'
        )
        +
        10 * countIf(
            e.ts > now() - INTERVAL 1 HOUR
            AND e.event_type = 'purchase'
        )
        AS popularity_score,

        countIf(
            e.ts > now() - INTERVAL 15 MINUTE
            AND e.event_type = 'view'
        )
        AS views_15m,

        countIf(
            e.ts <= now() - INTERVAL 15 MINUTE
            AND e.event_type = 'view'
        )
        AS views_previous_3h

    FROM shop.events AS e

    WHERE e.ts > now() - INTERVAL 195 MINUTE

    GROUP BY e.product_id
),


-- ============================================================
-- 4. TRENDING
-- ============================================================

signals AS (
    SELECT
        product_id,

        popularity_score,

        views_15m,

        views_previous_3h,

        least(
            views_15m /
            greatest(views_previous_3h / 12.0, 1),
            20
        ) AS trending_score

    FROM live_signals
),


-- ============================================================
-- 5. HISTORIAL DEL USUARIO
--
-- IMPORTANTE:
-- Solo analizamos el usuario de la request actual.
-- Esto evita recorrer innecesariamente todo el historial.
-- ============================================================

user_history AS (
    SELECT
        e.user_id::String AS user_id,
        e.product_id::String AS product_id,

        countIf(e.event_type = 'view')
        +
        5 * countIf(e.event_type = 'add_to_cart')
        +
        10 * countIf(e.event_type = 'purchase')
        AS user_product_score

    FROM shop.history_events AS e

    INNER JOIN request AS r
        ON e.user_id::String = r.user_id

    GROUP BY
        e.user_id,
        e.product_id
),


-- ============================================================
-- 6. AFINIDAD DEL USUARIO POR CATEGORÍA Y BRAND
--
-- Una sola pasada del historial.
-- ============================================================

user_affinity AS (
    SELECT
        e.user_id::String AS user_id,

        c.category::String AS category,

        c.brand::String AS brand,

        count() AS interactions

    FROM shop.history_events AS e

    INNER JOIN request AS r
        ON e.user_id::String = r.user_id

    INNER JOIN shop.catalog AS c
        ON c.product_id::String = e.product_id::String

    WHERE e.event_type IN (
        'view',
        'add_to_cart',
        'purchase'
    )

    GROUP BY
        e.user_id,
        c.category,
        c.brand
),


-- ============================================================
-- 7. CANDIDATOS
-- ============================================================

candidates AS (
    SELECT
        c.product_id,
        c.category,
        c.subcategory,
        c.brand,
        c.price_eur,

        r.user_id,
        r.is_new_user,

        coalesce(s.popularity_score, 0)
            AS popularity_score,

        coalesce(s.trending_score, 0)
            AS trending_score,

        coalesce(uh.user_product_score, 0)
            AS user_product_score,

        coalesce(ua.interactions, 0)
            AS affinity_score

    FROM catalog AS c

    CROSS JOIN request AS r

    LEFT JOIN signals AS s
        ON c.product_id = s.product_id

    LEFT JOIN user_history AS uh
        ON c.product_id = uh.product_id
        AND r.user_id = uh.user_id

    LEFT JOIN user_affinity AS ua
        ON c.category = ua.category
        AND c.brand = ua.brand
        AND r.user_id = ua.user_id
),


-- ============================================================
-- 8. NORMALIZACIÓN
--
-- Convertimos las señales a escalas comparables.
-- ============================================================

normalized AS (
    SELECT
        *,

        popularity_score
        / greatest(max(popularity_score) OVER (), 1)
        AS popularity_norm,

        trending_score
        / greatest(max(trending_score) OVER (), 1)
        AS trending_norm,

        user_product_score
        / greatest(max(user_product_score) OVER (), 1)
        AS user_product_norm,

        affinity_score
        / greatest(max(affinity_score) OVER (), 1)
        AS affinity_norm

    FROM candidates
),


-- ============================================================
-- 9. SCORE FINAL
--
-- 35% popularidad
-- 25% tendencia
-- 25% comportamiento del usuario
-- 15% afinidad categoría + brand
-- ============================================================

ranked AS (
    SELECT
        *,

        (
            0.35 * popularity_norm
            +
            0.25 * trending_norm
            +
            0.25 * user_product_norm
            +
            0.15 * affinity_norm
        ) AS final_score

    FROM normalized
)


-- ============================================================
-- 10. RECOMENDACIÓN FINAL
-- ============================================================

SELECT
    product_id,
    category,
    subcategory,
    brand,
    price_eur,

    popularity_norm,
    trending_norm,
    user_product_norm,
    affinity_norm,

    final_score

FROM ranked

ORDER BY
    final_score DESC,
    product_id

LIMIT 10;