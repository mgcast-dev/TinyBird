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
FROM team_NN.feedback
WHERE request_id != '__init__'
GROUP BY hour, served_by
ORDER BY hour DESC, served_by;


-- 6. Reliability: why did the fallback serve your shoppers, and how fast are you?
SELECT fallback_reason::String              AS fallback_reason,
       count()                              AS requests,
       quantile(0.5)(latency_ms::Float64)   AS p50_ms,
       quantile(0.95)(latency_ms::Float64)  AS p95_ms
FROM team_NN.feedback
WHERE ts > now() - INTERVAL 30 MINUTE AND request_id != '__init__'
GROUP BY fallback_reason;


-- 7. Which slots get clicked? (position bias in the old system's log)
SELECT indexOf(shown, clicked) AS slot, count() AS clicks
FROM shop.history_sessions
WHERE clicked IS NOT NULL
GROUP BY slot
ORDER BY slot;


-- 1. Current catalogue
-- Current state of every product, including the latest catalog update.
SELECT
    product_id,
    argMax(category, v)    AS category,
    argMax(subcategory, v) AS subcategory,
    argMax(brand, v)       AS brand,
    argMax(price_eur, v)   AS price_eur,
    argMax(is_active, v)   AS is_active,
    argMax(created_at, v)  AS created_at
FROM (
    SELECT
        product_id::String AS product_id,
        category::String AS category,
        subcategory::String AS subcategory,
        brand::String AS brand,
        price_eur::Float64 AS price_eur,
        is_active::Bool AS is_active,
        created_at::DateTime64(3) AS created_at,
        '' AS v
    FROM shop.catalog

    UNION ALL

    SELECT
        product_id::String,
        category::String,
        subcategory::String,
        brand::String,
        price_eur::Float64,
        is_active::Bool,
        created_at::DateTime64(3),
        toString(ts)
    FROM shop.catalog_updates
)
GROUP BY product_id;


-- 2. Product popularity in the last hour
-- Live organic activity, restricted to products that are currently active.
SELECT
    c.category::String AS category,
    e.product_id::String AS product_id,
    countIf(e.event_type = 'view')        AS views,
    countIf(e.event_type = 'add_to_cart') AS carts,
    countIf(e.event_type = 'purchase')    AS purchases
FROM shop.events AS e
INNER JOIN shop.catalog AS c
    ON c.product_id::String = e.product_id::String
WHERE e.ts > now() - INTERVAL 1 HOUR
GROUP BY category, product_id
ORDER BY purchases DESC, carts DESC, views DESC
LIMIT 5 BY category;


-- 3. What is trending right now?
-- Compare views in the last 15 minutes with the previous 3 hours.
SELECT
    c.subcategory::String AS subcategory,

    countIf(
        e.ts > now() - INTERVAL 15 MINUTE
    ) AS views_15m,

    countIf(
        e.ts <= now() - INTERVAL 15 MINUTE
        AND e.ts > now() - INTERVAL 195 MINUTE
    ) / 12.0 AS avg_views_per_15m_before,

    views_15m /
        greatest(avg_views_per_15m_before, 1) AS lift

FROM shop.events AS e

INNER JOIN shop.catalog AS c
    ON c.product_id::String = e.product_id::String

WHERE e.event_type = 'view'
  AND e.ts > now() - INTERVAL 195 MINUTE

GROUP BY subcategory

ORDER BY lift DESC
LIMIT 10;


-- 4. Co-views for a given product
-- Products that users historically viewed in the same session.
SELECT
    b.product_id AS also_viewed,
    count() AS sessions
FROM (
    SELECT
        session_id::String AS session_id,
        product_id::String AS product_id
    FROM shop.history_events
    WHERE event_type = 'view'
) AS a
INNER JOIN (
    SELECT
        session_id::String AS session_id,
        product_id::String AS product_id
    FROM shop.history_events
    WHERE event_type = 'view'
) AS b
    ON a.session_id = b.session_id
WHERE a.product_id = 'p_xxxxxx'
  AND b.product_id != a.product_id
GROUP BY also_viewed
ORDER BY sessions DESC
LIMIT 20;


-- 5. Your feedback: conversion and revenue per hour
-- Compare different recommendation strategies / served_by values.
SELECT
    toStartOfHour(ts::DateTime64(3)) AS hour,
    served_by::String AS served_by,
    count() AS requests,
    100 * countIf(clicked != '') / count() AS ctr_pct,
    100 * countIf(purchased != '') / count() AS conversion_pct,
    sum(revenue_eur::Float64) / count() AS revenue_per_request
FROM team_NN.feedback
WHERE request_id != '__init__'
GROUP BY hour, served_by
ORDER BY hour DESC, served_by;


-- 6. Reliability
-- Monitor fallback reasons and latency.
SELECT
    fallback_reason::String AS fallback_reason,
    count() AS requests,
    quantile(0.5)(latency_ms::Float64) AS p50_ms,
    quantile(0.95)(latency_ms::Float64) AS p95_ms
FROM team_NN.feedback
WHERE ts > now() - INTERVAL 30 MINUTE
  AND request_id != '__init__'
GROUP BY fallback_reason
ORDER BY requests DESC;


-- 7. Position bias in the old recommendation system
-- Measure how often each carousel position received the click.
SELECT
    indexOf(shown, clicked) AS slot,
    count() AS clicks
FROM shop.history_sessions
WHERE clicked IS NOT NULL
GROUP BY slot
ORDER BY slot;


-- 8. Weighted product popularity in the last hour
-- Purchases have more weight than add-to-cart, which has more weight than views.
SELECT
    e.product_id::String AS product_id,

    countIf(e.event_type = 'view') AS views,
    countIf(e.event_type = 'add_to_cart') AS carts,
    countIf(e.event_type = 'purchase') AS purchases,

    countIf(e.event_type = 'view')
        + 5 * countIf(e.event_type = 'add_to_cart')
        + 10 * countIf(e.event_type = 'purchase')
        AS popularity_score

FROM shop.events AS e

WHERE e.ts > now() - INTERVAL 1 HOUR

GROUP BY product_id

ORDER BY popularity_score DESC
LIMIT 100;


-- 9. Product-level trending score
-- Detect products whose activity is increasing right now.
SELECT
    e.product_id::String AS product_id,

    countIf(
        e.ts > now() - INTERVAL 15 MINUTE
    ) AS views_15m,

    countIf(
        e.ts <= now() - INTERVAL 15 MINUTE
        AND e.ts > now() - INTERVAL 195 MINUTE
    ) AS views_previous_3h,

    views_15m /
        greatest(views_previous_3h / 12.0, 1) AS trending_score

FROM shop.events AS e

WHERE e.event_type = 'view'
  AND e.ts > now() - INTERVAL 195 MINUTE

GROUP BY product_id

ORDER BY trending_score DESC
LIMIT 100;


-- 10. User-category affinity from historical organic behaviour
-- history_events does not contain category, so we recover it through the catalogue.
SELECT
    e.user_id::String AS user_id,
    c.category::String AS category,
    count() AS interactions

FROM shop.history_events AS e

INNER JOIN shop.catalog AS c
    ON c.product_id::String = e.product_id::String

WHERE e.event_type IN ('view', 'add_to_cart', 'purchase')

GROUP BY user_id, category

ORDER BY user_id, interactions DESC;


-- 11. User-brand affinity from historical organic behaviour
-- Brand is obtained through shop.catalog.
SELECT
    e.user_id::String AS user_id,
    c.brand::String AS brand,
    count() AS interactions

FROM shop.history_events AS e

INNER JOIN shop.catalog AS c
    ON c.product_id::String = e.product_id::String

WHERE e.event_type IN ('view', 'add_to_cart', 'purchase')

GROUP BY user_id, brand

ORDER BY user_id, interactions DESC;


-- 12. Recent views are NOT reconstructed from history_events.
-- They are already provided directly in shop.requests/history_sessions
-- and include the products viewed earlier in the current session.
--
-- We therefore use recent_views directly when generating recommendations.
-- No historical query is required here.


-- 13. Historical user-product interaction score
-- Useful for learning what each user tends to interact with.
SELECT
    user_id,
    product_id,
    views,
    carts,
    purchases,

    views
        + 5 * carts
        + 10 * purchases
        AS interaction_score,

    last_interaction

FROM (
    SELECT
        user_id::String AS user_id,
        product_id::String AS product_id,

        countIf(event_type = 'view') AS views,
        countIf(event_type = 'add_to_cart') AS carts,
        countIf(event_type = 'purchase') AS purchases,

        max(ts) AS last_interaction

    FROM shop.history_events

    GROUP BY user_id, product_id
);


-- 14. Normalised user-category affinity
-- Shows the percentage of each user's historical interactions
-- that belong to each category.
SELECT
    user_id,
    category,
    interactions,

    interactions
        / sum(interactions) OVER (PARTITION BY user_id)
        AS category_affinity

FROM (
    SELECT
        e.user_id::String AS user_id,
        c.category::String AS category,
        count() AS interactions

    FROM shop.history_events AS e

    INNER JOIN shop.catalog AS c
        ON c.product_id::String = e.product_id::String

    WHERE e.event_type IN ('view', 'add_to_cart', 'purchase')

    GROUP BY user_id, category
)

ORDER BY user_id, category_affinity DESC;


-- 15. Candidate generation
-- Combine several sources of candidates:
-- current active products + popularity + recent trending activity.
SELECT
    c.product_id::String AS product_id,
    c.category::String AS category,
    c.subcategory::String AS subcategory,
    c.brand::String AS brand,
    c.price_eur::Float64 AS price_eur,

    coalesce(p.popularity_score, 0) AS popularity_score,
    coalesce(t.trending_score, 0) AS trending_score

FROM shop.catalog AS c

LEFT JOIN (
    SELECT
        product_id::String AS product_id,

        countIf(event_type = 'view')
            + 5 * countIf(event_type = 'add_to_cart')
            + 10 * countIf(event_type = 'purchase')
            AS popularity_score

    FROM shop.events

    WHERE ts > now() - INTERVAL 1 HOUR

    GROUP BY product_id
) AS p
    ON c.product_id::String = p.product_id

LEFT JOIN (
    SELECT
        product_id::String AS product_id,

        countIf(
            ts > now() - INTERVAL 15 MINUTE
        )
        /
        greatest(
            countIf(
                ts <= now() - INTERVAL 15 MINUTE
                AND ts > now() - INTERVAL 195 MINUTE
            ) / 12.0,
            1
        ) AS trending_score

    FROM shop.events

    WHERE event_type = 'view'
      AND ts > now() - INTERVAL 195 MINUTE

    GROUP BY product_id
) AS t
    ON c.product_id::String = t.product_id

WHERE c.is_active = true

ORDER BY
    popularity_score DESC,
    trending_score DESC

LIMIT 200;


-- 16. Global ranking baseline
-- First complete recommendation baseline:
-- popularity + real-time trend.
SELECT
    c.product_id::String AS product_id,
    c.category::String AS category,
    c.subcategory::String AS subcategory,
    c.brand::String AS brand,
    c.price_eur::Float64 AS price_eur,

    coalesce(p.popularity_score, 0) AS popularity_score,
    coalesce(t.trending_score, 0) AS trending_score,

    (
        0.60 * coalesce(p.popularity_score, 0)
        +
        0.40 * coalesce(t.trending_score, 0)
    ) AS final_score

FROM shop.catalog AS c

LEFT JOIN (
    SELECT
        product_id::String AS product_id,

        countIf(event_type = 'view')
            + 5 * countIf(event_type = 'add_to_cart')
            + 10 * countIf(event_type = 'purchase')
            AS popularity_score

    FROM shop.events

    WHERE ts > now() - INTERVAL 1 HOUR

    GROUP BY product_id
) AS p
    ON c.product_id::String = p.product_id

LEFT JOIN (
    SELECT
        product_id::String AS product_id,

        countIf(
            ts > now() - INTERVAL 15 MINUTE
        )
        /
        greatest(
            countIf(
                ts <= now() - INTERVAL 15 MINUTE
                AND ts > now() - INTERVAL 195 MINUTE
            ) / 12.0,
            1
        ) AS trending_score

    FROM shop.events

    WHERE event_type = 'view'
      AND ts > now() - INTERVAL 195 MINUTE

    GROUP BY product_id
) AS t
    ON c.product_id::String = t.product_id

WHERE c.is_active = true

ORDER BY final_score DESC

LIMIT 10;


-- 17. Current-request personalised ranking
-- recent_views is taken directly from the current request.
-- Replace 'USER_ID_HERE' with the user_id of the request being served.
--
-- The important idea is:
--   recent_views = immediate user intent
--   popularity = what is working now
--   trending = what is accelerating now
--
-- Products already viewed in the current session are excluded.
SELECT
    c.product_id::String AS product_id,
    c.category::String AS category,
    c.subcategory::String AS subcategory,
    c.brand::String AS brand,
    c.price_eur::Float64 AS price_eur,

    coalesce(p.popularity_score, 0) AS popularity_score,
    coalesce(t.trending_score, 0) AS trending_score

FROM shop.catalog AS c

LEFT JOIN (
    SELECT
        product_id::String AS product_id,

        countIf(event_type = 'view')
            + 5 * countIf(event_type = 'add_to_cart')
            + 10 * countIf(event_type = 'purchase')
            AS popularity_score

    FROM shop.events

    WHERE ts > now() - INTERVAL 1 HOUR

    GROUP BY product_id
) AS p
    ON c.product_id::String = p.product_id

LEFT JOIN (
    SELECT
        product_id::String AS product_id,

        countIf(
            ts > now() - INTERVAL 15 MINUTE
        )
        /
        greatest(
            countIf(
                ts <= now() - INTERVAL 15 MINUTE
                AND ts > now() - INTERVAL 195 MINUTE
            ) / 12.0,
            1
        ) AS trending_score

    FROM shop.events

    WHERE event_type = 'view'
      AND ts > now() - INTERVAL 195 MINUTE

    GROUP BY product_id
) AS t
    ON c.product_id::String = t.product_id

WHERE c.is_active = true

ORDER BY
    (
        0.60 * coalesce(p.popularity_score, 0)
        +
        0.40 * coalesce(t.trending_score, 0)
    ) DESC

LIMIT 10;