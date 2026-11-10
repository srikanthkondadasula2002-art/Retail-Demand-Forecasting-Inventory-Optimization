-- agg_monthly_sales.sql
-- Monthly sales rollup by store, category, and state with MoM metrics

{{ config(materialized = 'table') }}

WITH monthly_base AS (
    SELECT
        calendar_year AS year,
        month_num AS month,
        store_id,
        state_id,
        cat_id,
        SUM(sales_units) AS monthly_units
    FROM {{ ref('fct_daily_sales') }}
    GROUP BY calendar_year, month_num, store_id, state_id, cat_id
)
SELECT
    year,
    month,
    store_id,
    state_id,
    cat_id,
    monthly_units,
    LAG(monthly_units, 1) OVER (
        PARTITION BY store_id, cat_id ORDER BY year, month
    ) AS prev_month_units,
    ROUND(
        (CAST(monthly_units AS FLOAT64) - LAG(monthly_units, 1) OVER (PARTITION BY store_id, cat_id ORDER BY year, month)) /
        NULLIF(LAG(monthly_units, 1) OVER (PARTITION BY store_id, cat_id ORDER BY year, month), 0) * 100,
        2
    ) AS mom_growth_pct
FROM monthly_base
ORDER BY store_id, cat_id, year, month
