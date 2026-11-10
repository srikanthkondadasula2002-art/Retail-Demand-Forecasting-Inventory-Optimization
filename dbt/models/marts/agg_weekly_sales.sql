-- agg_weekly_sales.sql
-- Weekly sales aggregations with week-over-week (WoW) growth rates

{{ config(materialized = 'table') }}

WITH base AS (
    SELECT
        wm_yr_wk,
        MIN(date) AS week_start_date,
        MAX(date) AS week_end_date,
        store_id,
        cat_id,
        dept_id,
        SUM(sales_units) AS weekly_units
    FROM {{ ref('fct_daily_sales') }}
    GROUP BY wm_yr_wk, store_id, cat_id, dept_id
)
SELECT
    wm_yr_wk,
    week_start_date,
    week_end_date,
    store_id,
    cat_id,
    dept_id,
    weekly_units,
    LAG(weekly_units, 1) OVER (
        PARTITION BY store_id, dept_id ORDER BY wm_yr_wk
    ) AS prev_week_units,
    ROUND(
        (CAST(weekly_units AS FLOAT64) - LAG(weekly_units, 1) OVER (PARTITION BY store_id, dept_id ORDER BY wm_yr_wk)) /
        NULLIF(LAG(weekly_units, 1) OVER (PARTITION BY store_id, dept_id ORDER BY wm_yr_wk), 0) * 100,
        2
    ) AS wow_growth_pct,
    ROUND(AVG(weekly_units) OVER (
        PARTITION BY store_id, dept_id ORDER BY wm_yr_wk
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ), 1) AS rolling_4w_avg_units
FROM base
ORDER BY store_id, dept_id, wm_yr_wk
