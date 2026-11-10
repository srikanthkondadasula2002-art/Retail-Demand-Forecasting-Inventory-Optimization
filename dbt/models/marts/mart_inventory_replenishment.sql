-- mart_inventory_replenishment.sql
-- Clean data mart for replenishment planning: Safety Stock, ROP, and EOQ

{{ config(materialized = 'table') }}

WITH daily_stats AS (
    SELECT
        store_id,
        state_id,
        cat_id,
        dept_id,
        AVG(sales_units) AS mean_daily_demand,
        STDDEV(sales_units) AS std_daily_demand
    FROM {{ ref('fct_daily_sales') }}
    GROUP BY store_id, state_id, cat_id, dept_id
)
SELECT
    store_id,
    state_id,
    cat_id,
    dept_id,
    ROUND(mean_daily_demand, 2) AS mean_daily_demand,
    ROUND(std_daily_demand, 2) AS std_daily_demand,
    -- 95% Service Level: Z = 1.645, Lead Time = 7 days
    CEIL(1.645 * std_daily_demand * SQRT(7)) AS safety_stock_units,
    CEIL(mean_daily_demand * 7) AS lead_time_demand_units,
    CEIL((mean_daily_demand * 7) + (1.645 * std_daily_demand * SQRT(7))) AS reorder_point_units,
    -- Wilson EOQ with Setup Cost S=$50, Holding Rate H=25%
    CEIL(SQRT((2 * (mean_daily_demand * 365.25) * 50.0) / (2.50 * 0.25))) AS eoq_order_units
FROM daily_stats
