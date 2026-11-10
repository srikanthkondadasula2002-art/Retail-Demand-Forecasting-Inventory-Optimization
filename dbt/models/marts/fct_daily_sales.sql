-- fct_daily_sales.sql
-- Daily store and department sales fact table enriched with calendar dimensions

{{
  config(
    materialized = 'table',
    partition_by = {
      "field": "date",
      "data_type": "date",
      "granularity": "day"
    },
    cluster_by = ["store_id", "cat_id", "dept_id"]
  )
}}

WITH sales AS (
    SELECT * FROM {{ ref('stg_sales') }}
),
calendar AS (
    SELECT * FROM {{ ref('stg_calendar') }}
)
SELECT
    s.date,
    s.day_index,
    s.store_id,
    s.state_id,
    s.cat_id,
    s.dept_id,
    s.sales_units,
    c.wm_yr_wk,
    c.weekday_name,
    c.day_of_week_num,
    c.month_num,
    c.calendar_year,
    c.event_name_1,
    c.event_type_1,
    c.is_weekend,
    c.is_christmas,
    CASE
        WHEN s.state_id = 'CA' THEN c.snap_CA
        WHEN s.state_id = 'TX' THEN c.snap_TX
        WHEN s.state_id = 'WI' THEN c.snap_WI
        ELSE 0
    END AS snap_active
FROM sales s
LEFT JOIN calendar c ON s.date = c.date
