-- stg_sales.sql
-- Staging model for store and department daily sales

WITH source AS (
    SELECT * FROM {{ source('raw_m5', 'store_dept_daily') }}
)
SELECT
    CAST(date AS DATE) AS date,
    CAST(d AS STRING) AS day_index,
    CAST(store_id AS STRING) AS store_id,
    CAST(state_id AS STRING) AS state_id,
    CAST(cat_id AS STRING) AS cat_id,
    CAST(dept_id AS STRING) AS dept_id,
    CAST(units AS INT64) AS sales_units,
    CAST(wm_yr_wk AS INT64) AS wm_yr_wk,
    CAST(year AS INT64) AS year,
    CAST(month AS INT64) AS month,
    CAST(wday AS INT64) AS wday,
    CAST(snap_CA AS INT64) AS snap_CA,
    CAST(snap_TX AS INT64) AS snap_TX,
    CAST(snap_WI AS INT64) AS snap_WI
FROM source
WHERE units >= 0
