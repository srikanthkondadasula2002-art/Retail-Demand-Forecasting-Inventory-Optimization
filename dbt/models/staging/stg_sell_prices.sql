-- stg_sell_prices.sql
-- Staging model for weekly sell prices

WITH source AS (
    SELECT * FROM {{ source('raw_m5', 'sell_prices') }}
)
SELECT
    CAST(store_id AS STRING) AS store_id,
    CAST(item_id AS STRING) AS item_id,
    CAST(wm_yr_wk AS INT64) AS wm_yr_wk,
    CAST(sell_price AS FLOAT64) AS sell_price,
    ROUND(CAST(sell_price AS FLOAT64) * 0.70, 2) AS estimated_unit_cost
FROM source
WHERE sell_price > 0
