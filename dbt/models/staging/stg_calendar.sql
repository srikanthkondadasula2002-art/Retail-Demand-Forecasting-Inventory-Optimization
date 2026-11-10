-- stg_calendar.sql
-- Cleans and standardizes the M5 calendar table

WITH source AS (
    SELECT * FROM {{ source('raw_m5', 'calendar') }}
)
SELECT
    CAST(date AS DATE) AS date,
    CAST(wm_yr_wk AS INT64) AS wm_yr_wk,
    CAST(weekday AS STRING) AS weekday_name,
    CAST(wday AS INT64) AS day_of_week_num,
    CAST(month AS INT64) AS month_num,
    CAST(year AS INT64) AS calendar_year,
    CAST(d AS STRING) AS day_index,
    COALESCE(event_name_1, 'None') AS event_name_1,
    COALESCE(event_type_1, 'None') AS event_type_1,
    COALESCE(event_name_2, 'None') AS event_name_2,
    COALESCE(event_type_2, 'None') AS event_type_2,
    CAST(snap_CA AS INT64) AS snap_CA,
    CAST(snap_TX AS INT64) AS snap_TX,
    CAST(snap_WI AS INT64) AS snap_WI,
    CASE WHEN wday IN (1, 2) THEN 1 ELSE 0 END AS is_weekend,
    CASE WHEN month = 12 AND EXTRACT(DAY FROM CAST(date AS DATE)) = 25 THEN 1 ELSE 0 END AS is_christmas
FROM source
