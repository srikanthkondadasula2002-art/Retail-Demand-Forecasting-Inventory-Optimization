-- mart_forecast_evaluation.sql
-- Comparative analytical mart evaluating LightGBM vs Prophet against ground truth actuals

{{ config(materialized = 'view') }}

WITH base AS (
    SELECT
        store_id,
        state_id,
        date,
        model_name,
        forecast_units,
        actual_units,
        absolute_error,
        percentage_error
    FROM {{ source('raw_m5', 'fct_forecasts') }}
)
SELECT
    store_id,
    state_id,
    date,
    actual_units,
    MAX(CASE WHEN model_name = 'LightGBM_Tweedie' THEN forecast_units END) AS lightgbm_forecast,
    MAX(CASE WHEN model_name = 'Prophet' THEN forecast_units END) AS prophet_forecast,
    MAX(CASE WHEN model_name = 'LightGBM_Tweedie' THEN absolute_error END) AS lightgbm_abs_err,
    MAX(CASE WHEN model_name = 'Prophet' THEN absolute_error END) AS prophet_abs_err,
    CASE
        WHEN MAX(CASE WHEN model_name = 'LightGBM_Tweedie' THEN absolute_error END) <=
             MAX(CASE WHEN model_name = 'Prophet' THEN absolute_error END)
        THEN 'LightGBM'
        ELSE 'Prophet'
    END AS winning_model
FROM base
GROUP BY store_id, state_id, date, actual_units
ORDER BY store_id, date
