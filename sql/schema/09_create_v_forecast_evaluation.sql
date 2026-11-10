-- =======================================================
-- Step 9: Analytical View: Model Forecast Comparison (v_forecast_evaluation)
-- Joins LightGBM and Prophet forecasts side-by-side with actuals
-- =======================================================
CREATE OR REPLACE VIEW `retail-demand-forecasting-dwh.retail_demand_dwh.v_forecast_evaluation` AS
WITH pivoted AS (
    SELECT
        store_id,
        state_id,
        date,
        actual_units,
        MAX(CASE WHEN model_name = 'LightGBM_Tweedie' THEN forecast_units END) AS lightgbm_forecast,
        MAX(CASE WHEN model_name = 'LightGBM_Tweedie' THEN absolute_error END) AS lightgbm_abs_error,
        MAX(CASE WHEN model_name = 'Prophet' THEN forecast_units END) AS prophet_forecast,
        MAX(CASE WHEN model_name = 'Prophet' THEN absolute_error END) AS prophet_abs_error
    FROM `retail-demand-forecasting-dwh.retail_demand_dwh.fct_forecasts`
    GROUP BY store_id, state_id, date, actual_units
)
SELECT
    store_id,
    state_id,
    date,
    actual_units,
    ROUND(lightgbm_forecast, 1) AS lightgbm_forecast,
    ROUND(prophet_forecast, 1) AS prophet_forecast,
    ROUND(lightgbm_forecast - actual_units, 1) AS lightgbm_delta,
    ROUND(prophet_forecast - actual_units, 1) AS prophet_delta,
    CASE
        WHEN lightgbm_abs_error <= prophet_abs_error THEN 'LightGBM'
        ELSE 'Prophet'
    END AS winning_model
FROM pivoted
ORDER BY store_id, date;
