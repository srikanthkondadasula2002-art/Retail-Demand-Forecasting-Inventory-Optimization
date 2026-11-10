-- =======================================================
-- Step 8: Fact Table: 28-Day Demand Forecasts (fct_forecasts)
-- Stores predictions from Facebook Prophet and LightGBM Tweedie models
-- Partitioned by date, clustered by store_id and model_name
-- =======================================================
CREATE TABLE IF NOT EXISTS `retail-demand-forecasting-dwh.retail_demand_dwh.fct_forecasts`
(
    store_id STRING NOT NULL OPTIONS(description="Store identifier (e.g. CA_1, TX_2)"),
    state_id STRING NOT NULL OPTIONS(description="US State code (CA, TX, WI)"),
    date DATE NOT NULL OPTIONS(description="Forecast date in evaluation window"),
    model_name STRING NOT NULL OPTIONS(description="Forecasting model name (LightGBM_Tweedie or Prophet)"),
    forecast_units FLOAT64 NOT NULL OPTIONS(description="Predicted daily demand units"),
    actual_units INT64 NOT NULL OPTIONS(description="Observed ground truth actual units"),
    error_units FLOAT64 NOT NULL OPTIONS(description="Forecast error: predicted - actual"),
    absolute_error FLOAT64 NOT NULL OPTIONS(description="Absolute forecast error: |predicted - actual|"),
    percentage_error FLOAT64 OPTIONS(description="Percentage error: (predicted - actual) / actual * 100"),
    lower_bound FLOAT64 OPTIONS(description="Lower confidence bound (if provided)"),
    upper_bound FLOAT64 OPTIONS(description="Upper confidence bound (if provided)")
)
PARTITION BY date
CLUSTER BY store_id, model_name
OPTIONS (
    description = "Partitioned demand forecasts and ground truth actuals for 28-day blind test horizon."
);
