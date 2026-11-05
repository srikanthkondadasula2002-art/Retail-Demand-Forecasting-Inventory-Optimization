-- =======================================================
-- Step 5: Fact Table: Daily Store & Department Sales (fct_daily_sales)
-- Partitioned by date, clustered by store_id and cat_id
-- =======================================================
CREATE TABLE IF NOT EXISTS `retail-demand-forecasting-dwh.retail_demand_dwh.fct_daily_sales`
(
    date DATE NOT NULL OPTIONS(description="Transaction date"),
    d STRING NOT NULL OPTIONS(description="Relative day index (e.g. d_1)"),
    store_id STRING NOT NULL OPTIONS(description="Store identifier"),
    state_id STRING NOT NULL OPTIONS(description="US State code"),
    cat_id STRING NOT NULL OPTIONS(description="Product category"),
    dept_id STRING NOT NULL OPTIONS(description="Product department"),
    units INT64 NOT NULL OPTIONS(description="Total daily units sold for this partition"),
    wm_yr_wk INT64 NOT NULL OPTIONS(description="Walmart fiscal week identifier"),
    weekday STRING NOT NULL OPTIONS(description="Day of week name"),
    wday INT64 NOT NULL OPTIONS(description="Day of week number (1-7)"),
    month INT64 NOT NULL OPTIONS(description="Month (1-12)"),
    year INT64 NOT NULL OPTIONS(description="Year (2011-2016)"),
    snap_active INT64 NOT NULL OPTIONS(description="1 if SNAP was active in this store's state, else 0")
)
PARTITION BY date
CLUSTER BY store_id, cat_id, dept_id
OPTIONS (
    description = "Partitioned daily sales transaction table at the store-department granularity."
);
