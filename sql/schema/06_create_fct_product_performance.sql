-- =======================================================
-- Step 6: Fact Table: Product Performance & Velocity Metrics (fct_product_performance)
-- Clustered by store_id, cat_id, dept_id
-- =======================================================
CREATE TABLE IF NOT EXISTS `retail-demand-forecasting-dwh.retail_demand_dwh.fct_product_performance`
(
    item_id STRING NOT NULL OPTIONS(description="Product SKU identifier"),
    store_id STRING NOT NULL OPTIONS(description="Store identifier"),
    state_id STRING NOT NULL OPTIONS(description="State code"),
    cat_id STRING NOT NULL OPTIONS(description="Product category"),
    dept_id STRING NOT NULL OPTIONS(description="Product department"),
    total_units INT64 NOT NULL OPTIONS(description="Total cumulative units sold over historical period"),
    zero_days INT64 NOT NULL OPTIONS(description="Number of days with zero sales recorded"),
    total_days INT64 NOT NULL OPTIONS(description="Total historical days tracked"),
    sparsity_pct FLOAT64 NOT NULL OPTIONS(description="Percentage of days with zero demand (intermittency metric)"),
    avg_sell_price FLOAT64 NOT NULL OPTIONS(description="Average selling price for this item and store"),
    est_revenue FLOAT64 NOT NULL OPTIONS(description="Estimated gross dollar revenue generated")
)
CLUSTER BY store_id, cat_id, dept_id
OPTIONS (
    description = "Pre-aggregated product-store lifecycle performance and velocity metrics table."
);
