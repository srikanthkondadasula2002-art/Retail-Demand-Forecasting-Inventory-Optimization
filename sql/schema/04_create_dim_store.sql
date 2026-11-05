-- =======================================================
-- Step 4: Store Dimension Table (dim_store)
-- =======================================================
CREATE TABLE IF NOT EXISTS `retail-demand-forecasting-dwh.retail_demand_dwh.dim_store`
(
    store_id STRING NOT NULL OPTIONS(description="Store identifier (e.g., CA_1, TX_2, WI_3)"),
    state_id STRING NOT NULL OPTIONS(description="US State code (CA, TX, WI)")
)
CLUSTER BY state_id
OPTIONS (
    description = "Dimension table containing retail store locations and state mappings."
);
