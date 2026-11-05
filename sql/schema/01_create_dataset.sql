-- =======================================================
-- Step 1: Create BigQuery Dataset for Retail Demand DWH
-- =======================================================
CREATE SCHEMA IF NOT EXISTS `retail-demand-forecasting-dwh.retail_demand_dwh`
OPTIONS (
    location = 'US',
    description = 'Retail Demand Forecasting & Inventory Optimization Data Warehouse'
);
