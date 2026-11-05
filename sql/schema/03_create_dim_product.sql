-- =======================================================
-- Step 3: Product Dimension Table (dim_product)
-- Clustered by category and department
-- =======================================================
CREATE TABLE IF NOT EXISTS `retail-demand-forecasting-dwh.retail_demand_dwh.dim_product`
(
    item_id STRING NOT NULL OPTIONS(description="Unique product identifier (e.g., HOBBIES_1_001)"),
    dept_id STRING NOT NULL OPTIONS(description="Department identifier (e.g., HOBBIES_1, FOODS_3)"),
    cat_id STRING NOT NULL OPTIONS(description="High-level category (FOODS, HOUSEHOLD, HOBBIES)")
)
CLUSTER BY cat_id, dept_id
OPTIONS (
    description = "Dimension table containing product hierarchy and category classifications."
);
