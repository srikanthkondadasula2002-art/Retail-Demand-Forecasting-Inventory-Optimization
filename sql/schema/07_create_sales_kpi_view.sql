-- =======================================================
-- Step 7: Analytical View: Monthly & Weekly Sales KPIs (v_sales_kpi_summary)
-- Provides instant business metrics for BI tools & dashboards
-- =======================================================
CREATE OR REPLACE VIEW `retail-demand-forecasting-dwh.retail_demand_dwh.v_sales_kpi_summary` AS
WITH daily_base AS (
    SELECT
        date,
        store_id,
        state_id,
        cat_id,
        dept_id,
        units,
        weekday,
        snap_active,
        CASE WHEN weekday IN ('Saturday', 'Sunday') THEN 1 ELSE 0 END AS is_weekend,
        DATE_TRUNC(date, MONTH) AS sales_month,
        DATE_TRUNC(date, WEEK(MONDAY)) AS sales_week
    FROM `retail-demand-forecasting-dwh.retail_demand_dwh.fct_daily_sales`
)
SELECT
    sales_month,
    store_id,
    state_id,
    cat_id,
    dept_id,
    SUM(units) AS monthly_units,
    AVG(units) AS avg_daily_units,
    SUM(CASE WHEN is_weekend = 1 THEN units ELSE 0 END) AS weekend_units,
    SUM(CASE WHEN is_weekend = 0 THEN units ELSE 0 END) AS weekday_units,
    SUM(CASE WHEN snap_active = 1 THEN units ELSE 0 END) AS snap_days_units,
    SUM(CASE WHEN snap_active = 0 THEN units ELSE 0 END) AS nonsnap_days_units
FROM daily_base
GROUP BY 
    sales_month,
    store_id,
    state_id,
    cat_id,
    dept_id;
