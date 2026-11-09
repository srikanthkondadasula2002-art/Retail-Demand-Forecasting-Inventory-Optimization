-- 04_store_performance_rankings.sql
-- Store performance rankings by total units, active days, and average daily volume

WITH store_totals AS (
    SELECT
        store_id,
        state_id,
        COUNT(DISTINCT date) AS active_days,
        SUM(units) AS total_units,
        ROUND(AVG(units), 1) AS avg_dept_daily_units,
        ROUND(CAST(SUM(units) AS REAL) / COUNT(DISTINCT date), 1) AS store_daily_avg_units
    FROM store_dept_daily
    GROUP BY store_id, state_id
)
SELECT
    RANK() OVER (ORDER BY total_units DESC) AS sales_rank,
    store_id,
    state_id,
    total_units,
    ROUND(CAST(total_units AS REAL) / (SELECT SUM(units) FROM store_dept_daily) * 100, 2) AS network_share_pct,
    active_days,
    store_daily_avg_units
FROM store_totals
ORDER BY sales_rank ASC;
