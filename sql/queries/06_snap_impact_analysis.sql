-- 06_snap_impact_analysis.sql
-- Quantifying the demand lift during SNAP benefit days (1-10 of month) vs non-SNAP days by state

WITH state_daily AS (
    SELECT
        date,
        state_id,
        CASE
            WHEN state_id = 'CA' THEN snap_CA
            WHEN state_id = 'TX' THEN snap_TX
            WHEN state_id = 'WI' THEN snap_WI
            ELSE 0
        END AS is_snap_day,
        SUM(units) AS daily_state_units
    FROM store_dept_daily
    GROUP BY date, state_id, is_snap_day
),
snap_comparison AS (
    SELECT
        state_id,
        is_snap_day,
        COUNT(DISTINCT date) AS day_count,
        SUM(daily_state_units) AS total_units,
        ROUND(AVG(daily_state_units), 1) AS avg_daily_units
    FROM state_daily
    GROUP BY state_id, is_snap_day
)
SELECT
    s.state_id,
    s.avg_daily_units AS snap_active_avg_units,
    ns.avg_daily_units AS snap_inactive_avg_units,
    ROUND(
        (s.avg_daily_units - ns.avg_daily_units) / ns.avg_daily_units * 100,
        2
    ) AS snap_demand_lift_pct
FROM snap_comparison s
JOIN snap_comparison ns
    ON s.state_id = ns.state_id
    AND s.is_snap_day = 1
    AND ns.is_snap_day = 0
ORDER BY snap_demand_lift_pct DESC;
