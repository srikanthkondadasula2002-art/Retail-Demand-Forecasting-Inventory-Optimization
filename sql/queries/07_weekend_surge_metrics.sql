-- 07_weekend_surge_metrics.sql
-- Weekend demand surge analysis across departments (wday 1, 2 = Saturday, Sunday vs weekdays)

WITH dept_day_type AS (
    SELECT
        cat_id,
        dept_id,
        CASE WHEN wday IN (1, 2) THEN 'Weekend' ELSE 'Weekday' END AS day_type,
        COUNT(DISTINCT date) AS day_count,
        SUM(units) AS total_units,
        ROUND(CAST(SUM(units) AS REAL) / COUNT(DISTINCT date), 1) AS avg_daily_units
    FROM store_dept_daily
    GROUP BY cat_id, dept_id, day_type
)
SELECT
    w.cat_id,
    w.dept_id,
    w.avg_daily_units AS weekend_avg_units,
    wd.avg_daily_units AS weekday_avg_units,
    ROUND(
        (w.avg_daily_units - wd.avg_daily_units) / wd.avg_daily_units * 100,
        2
    ) AS weekend_surge_pct
FROM dept_day_type w
JOIN dept_day_type wd
    ON w.dept_id = wd.dept_id
    AND w.day_type = 'Weekend'
    AND wd.day_type = 'Weekday'
ORDER BY weekend_surge_pct DESC;
