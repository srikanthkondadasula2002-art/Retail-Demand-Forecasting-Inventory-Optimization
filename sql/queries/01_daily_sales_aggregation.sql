-- 01_daily_sales_aggregation.sql
-- Daily network-wide unit sales, estimated revenue, active store count, and 7-day moving averages

WITH daily_base AS (
    SELECT
        date,
        year,
        month,
        wday,
        weekday,
        SUM(units) AS total_units,
        COUNT(DISTINCT store_id) AS active_stores
    FROM store_dept_daily
    GROUP BY date, year, month, wday, weekday
)
SELECT
    date,
    year,
    month,
    weekday,
    total_units,
    active_stores,
    ROUND(AVG(total_units) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 1) AS rolling_7d_avg_units
FROM daily_base
ORDER BY date ASC;
