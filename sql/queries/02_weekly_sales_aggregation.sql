-- 02_weekly_sales_aggregation.sql
-- Weekly total sales units, WoW growth rate, and 4-week moving average

WITH weekly_base AS (
    SELECT
        wm_yr_wk,
        MIN(date) AS week_start_date,
        MAX(date) AS week_end_date,
        year,
        SUM(units) AS weekly_units
    FROM store_dept_daily
    GROUP BY wm_yr_wk, year
)
SELECT
    wm_yr_wk,
    week_start_date,
    week_end_date,
    weekly_units,
    LAG(weekly_units, 1) OVER (ORDER BY wm_yr_wk) AS prev_week_units,
    ROUND(
        (CAST(weekly_units AS REAL) - LAG(weekly_units, 1) OVER (ORDER BY wm_yr_wk)) /
        NULLIF(LAG(weekly_units, 1) OVER (ORDER BY wm_yr_wk), 0) * 100,
        2
    ) AS wow_growth_pct,
    ROUND(AVG(weekly_units) OVER (
        ORDER BY wm_yr_wk
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ), 1) AS rolling_4w_avg_units
FROM weekly_base
ORDER BY wm_yr_wk ASC;
