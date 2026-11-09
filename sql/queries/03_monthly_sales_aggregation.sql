-- 03_monthly_sales_aggregation.sql
-- Monthly rollup by year and month with category breakdown and MoM growth rate

WITH monthly_category AS (
    SELECT
        year,
        month,
        cat_id,
        SUM(units) AS monthly_units
    FROM store_dept_daily
    GROUP BY year, month, cat_id
),
monthly_totals AS (
    SELECT
        year,
        month,
        SUM(monthly_units) AS total_units,
        SUM(CASE WHEN cat_id = 'FOODS' THEN monthly_units ELSE 0 END) AS foods_units,
        SUM(CASE WHEN cat_id = 'HOUSEHOLD' THEN monthly_units ELSE 0 END) AS household_units,
        SUM(CASE WHEN cat_id = 'HOBBIES' THEN monthly_units ELSE 0 END) AS hobbies_units
    FROM monthly_category
    GROUP BY year, month
)
SELECT
    year,
    month,
    total_units,
    foods_units,
    ROUND(CAST(foods_units AS REAL) / total_units * 100, 2) AS foods_share_pct,
    household_units,
    ROUND(CAST(household_units AS REAL) / total_units * 100, 2) AS household_share_pct,
    hobbies_units,
    ROUND(CAST(hobbies_units AS REAL) / total_units * 100, 2) AS hobbies_share_pct,
    LAG(total_units, 1) OVER (ORDER BY year, month) AS prev_month_units,
    ROUND(
        (CAST(total_units AS REAL) - LAG(total_units, 1) OVER (ORDER BY year, month)) /
        NULLIF(LAG(total_units, 1) OVER (ORDER BY year, month), 0) * 100,
        2
    ) AS mom_growth_pct
FROM monthly_totals
ORDER BY year ASC, month ASC;
