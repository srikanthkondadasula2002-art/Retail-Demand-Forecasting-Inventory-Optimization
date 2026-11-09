-- 05_category_velocity_analysis.sql
-- Department-level sales contribution and category share using window functions

WITH dept_totals AS (
    SELECT
        cat_id,
        dept_id,
        SUM(units) AS dept_units
    FROM store_dept_daily
    GROUP BY cat_id, dept_id
)
SELECT
    cat_id,
    dept_id,
    dept_units,
    SUM(dept_units) OVER (PARTITION BY cat_id) AS category_total_units,
    ROUND(
        CAST(dept_units AS REAL) / SUM(dept_units) OVER (PARTITION BY cat_id) * 100,
        2
    ) AS share_within_category_pct,
    ROUND(
        CAST(dept_units AS REAL) / (SELECT SUM(units) FROM store_dept_daily) * 100,
        2
    ) AS share_of_total_network_pct,
    DENSE_RANK() OVER (PARTITION BY cat_id ORDER BY dept_units DESC) AS dept_rank_in_cat
FROM dept_totals
ORDER BY cat_id, dept_rank_in_cat;
