# M5 Retail Sales Analytical SQL Reporting Summary

> **Dataset**: Walmart M5 Forecasting Dataset (1,913 Historical Days)
> **Database**: In-Memory Relational Analytical Engine (Standard ANSI SQL)

---

## 1. Store Performance Rankings

| sales_rank | store_id | state_id | total_units | network_share_pct | active_days | store_daily_avg_units |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | CA_3 | CA | 11188180 | 17.03 | 1913 | 5848.5 |
| 2 | CA_1 | CA | 7698216 | 11.72 | 1913 | 4024.2 |
| 3 | TX_2 | TX | 7214384 | 10.98 | 1913 | 3771.2 |
| 4 | WI_2 | WI | 6544012 | 9.96 | 1913 | 3420.8 |
| 5 | WI_3 | WI | 6427782 | 9.78 | 1913 | 3360.1 |
| 6 | TX_3 | TX | 6089330 | 9.27 | 1913 | 3183.1 |
| 7 | CA_2 | CA | 5685475 | 8.65 | 1913 | 2972.0 |
| 8 | TX_1 | TX | 5595292 | 8.52 | 1913 | 2924.9 |
| 9 | WI_1 | WI | 5149062 | 7.84 | 1913 | 2691.6 |
| 10 | CA_4 | CA | 4103676 | 6.25 | 1913 | 2145.2 |

## 2. Category & Department Contribution Analysis

| cat_id | dept_id | dept_units | category_total_units | share_within_category_pct | share_of_total_network_pct | dept_rank_in_cat |
| --- | --- | --- | --- | --- | --- | --- |
| FOODS | FOODS_3 | 32372076 | 45089939 | 71.79 | 49.28 | 1 |
| FOODS | FOODS_2 | 7629822 | 45089939 | 16.92 | 11.61 | 2 |
| FOODS | FOODS_1 | 5088041 | 45089939 | 11.28 | 7.74 | 3 |
| HOBBIES | HOBBIES_1 | 5596460 | 6124800 | 91.37 | 8.52 | 1 |
| HOBBIES | HOBBIES_2 | 528340 | 6124800 | 8.63 | 0.8 | 2 |
| HOUSEHOLD | HOUSEHOLD_1 | 11500526 | 14480670 | 79.42 | 17.51 | 1 |
| HOUSEHOLD | HOUSEHOLD_2 | 2980144 | 14480670 | 20.58 | 4.54 | 2 |

## 3. SNAP Policy Benefit Demand Lift

| state_id | snap_active_avg_units | snap_inactive_avg_units | snap_demand_lift_pct |
| --- | --- | --- | --- |
| WI | 10763.6 | 8838.5 | 21.78 |
| TX | 10612.7 | 9519.1 | 11.49 |
| CA | 15771.9 | 14605.8 | 7.98 |

## 4. Weekend Demand Surge Analysis

| cat_id | dept_id | weekend_avg_units | weekday_avg_units | weekend_surge_pct |
| --- | --- | --- | --- | --- |
| HOUSEHOLD | HOUSEHOLD_2 | 1930.7 | 1408.2 | 37.1 |
| HOUSEHOLD | HOUSEHOLD_1 | 7448.5 | 5435.0 | 37.05 |
| FOODS | FOODS_3 | 20477.0 | 15495.0 | 32.15 |
| HOBBIES | HOBBIES_1 | 3441.6 | 2718.3 | 26.61 |
| FOODS | FOODS_2 | 4677.8 | 3711.6 | 26.03 |
| FOODS | FOODS_1 | 3057.2 | 2500.1 | 22.28 |
| HOBBIES | HOBBIES_2 | 305.6 | 264.4 | 15.58 |

## 5. Recent Monthly Rollups & MoM Growth

| year | month | total_units | foods_units | foods_share_pct | household_units | household_share_pct | hobbies_units | hobbies_share_pct | prev_month_units | mom_growth_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2015.0 | 5.0 | 1153014.0 | 741404.0 | 64.3 | 291617.0 | 25.29 | 119993.0 | 10.41 | 1093429.0 | 5.45 |
| 2015.0 | 6.0 | 1155599.0 | 763726.0 | 66.09 | 280030.0 | 24.23 | 111843.0 | 9.68 | 1153014.0 | 0.22 |
| 2015.0 | 7.0 | 1232147.0 | 809505.0 | 65.7 | 303699.0 | 24.65 | 118943.0 | 9.65 | 1155599.0 | 6.62 |
| 2015.0 | 8.0 | 1254341.0 | 811161.0 | 64.67 | 321877.0 | 25.66 | 121303.0 | 9.67 | 1232147.0 | 1.8 |
| 2015.0 | 9.0 | 1188913.0 | 780594.0 | 65.66 | 293785.0 | 24.71 | 114534.0 | 9.63 | 1254341.0 | -5.22 |
| 2015.0 | 10.0 | 1248437.0 | 819910.0 | 65.67 | 300297.0 | 24.05 | 128230.0 | 10.27 | 1188913.0 | 5.01 |
| 2015.0 | 11.0 | 1113976.0 | 719316.0 | 64.57 | 275016.0 | 24.69 | 119644.0 | 10.74 | 1248437.0 | -10.77 |
| 2015.0 | 12.0 | 1126236.0 | 725042.0 | 64.38 | 276665.0 | 24.57 | 124529.0 | 11.06 | 1113976.0 | 1.1 |
| 2016.0 | 1.0 | 1215286.0 | 807440.0 | 66.44 | 290293.0 | 23.89 | 117553.0 | 9.67 | 1126236.0 | 7.91 |
| 2016.0 | 2.0 | 1212927.0 | 804739.0 | 66.35 | 295249.0 | 24.34 | 112939.0 | 9.31 | 1215286.0 | -0.19 |
| 2016.0 | 3.0 | 1275746.0 | 855454.0 | 67.06 | 304051.0 | 23.83 | 116241.0 | 9.11 | 1212927.0 | 5.18 |
| 2016.0 | 4.0 | 1046688.0 | 702312.0 | 67.1 | 247357.0 | 23.63 | 97019.0 | 9.27 | 1275746.0 | -17.95 |
