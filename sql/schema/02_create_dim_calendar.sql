-- =======================================================
-- Step 2: Calendar Dimension Table (dim_calendar)
-- Partitioned by date for fast temporal range scans
-- =======================================================
CREATE TABLE IF NOT EXISTS `retail-demand-forecasting-dwh.retail_demand_dwh.dim_calendar`
(
    date DATE NOT NULL OPTIONS(description="Calendar date in YYYY-MM-DD format"),
    wm_yr_wk INT64 NOT NULL OPTIONS(description="Walmart fiscal year week identifier"),
    weekday STRING NOT NULL OPTIONS(description="Name of day of week (e.g., Saturday)"),
    wday INT64 NOT NULL OPTIONS(description="Numeric day of week (1=Saturday .. 7=Friday)"),
    month INT64 NOT NULL OPTIONS(description="Calendar month (1-12)"),
    year INT64 NOT NULL OPTIONS(description="Calendar year"),
    d STRING NOT NULL OPTIONS(description="Day identifier (d_1 to d_1969)"),
    event_name_1 STRING OPTIONS(description="Primary holiday or special event name"),
    event_type_1 STRING OPTIONS(description="Primary event category (Cultural, National, Religious, Sporting)"),
    event_name_2 STRING OPTIONS(description="Secondary holiday event name"),
    event_type_2 STRING OPTIONS(description="Secondary event category"),
    snap_CA INT64 NOT NULL OPTIONS(description="1 if SNAP vouchers active in California, else 0"),
    snap_TX INT64 NOT NULL OPTIONS(description="1 if SNAP vouchers active in Texas, else 0"),
    snap_WI INT64 NOT NULL OPTIONS(description="1 if SNAP vouchers active in Wisconsin, else 0")
)
PARTITION BY date
OPTIONS (
    description = "Dimension table containing calendar, holiday, and state SNAP benefits metadata."
);
