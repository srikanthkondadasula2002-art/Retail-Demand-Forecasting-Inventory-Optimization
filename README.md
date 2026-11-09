# Retail Demand Forecasting & Inventory Optimization

Retail Demand Forecasting &amp; Inventory Optimization using Python, SQL, BigQuery, dbt, Prophet, LightGBM, and Streamlit. Analyzes Walmart M5 historical sales data to predict product demand, optimize inventory, and generate data-driven restocking recommendations.

---

## 📌 Project Overview
This repository contains an end-to-end retail forecasting and supply chain optimization system built on the Walmart M5 dataset. The goal is to forecast daily unit sales 28 days into the future across thousands of products and multiple stores, and translate those demand forecasts into actionable inventory policies (Safety Stock, Reorder Points, and Economic Order Quantities).

---

## 🗓️ 14-Day Work Plan
- [x] **Day 1: Environment Setup, Data Acquisition & Deep Inspection**
  - Download and verify M5 dataset integrity across all 5 raw files (429.6 MB total).
  - Profile schemas, sparsity (intermittent demand), product hierarchies, and store distributions.
  - Establish Git version control and data download automation (`download_data.py`, `src/verify_setup.py`).
- [x] **Day 2: Exploratory Data Analysis (EDA) & Demand Profiling**
  - Analyzed product velocity, department contributions, and intermittent demand (68.2% zero-ratio).
  - Profiled store volume rankings, 2.72x performance spread, and 2011–2016 trajectories.
  - Quantified intra-week weekend spikes (+38%) and state-specific SNAP stimulus lifts (+21.8% in WI).
  - Delivered modular pipeline `src/eda_analysis.py`, `src/product_store_eda.py`, figures in `reports/figures/`, and interactive notebook `notebooks/01_exploratory_data_analysis.ipynb`.
- [x] **Day 3: Memory Optimization & Data Preprocessing Pipeline**
  - Downcasting data types (`int64` -> `int16`, `float64` -> `float32`, string -> `category`) achieving **75.9% RAM reduction**.
  - Verified monotonic date continuity (1,969 consecutive days without gaps).
  - Imputed missing event names and types with `'None'` in `src/data_preprocessing.py`.
  - Generated `data/processed/clean_calendar.parquet` and `data/processed/clean_sales_summary.parquet`.
- [x] **Day 4: BigQuery Enterprise Data Warehouse Integration**
  - Configured BigQuery connection manager and client authentication in `src/bigquery_config.py`.
  - Designed partitioned and clustered relational DDL schemas in `sql/schema/` (`dim_calendar`, `dim_product`, `dim_store`, `fct_daily_sales`, `fct_product_performance`, and analytical view `v_sales_kpi_summary`).
  - Implemented automated BigQuery loader pipeline in `src/bigquery_loader.py` (validated 169,428 records with dry-run and live GCP upload support).
- [x] **Day 5: Forecasting Feature Engineering & Anti-Leakage Architecture**
  - Engineered anti-leakage autoregressive lags ($\ge 28\text{D}$: `lag_28`, `lag_35`, `lag_42`, `lag_49`, `lag_56`).
  - Rolling velocity, volatility, and intermittency metrics shifted by 28 days (`roll_mean_7`, `roll_mean_28`, `roll_std_28`, `roll_zero_ratio_28`).
  - Harmonic trigonometric calendar encodings (`sin_wday`, `cos_wday`, `sin_month`, `cos_month`).
  - SNAP interaction flags and 4-week price momentum in `src/feature_engineering.py`.
  - Exported test feature matrix `data/processed/test_features_matrix.parquet`.
- [x] **Day 6: Gradient Boosted Decision Trees (LightGBM)**
  - Formulated Tweedie deviance loss ($p=1.15$) to handle 68.2% zero-inflation and overdispersed retail demand.
  - Generated 28-day out-of-sample predictions achieving an overall **6.80% WAPE** across all 10 stores (champion model in 8 of 10 stores).
  - Exported predictions to `data/predictions/lightgbm_forecast_28d.csv` and telemetry to `reports/lightgbm_evaluation_metrics.json`.
- [x] **Day 7: Time Series Trends, Seasonality Decomposition & Holiday Shocks**
  - Additive STL time series decomposition into Trend, 7-day Weekly Seasonality, and Residuals.
  - Autocorrelation Function (ACF) and PACF confirming multi-week harmonic peaks at lags 7, 14, 21, 28, 35, 56.
  - Quantified demand lift across all 30 calendar events in `src/trend_seasonality.py`.
- [x] **Day 8: Facebook Prophet Hierarchical Forecasting**
  - Implemented tuned Prophet models with multiplicative seasonality, holiday effects, and state SNAP regressors in `src/prophet_model.py`.
  - Achieved an overall **7.79% WAPE** across all 10 stores (flagship store `CA_3` achieved **4.06% WAPE**).
  - Exported predictions to `data/predictions/prophet_forecast_28d.csv` and telemetry to `reports/prophet_evaluation_metrics.json`.
- [x] **Day 9: Analytical SQL Query Suite & Sales Aggregations**
  - Developed standard ANSI SQL analytical queries in `sql/queries/` covering daily, weekly, and monthly rollups.
  - Store performance ranking via window functions (`RANK() OVER (...)`).
  - Category velocity contribution, SNAP benefit lift, and weekend surge analysis.
  - Automated SQL execution pipeline in `src/run_sql_analytics.py` with tabular outputs in `reports/sql_analytics/`.
- [ ] **Day 10: Inventory Optimization Engine**
  - Calculation of Safety Stock under uncertain lead time and demand.
  - Reorder Points (ROP) and Economic Order Quantity (EOQ).
- [ ] **Day 11: Inventory Simulation & Stockout Prevention**
  - Backtesting inventory replenishment policies over 28-day evaluation horizon.
  - Quantifying holding costs vs. stockout penalty trade-offs.
- [x] **Day 12: Interactive Dashboard Development (Streamlit)**
  - Implemented real-time interactive sales analytics dashboard in `app.py`.
  - Built polished Sales KPI cards (Units, Gross Revenue, Velocity, Catalog Count, Weekend Surge).
  - Implemented dynamic sales trend charts with 7D/30D smoothing, store benchmarking, category mix, and intra-week rhythm.
- [ ] **Day 13: Model Evaluation, Error Analysis & Hyperparameter Tuning**
  - Deep-dive into underperforming departments and intermittent items.
- [ ] **Day 14: Documentation, Testing & Final Deployment**
  - Production packaging and cloud deployment.

---

## 📊 Dataset Structure & Data Cleaning Audit
The dataset comprises point-of-sale transactional data from Walmart across 10 stores in California, Texas, and Wisconsin:
- **`calendar.csv`** (1,969 days): Dates, day-of-week, fiscal week (`wm_yr_wk`), 30 special event types, and SNAP benefit flags.
- **`sales_train_validation.csv`** (30,490 time series × 1,913 days): Unit sales per product & store from `d_1` (2011-01-29) to `d_1913` (2016-04-24).
- **`sales_train_evaluation.csv`** (30,490 time series × 1,941 days): Includes validation window up to `d_1941`.
- **`sell_prices.csv`** (6.84M rows): Weekly selling prices by store, item, and fiscal week.
- **`sample_submission.csv`** (60,980 rows): Submission template for 28-day forecast horizons (`F1`–`F28`).

### Data Cleaning & Memory Optimization Audit
- **RAM Optimization**: Total memory footprint was reduced from **766.6 MB to 184.5 MB (-75.93% reduction)** via schema downcasting.
- **Missing Value Resolution**:
  - `event_name_1`: 1,807 nulls imputed with `'None'`.
  - `event_type_1`: 1,807 nulls imputed with `'None'`.
  - `event_name_2`: 1,964 nulls imputed with `'None'`.
  - `event_type_2`: 1,964 nulls imputed with `'None'`.
- **Date Continuity**: Verified strictly continuous date range from `2011-01-29` to `2016-06-19` (1,969 days, zero missing dates).
- **Integrity**: 0 null sales values, 0 negative values across all 30,490 time series.

*Telemetry: [`reports/data_cleaning_audit.json`](reports/data_cleaning_audit.json)*

---

## 🚀 Pipeline Execution Guide

### 1. Verify Environment & Datasets
```bash
python src/verify_setup.py
```

### 2. Run Data Preprocessing & Cleaning Pipeline
```bash
python src/data_preprocessing.py
```

### 3. Run Product & Store Exploratory Data Analysis (Pareto ABC)
```bash
python src/product_store_eda.py
```

### 4. Run Time Series Trend, Seasonality & Holiday Shocks Analysis
```bash
python src/trend_seasonality.py
```

### 5. Run Forecasting Feature Engineering Pipeline
```bash
python src/feature_engineering.py
```

### 6. Run Analytical SQL Queries & Sales Aggregations
```bash
python src/run_sql_analytics.py
```

### 7. Run Machine Learning Forecasting Models
```bash
# LightGBM Tweedie Model (Champion: 6.80% WAPE)
python src/lightgbm_model.py

# Facebook Prophet Model (7.79% WAPE)
python src/prophet_model.py
```

### 8. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📈 Key Insights & Analytical Findings

### 1. ABC Inventory Classification (Pareto Analysis)
- **Class A**: **36.6% of series** generate **80.0% of total revenue** (\$150.7M).
- **Class B**: **30.8% of series** generate **15.0% of total revenue** (\$28.3M).
- **Class C**: **32.6% of series** generate **5.0% of total revenue** (\$9.4M).

### 2. Product Velocity Clustering
- **Fast Movers**: 5.1% of catalog (1,560 series), 15.3% zero-demand days, 6.90 average daily units.
- **Medium Movers**: 26.9% of catalog (8,199 series), 45.4% zero-demand days, 2.01 average daily units.
- **Slow Movers**: 37.9% of catalog (11,544 series), 73.1% zero-demand days, 0.65 average daily units.
- **Lumpy / Intermittent**: 30.1% of catalog (9,187 series), 92.4% zero-demand days, 0.17 average daily units.

### 3. Temporal Dynamics & Holiday Shocks
- **Annual Growth**: M5 network daily sales grew from 27,135 to 40,754 units (~8.1% annualized growth).
- **Harmonic Cycles**: ACF displays strong cyclical peaks at **lag 7 (0.808)**, **lag 14 (0.736)**, **lag 21 (0.750)**, and **lag 28 (0.816)**.
- **Top Demand Shocks**:
  - Labor Day: **+22.2%** demand lift
  - Super Bowl: **+18.7%** demand lift
  - Easter: **+16.6%** demand lift
  - Christmas Day (`Dec 25`): **-99.4%** drop (store closure)

### 4. Policy Stimulus (SNAP Benefits)
- **Wisconsin (WI)**: **+21.8%** sales lift during SNAP active days (10,764 vs 8,839 units/day).
- **Texas (TX)**: **+11.5%** sales lift during SNAP active days (10,613 vs 9,519 units/day).
- **California (CA)**: **+8.0%** sales lift during SNAP active days (15,772 vs 14,606 units/day).

---

## ⚡ Model Performance Benchmark (28-Day Blind Horizon)

| Store ID | State | Prophet WAPE (%) | LightGBM WAPE (%) | Performance Delta | Champion Model |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`CA_1`** | CA | 6.38% | **5.33%** | -1.05% | 🏆 **LightGBM** |
| **`CA_2`** | CA | **4.68%** | 9.22% | +4.54% | 🏆 **Prophet** |
| **`CA_3`** | CA | **4.06%** | 4.97% | +0.91% | 🏆 **Prophet** |
| **`CA_4`** | CA | 8.41% | **7.28%** | -1.13% | 🏆 **LightGBM** |
| **`TX_1`** | TX | 8.63% | **6.97%** | -1.66% | 🏆 **LightGBM** |
| **`TX_2`** | TX | 6.49% | **5.68%** | -0.81% | 🏆 **LightGBM** |
| **`TX_3`** | TX | 8.84% | **8.30%** | -0.54% | 🏆 **LightGBM** |
| **`WI_1`** | WI | 8.43% | **7.46%** | -0.97% | 🏆 **LightGBM** |
| **`WI_2`** | WI | 13.48% | **7.30%** | **-6.18%** | 🏆 **LightGBM** |
| **`WI_3`** | WI | 9.55% | **6.26%** | -3.29% | 🏆 **LightGBM** |
| **OVERALL** | **ALL** | **7.79%** | **6.80%** | **-0.99%** | 🏆 **LightGBM (Overall Champion)** |

---

## 🏛️ Project Directory Layout

```text
├── app.py                                   # Streamlit Sales Analytics Dashboard
├── download_data.py                         # Zenodo dataset downloader
├── inspect_m5.py                            # Schema, sparsity & memory profiler
├── inspect_aggregations.py                  # Hierarchical aggregations & pricing inspector
├── requirements.txt                         # Python dependencies
├── README.md                                # Executive project documentation
├── data/
│   ├── calendar.csv                         # 1,969 days calendar metadata
│   ├── sales_train_validation.csv           # 30,490 series x 1,913 days
│   ├── sales_train_evaluation.csv           # 30,490 series x 1,941 days
│   ├── sell_prices.csv                      # 6.84M weekly store-item prices
│   ├── sample_submission.csv                # 28-day submission format
│   ├── processed/
│   │   ├── clean_calendar.parquet           # Cleaned and typed calendar table
│   │   ├── clean_sales_summary.parquet      # 30,490 series summary metrics
│   │   ├── store_dept_daily.parquet         # Store-department daily rollups
│   │   ├── product_performance.parquet      # Store-item performance metrics
│   │   └── test_features_matrix.parquet     # Anti-leakage test feature matrix
│   └── predictions/
│       ├── lightgbm_forecast_28d.csv        # 28-day LightGBM forecast predictions
│       └── prophet_forecast_28d.csv         # 28-day Prophet forecast predictions
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb   # Interactive EDA notebook
├── reports/
│   ├── environment_setup.json               # Environment verification report
│   ├── data_cleaning_audit.json             # Cleaning and memory audit
│   ├── abc_classification_summary.json      # Pareto ABC classification
│   ├── trend_seasonality_metrics.json       # STL, ACF/PACF and holiday lift
│   ├── feature_engineering_metadata.json    # Anti-leakage features metadata
│   ├── lightgbm_evaluation_metrics.json     # GBDT error metrics per store
│   ├── prophet_evaluation_metrics.json      # Prophet error metrics per store
│   ├── figures/                             # 14 publication-grade figures
│   └── sql_analytics/                       # Tabular SQL outputs & markdown summary
├── sql/
│   ├── schema/                              # BigQuery DDL schema definitions
│   └── queries/                             # 7 analytical SQL aggregation queries
└── src/
    ├── config.py                            # Centralized paths and taxonomies
    ├── verify_setup.py                      # Environment and dataset validator
    ├── data_preprocessing.py                # Preprocessing and downcasting pipeline
    ├── product_store_eda.py                 # ABC Pareto and velocity clustering
    ├── trend_seasonality.py                 # Time series decomposition and holiday shocks
    ├── feature_engineering.py               # Anti-leakage feature engineering
    ├── run_sql_analytics.py                 # In-memory SQL execution engine
    ├── eda_analysis.py                      # Exploratory data analysis suite
    ├── bigquery_config.py                   # GCP BigQuery connection manager
    ├── bigquery_loader.py                   # BigQuery ingestion pipeline
    ├── lightgbm_model.py                    # LightGBM Tweedie forecasting model
    └── prophet_model.py                     # Facebook Prophet forecasting model
```

---

## 🛠️ Tech Stack
- **Languages & Frameworks**: Python 3.14, Pandas, NumPy, Scikit-Learn
- **Modeling**: LightGBM (Tweedie), Facebook Prophet
- **Data Warehousing & Transforms**: Google BigQuery, SQL (ANSI SQLite / BigQuery Standard SQL)
- **Visualization & UI**: Streamlit, Plotly, Matplotlib, Seaborn
