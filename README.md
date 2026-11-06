# Retail Demand Forecasting & Inventory Optimization

Retail Demand Forecasting &amp; Inventory Optimization using Python, SQL, BigQuery, dbt, Prophet, LightGBM, and Streamlit. Analyzes Walmart M5 historical sales data to predict product demand, optimize inventory, and generate data-driven restocking recommendations.

---

## 📌 Project Overview
This repository contains an end-to-end retail forecasting and supply chain optimization system built on the Walmart M5 dataset. The goal is to forecast daily unit sales 28 days into the future across thousands of products and multiple stores, and translate those demand forecasts into actionable inventory policies (Safety Stock, Reorder Points, and Economic Order Quantities).

---

## 🗓️ 14-Day Work Plan
- [x] **Day 1: Environment Setup, Data Acquisition & Deep Inspection**
  - Download and verify M5 dataset integrity.
  - Profile schemas, sparsity (intermittent demand), product hierarchies, and store distributions.
  - Establish Git version control and data download automation.
- [x] **Day 2: Exploratory Data Analysis (EDA) & Demand Profiling**
  - Analyzed product velocity, department contributions, and intermittent demand (68.2% zero-ratio).
  - Profiled store volume rankings, 2.72x performance spread, and 2011–2016 trajectories.
  - Quantified intra-week weekend spikes (+38%) and state-specific SNAP stimulus lifts (+21.8% in WI).
  - Delivered modular pipeline `src/eda_analysis.py`, figures in `reports/figures/`, and interactive notebook `notebooks/01_exploratory_data_analysis.ipynb`.
- [ ] **Day 3: Memory Optimization & Data Pipeline (Wide-to-Long Melt)**
  - Downcasting data types (`int16`, `float16`, categoricals).
  - Merging `calendar.csv` and `sell_prices.csv`.
- [x] **Day 4: BigQuery Enterprise Data Warehouse Integration**
  - Configured BigQuery connection manager and client authentication in `src/bigquery_config.py`.
  - Designed partitioned and clustered relational DDL schemas in `sql/schema/` (`dim_calendar`, `dim_product`, `dim_store`, `fct_daily_sales`, `fct_product_performance`, and analytical view `v_sales_kpi_summary`).
  - Implemented automated BigQuery loader pipeline in `src/bigquery_loader.py` (validated 169,428 records with dry-run and live GCP upload support).
- [ ] **Day 5: Feature Engineering II (Calendar, Events, SNAP & Pricing)**
  - Day-of-week, month, weekend indicators, Christmas closures.
  - Price momentum, discount depth, and SNAP-food interactions.
- [ ] **Day 6: Baseline Forecasting Models**
  - Naive, Moving Average, and Exponential Smoothing baselines.
  - Evaluate using RMSE, WAPE, and WRMSSE.
- [ ] **Day 7: Gradient Boosting with LightGBM & CatBoost**
  - Tweedie deviance objective function to address 68% zero inflation.
  - Group time-series cross-validation.
- [x] **Day 8: Facebook Prophet Hierarchical Forecasting**
  - Implemented tuned Prophet models with multiplicative seasonality, holiday effects, and state SNAP regressors in `src/prophet_model.py`.
  - Generated 28-day out-of-sample forecast horizon (`d_1914` to `d_1941`) and evaluated against true transactional actuals.
  - Achieved an overall **7.79% WAPE** across all 10 stores (flagship store `CA_3` achieved **4.06% WAPE**).
  - Exported predictions to `data/predictions/prophet_forecast_28d.csv` and telemetry to `reports/prophet_evaluation_metrics.json`.
- [ ] **Day 9: SQL & BigQuery / dbt Transformation Layer**
  - Modeling feature tables and aggregated metrics in BigQuery.
  - dbt models for automated daily forecast tracking.
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
  - Production packaging and documentation.

---

## 📊 Dataset Structure
The dataset comprises point-of-sale transactional data from Walmart across 10 stores in California, Texas, and Wisconsin:
- **`calendar.csv`** (1,969 days): Dates, day-of-week, fiscal week (`wm_yr_wk`), 30 special event types, and SNAP benefit flags.
- **`sales_train_validation.csv`** (30,490 time series × 1,913 days): Unit sales per product & store from `d_1` (2011-01-29) to `d_1913` (2016-04-24).
- **`sales_train_evaluation.csv`** (30,490 time series × 1,941 days): Includes validation window up to `d_1941`.
- **`sell_prices.csv`** (6.84M rows): Weekly selling prices by store, item, and fiscal week.
- **`sample_submission.csv`** (60,980 rows): Submission template for 28-day forecast horizons (`F1`–`F28`).

### Key Inspection Insights
- **Intermittent Demand**: **68.20%** of all daily observations are zero sales (`HOBBIES`: 77.3%, `HOUSEHOLD`: 71.8%, `FOODS`: 62.0%).
- **Sales Volume by Category**: `FOODS` represents **68.6%** of all units sold, followed by `HOUSEHOLD` (22.0%) and `HOBBIES` (9.3%).
- **Regional Contribution**: California accounts for **43.7%** of volume (with store `CA_3` contributing 17.0% alone), Texas 28.8%, Wisconsin 27.6%.
- **Seasonality & Shocks**:
  - **Weekend Peak**: Weekend demand is **+38% higher** than midweek.
  - **Holiday Closure**: Walmart is closed on Christmas Day (`Dec 25`); sales drop to near zero.
  - **SNAP Distribution**: First 10–15 days of each month show significant demand surges in `FOODS`.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/srikanthkondadasula2002-art/Retail-Demand-Forecasting-Inventory-Optimization.git
cd Retail-Demand-Forecasting-Inventory-Optimization
```

### 2. Download M5 Dataset
Automated script to fetch and unpack the verified M5 archive into `data/`:
```bash
python download_data.py
```

### 3. Run Profiling & Aggregation Inspections
```bash
# General schema, sparsity, and memory inspection
python inspect_m5.py

# Hierarchical aggregation, day-of-week, and pricing analysis
python inspect_aggregations.py
```

### 4. Run Full Exploratory Data Analysis (EDA) Pipeline
```bash
# Generates store/product figures in reports/figures/ and metrics in reports/
python src/eda_analysis.py

# Or explore interactively via Jupyter:
# notebooks/01_exploratory_data_analysis.ipynb
```

### 5. Launch the Interactive Sales Analytics Dashboard
```bash
# Install dependencies
pip install -r requirements.txt

# Start the Streamlit application
streamlit run app.py
```

### 6. BigQuery Data Warehouse Integration
```bash
# Validate schemas, constraints, and partitioning in dry-run mode
python src/bigquery_loader.py --dry-run

# Load processed telemetry to live Google Cloud BigQuery
python src/bigquery_loader.py --project your-gcp-project-id --dataset retail_demand_dwh
```

### 7. Run Facebook Prophet Demand Forecasting Model
```bash
# Train multi-scale seasonal models with holiday and SNAP regressors
# Evaluates 28-day blind predictions against ground-truth actuals
python src/prophet_model.py
```

---

## 🔮 Facebook Prophet Model Performance (28-Day Horizon)
Tuned multi-scale Prophet models with holiday schedules, Christmas closure flags, and state SNAP regressors achieved an overall **7.79% WAPE** across all 10 Walmart locations:
- **`CA_3` (Flagship Location)**: **4.06% WAPE** (MAE: 254.3 units/day, Total Forecast: 174,302 vs Actual: 175,091).
- **`CA_2`**: **4.68% WAPE** (MAE: 222.2 units/day).
- **`CA_1`**: **6.38% WAPE** (MAE: 305.4 units/day).
- **`TX_2`**: **6.49% WAPE** (MAE: 267.0 units/day).
- **Network Overall**: **7.79% WAPE** (1.15M total forecast vs 1.23M actual units).

*Predictions saved to [`data/predictions/prophet_forecast_28d.csv`](data/predictions/prophet_forecast_28d.csv) and visualized in [`reports/figures/figure7_prophet_forecast_actuals.png`](reports/figures/figure7_prophet_forecast_actuals.png).*

---

## 🏛️ BigQuery Data Warehouse Architecture
The enterprise analytics layer in BigQuery is designed using a Kimball star-schema:
- **`dim_calendar`**: 1,969 records partitioned by `date` (DAY) with fiscal calendar, holiday types, and SNAP indicators.
- **`dim_product`**: 3,049 items clustered by `cat_id` and `dept_id`.
- **`dim_store`**: 10 retail stores clustered by `state_id`.
- **`fct_daily_sales`**: 133,910 rows partitioned by `date` (DAY) and clustered by `store_id`, `cat_id`, and `dept_id` for performant filtered slice-and-dice queries.
- **`fct_product_performance`**: 30,490 store-item records with velocity metrics, intermittency zero ratios, and gross revenue.
- **`v_sales_kpi_summary`**: Materialized analytical view aggregating monthly, weekly, and weekend sales throughput.

---

## 🖥️ Sales Analytics Dashboard Features
The interactive dashboard in [`app.py`](app.py) provides sub-second analytics powered by precomputed Parquet telemetry:
- **Executive Sales KPI Cards**: Real-time cards displaying *Total Units Sold*, *Estimated Gross Revenue ($)*, *Daily Velocity*, *Active Catalog SKU Count*, and *Weekend Surge Lift (+%)*.
- **Interactive Multi-Granularity Trends**: Toggleable smoothing between *Raw Daily Actuals*, *7-Day Moving Average*, and *30-Day Moving Average* with hover inspection.
- **Store Performance Benchmarks**: Multi-line longitudinal trajectory comparison across all 10 Walmart stores with state-level volume rankings.
- **Category & Department Area Composition**: Longitudinal demand composition highlighting grocery vs. general merchandise velocity.
- **Intra-Week & Macro Drivers**: Monday–Sunday demand curves and SNAP benefit vs. non-SNAP sales lift charts for CA, TX, and WI.
- **Product Leaderboard**: Top products ranked by volume and revenue with dynamic filtering and CSV export.

---

## 📈 Key EDA Findings & Visualizations
- **Store-Wise Trajectory**: Store `CA_3` is the top-performing location (\$32.1M revenue, 17.0% volume share), whereas `WI_2` underwent the most significant secular growth from 2011 to 2016.
- **Product-Wise Concentration**: `FOODS` drives 68.6% of volume (`FOODS_3` alone represents 49.3% of the total catalog).
- **Intermittent Demand**: 68.2% of all daily item observations are zeros (`HOBBIES_2` reaches 88.5% zero-ratio), demanding Tweedie or Croston's loss formulations.
- **Intra-Week Seasonality**: Saturday and Sunday demand surges by **+38%** over midweek baselines.
- **SNAP Benefit Stimulus**: Generates a **+21.8% demand lift** in Wisconsin, +11.5% in Texas, and +8.0% in California.

*All generated figures are saved in [`reports/figures/`](reports/figures/) and documented in the interactive notebook [`notebooks/01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb).*

---

## 🛠️ Tech Stack
- **Languages & Frameworks**: Python 3, Pandas, NumPy, Scikit-Learn
- **Modeling**: LightGBM (Tweedie), CatBoost, Prophet
- **Data Warehousing & Transforms**: Google BigQuery, SQL, dbt
- **Visualization & UI**: Streamlit, Plotly, Matplotlib, Seaborn
