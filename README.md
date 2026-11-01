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
- [ ] **Day 2: Exploratory Data Analysis (EDA) & Demand Profiling**
  - Category, store, and department sales trends.
  - Weekend surges, holiday shocks, and SNAP program analysis.
- [ ] **Day 3: Memory Optimization & Data Pipeline (Wide-to-Long Melt)**
  - Downcasting data types (`int16`, `float16`, categoricals).
  - Merging `calendar.csv` and `sell_prices.csv`.
- [ ] **Day 4: Feature Engineering I (Lags & Rolling Aggregations)**
  - Autoregressive lags ($\ge 28$ days to prevent data leakage).
  - Rolling means, standard deviations, and zero-demand ratios.
- [ ] **Day 5: Feature Engineering II (Calendar, Events, SNAP & Pricing)**
  - Day-of-week, month, weekend indicators, Christmas closures.
  - Price momentum, discount depth, and SNAP-food interactions.
- [ ] **Day 6: Baseline Forecasting Models**
  - Naive, Moving Average, and Exponential Smoothing baselines.
  - Evaluate using RMSE, WAPE, and WRMSSE.
- [ ] **Day 7: Gradient Boosting with LightGBM & CatBoost**
  - Tweedie deviance objective function to address 68% zero inflation.
  - Group time-series cross-validation.
- [ ] **Day 8: Prophet & Hierarchical Forecasting**
  - Department and store-level trend & seasonality decomposition.
  - Bottom-up vs. top-down hierarchical reconciliation.
- [ ] **Day 9: SQL & BigQuery / dbt Transformation Layer**
  - Modeling feature tables and aggregated metrics in BigQuery.
  - dbt models for automated daily forecast tracking.
- [ ] **Day 10: Inventory Optimization Engine**
  - Calculation of Safety Stock under uncertain lead time and demand.
  - Reorder Points (ROP) and Economic Order Quantity (EOQ).
- [ ] **Day 11: Inventory Simulation & Stockout Prevention**
  - Backtesting inventory replenishment policies over 28-day evaluation horizon.
  - Quantifying holding costs vs. stockout penalty trade-offs.
- [ ] **Day 12: Interactive Dashboard Development (Streamlit)**
  - Interactive store/item demand forecaster.
  - Inventory health, stockout risk alerts, and reorder recommendations.
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

---

## 🛠️ Tech Stack
- **Languages & Frameworks**: Python 3, Pandas, NumPy, Scikit-Learn
- **Modeling**: LightGBM (Tweedie), CatBoost, Prophet
- **Data Warehousing & Transforms**: Google BigQuery, SQL, dbt
- **Visualization & UI**: Streamlit, Plotly, Matplotlib, Seaborn
