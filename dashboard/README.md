# Dashboard Module — Person 3 (Day 10 Final)

**Retail Demand Forecasting & Inventory Optimization**  
Interactive Streamlit dashboard built on the M5 Walmart dataset.

---

## Tech Stack
- Python 3.10+, Streamlit ≥ 1.32, Plotly ≥ 5.20, Pandas ≥ 2.0, NumPy ≥ 1.26

---

## Folder Structure
```
dashboard/
├── app.py                        # Main Streamlit entry point (all 5 pages)
├── requirements.txt              # Dashboard-only dependencies
├── README.md                     # This file
└── components/
    ├── __init__.py
    ├── forecasting_engine.py     # Prophet & LightGBM metrics + forecast series
    ├── inventory_engine.py       # Safety stock, ROP, EOQ, replenishment engine
    └── scenario_engine.py        # Price elasticity & what-if simulation engine
```

---

## Run the Dashboard
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```
Dashboard opens at `http://localhost:8501`

---

## Pages

| Day | Page | Description |
|-----|------|-------------|
| 1 | — | Streamlit app scaffolding & folder structure |
| 2 | 🏠 Overview | Navigation sidebar, project summary, build progress tracker |
| 3 | 📊 Sales Analytics | KPI cards (units, revenue, avg daily, top store), 7-day rolling trend |
| 4 | 📊 Sales Analytics | Store/department/date filters, product velocity leaderboard, CSV export |
| 5 | 🔮 Demand Forecasting | Historical context window, forecast horizon selector |
| 6 | 🔮 Demand Forecasting | Prophet vs LightGBM predicted vs actual, residuals, cumulative volume, model matrix |
| 7 | 📦 Inventory Optimization | Safety stock, ROP, EOQ, sawtooth replenishment cycle, PO export |
| 8 | 🎯 What-If Scenario | Price elasticity simulation, revenue & margin impact, supply chain alert |
| 9 | — | Integration tests (`tests/test_dashboard.py`) — all engines validated |
| 10 | All pages | Final polish, README, screenshots |

---

## Component Engines

### `forecasting_engine.py`
- `load_forecast_series(store_id, horizon_days)` — returns 30-day Prophet / LightGBM / Ensemble forecast DataFrame
- `get_store_metrics_summary()` — MAE, RMSE, WAPE comparison table across all 10 stores
- `MODEL_EVALUATION_METRICS` — benchmark telemetry dict for all stores

### `inventory_engine.py`
- `calculate_safety_stock(d_avg, sigma_d, L, sigma_L, service_level)` — statistical SS formula
- `calculate_reorder_point(d_avg, L, ss)` — ROP = d_avg × L + SS
- `calculate_eoq(D, S, H)` — Wilson EOQ formula
- `generate_inventory_recommendations(df_products, store_id, service_level, ...)` — full SKU replenishment table

### `scenario_engine.py`
- `simulate_price_scenario(base_forecast_df, price_change_pct, elasticity_regime, category_focus)` — returns trajectory, summary KPIs, and category elasticity matrix
- `CATEGORY_ELASTICITIES` — Conservative / Moderate / Aggressive elasticity profiles

---

## Dashboard Screenshots

> Screenshots captured from the final Day 10 build running locally.

### 🏠 Overview Page
![Overview](../reports/figures/dashboard_overview.png)

### 📊 Sales Analytics
![Sales Analytics](../reports/figures/dashboard_sales_analytics.png)

### 🔮 Demand Forecasting
![Demand Forecasting](../reports/figures/dashboard_forecasting.png)

### 📦 Inventory Optimization
![Inventory](../reports/figures/dashboard_inventory.png)

### 🎯 What-If Scenario Analysis
![Scenario](../reports/figures/dashboard_scenario.png)

> *Note: Screenshot images are generated after running the dashboard locally and saving via browser. Placeholder paths shown above — replace with actual screenshots before final submission.*

---

## Tests
```bash
# From project root
pip install pytest
pytest tests/test_dashboard.py -v
```
All 30 unit + integration tests must pass before merging to `main`.

---

## Key Design Decisions
- All data is **simulated from M5 dataset statistics** (no raw CSV required to run the dashboard)
- Inventory parameters (lead time, demand CV, holding cost) are **category-calibrated** from M5 domain knowledge
- Price elasticity values are sourced from **retail economics literature** (FOODS ~-0.95, HOUSEHOLD ~-1.35, HOBBIES ~-1.90)
- `@st.cache_data` used on all heavy data generators to prevent re-computation on widget interaction
