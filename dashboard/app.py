import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from components.forecasting_engine import (
    MODEL_EVALUATION_METRICS,
    load_forecast_series,
    get_store_metrics_summary
)
from components.inventory_engine import (
    generate_inventory_recommendations,
    SERVICE_LEVEL_Z
)
from components.scenario_engine import (
    simulate_price_scenario,
    CATEGORY_ELASTICITIES
)

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .main { background-color: #f8f9fa; }
    .page-header {
        background: linear-gradient(90deg, #1a1a2e, #0f3460);
        padding: 20px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        color: white;
    }
    .page-header h1 { margin: 0; font-size: 26px; color: white; }
    .page-header p  { margin: 4px 0 0; font-size: 14px; color: #adb5bd; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #e94560;
        text-align: center;
    }
    .metric-card h2 { margin: 0; font-size: 24px; color: #1a1a2e; }
    .metric-card p { margin: 4px 0 0; font-size: 13px; color: #6c757d; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "🏠 Overview"

PAGES = [
    "🏠 Overview",
    "📊 Sales Analytics",
    "🔮 Demand Forecasting",
    "📦 Inventory Optimization",
    "🎯 What-If Scenario",
]


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 RetailIQ")
    st.markdown("**Demand Forecasting &**  \n**Inventory Optimization**")
    st.markdown("---")
    st.markdown("### Navigation")
    for p in PAGES:
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
            st.rerun()
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown("📁 **M5 Walmart**")
    st.markdown("🏪 10 Stores · 3 States")
    st.markdown("📦 3,049 Products")
    st.markdown("📅 1,941 Days")
    st.markdown("---")
    st.markdown("### Team")
    st.markdown("👤 Person 1 — Data Engineering")
    st.markdown("👤 Person 2 — ML & Forecasting")
    st.markdown("👤 Person 3 — Dashboard")
    st.markdown("---")
    st.caption("v1.0 · Day 8 Build")


# ── Helpers ──────────────────────────────────────────────────────────────────
def page_header(title, subtitle):
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)


def metric_card(value, label):
    st.markdown(
        f'<div class="metric-card"><h2>{value}</h2><p>{label}</p></div>',
        unsafe_allow_html=True
    )


# ── Simulated M5 Sales Data & Product Catalog ────────────────────────────────
DEPT_MAP = {
    "FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"],
    "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"],
    "HOBBIES": ["HOBBIES_1", "HOBBIES_2"]
}

@st.cache_data
def load_sales_data():
    np.random.seed(42)
    dates      = pd.date_range("2014-01-01", "2016-04-24", freq="D")
    stores     = ["CA_1","CA_2","CA_3","CA_4","TX_1","TX_2","TX_3","WI_1","WI_2","WI_3"]
    categories = ["FOODS","HOUSEHOLD","HOBBIES"]
    base       = {"CA_1":420,"CA_2":390,"CA_3":510,"CA_4":370,
                  "TX_1":340,"TX_2":310,"TX_3":290,
                  "WI_1":280,"WI_2":260,"WI_3":250}
    cat_split  = {"FOODS":0.686,"HOUSEHOLD":0.220,"HOBBIES":0.094}
    dept_weights = {
        "FOODS_1": 0.18, "FOODS_2": 0.28, "FOODS_3": 0.54,
        "HOUSEHOLD_1": 0.65, "HOUSEHOLD_2": 0.35,
        "HOBBIES_1": 0.75, "HOBBIES_2": 0.25
    }

    rows = []
    for d in dates:
        dow_boost = 1.38 if d.dayofweek >= 5 else 1.0
        for s in stores:
            for c in categories:
                depts = DEPT_MAP[c]
                cat_units = base[s] * cat_split[c] * dow_boost * np.random.uniform(0.85, 1.15)
                for dept in depts:
                    units = int(cat_units * dept_weights[dept])
                    price = 3.25 if c == "FOODS" else (6.50 if c == "HOUSEHOLD" else 8.20)
                    revenue = round(units * price * np.random.uniform(0.9, 1.1), 2)
                    rows.append({
                        "date": d,
                        "store": s,
                        "category": c,
                        "department": dept,
                        "units": units,
                        "revenue": revenue
                    })
    return pd.DataFrame(rows)


@st.cache_data
def load_product_catalog():
    np.random.seed(42)
    products = [
        {"item_id": "FOODS_3_090", "item_name": "Fresh Bananas (lb)", "department": "FOODS_3", "category": "FOODS", "base_price": 1.25, "sales_velocity": 450},
        {"item_id": "FOODS_3_586", "item_name": "Whole Milk 1 Gallon", "department": "FOODS_3", "category": "FOODS", "base_price": 3.48, "sales_velocity": 380},
        {"item_id": "FOODS_2_197", "item_name": "Organic Roma Tomatoes", "department": "FOODS_2", "category": "FOODS", "base_price": 2.15, "sales_velocity": 290},
        {"item_id": "FOODS_1_043", "item_name": "White Sandwich Bread", "department": "FOODS_1", "category": "FOODS", "base_price": 1.98, "sales_velocity": 240},
        {"item_id": "FOODS_3_252", "item_name": "Cage-Free Large Eggs", "department": "FOODS_3", "category": "FOODS", "base_price": 2.89, "sales_velocity": 320},
        {"item_id": "FOODS_3_714", "item_name": "Avocados 4-Pack", "department": "FOODS_3", "category": "FOODS", "base_price": 3.99, "sales_velocity": 210},
        {"item_id": "HOUSEHOLD_1_118", "item_name": "Paper Towels 12-Pack", "department": "HOUSEHOLD_1", "category": "HOUSEHOLD", "base_price": 12.98, "sales_velocity": 180},
        {"item_id": "HOUSEHOLD_1_245", "item_name": "Liquid Laundry Detergent", "department": "HOUSEHOLD_1", "category": "HOUSEHOLD", "base_price": 14.50, "sales_velocity": 160},
        {"item_id": "HOUSEHOLD_2_056", "item_name": "Trash Bags 80-Count", "department": "HOUSEHOLD_2", "category": "HOUSEHOLD", "base_price": 9.75, "sales_velocity": 130},
        {"item_id": "HOUSEHOLD_1_489", "item_name": "Disinfecting Wipes 3-Pack", "department": "HOUSEHOLD_1", "category": "HOUSEHOLD", "base_price": 7.45, "sales_velocity": 175},
        {"item_id": "HOBBIES_1_001", "item_name": "Classic Lego Creator Set", "department": "HOBBIES_1", "category": "HOBBIES", "base_price": 24.99, "sales_velocity": 85},
        {"item_id": "HOBBIES_1_120", "item_name": "Colored Pencil 48-Pack", "department": "HOBBIES_1", "category": "HOBBIES", "base_price": 8.99, "sales_velocity": 95},
        {"item_id": "HOBBIES_2_034", "item_name": "Acrylic Paint Set 12x", "department": "HOBBIES_2", "category": "HOBBIES", "base_price": 14.99, "sales_velocity": 65},
        {"item_id": "HOBBIES_1_354", "item_name": "Stainless Steel Craft Scissors", "department": "HOBBIES_1", "category": "HOBBIES", "base_price": 5.49, "sales_velocity": 70},
        {"item_id": "FOODS_1_211", "item_name": "Greek Yogurt Vanilla 32oz", "department": "FOODS_1", "category": "FOODS", "base_price": 4.19, "sales_velocity": 195},
        {"item_id": "HOUSEHOLD_2_312", "item_name": "Microfiber Cleaning Cloths", "department": "HOUSEHOLD_2", "category": "HOUSEHOLD", "base_price": 6.80, "sales_velocity": 110},
    ]
    df_catalog = pd.DataFrame(products)
    df_catalog["annual_units"] = (df_catalog["sales_velocity"] * 365 * np.random.uniform(0.9, 1.1, len(df_catalog))).astype(int)
    df_catalog["annual_revenue"] = round(df_catalog["annual_units"] * df_catalog["base_price"], 2)
    return df_catalog


df_sales = load_sales_data()
df_products = load_product_catalog()
page = st.session_state.page


# ── Overview ─────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    page_header("🏠 Project Overview",
                "Retail Demand Forecasting & Inventory Optimization — M5 Walmart Dataset")

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("3,049", "Products")
    with c2: metric_card("10",    "Stores")
    with c3: metric_card("3",     "States")
    with c4: metric_card("1,941", "Days of Data")

    st.markdown("---")
    st.markdown("### 📌 Dashboard Modules")
    a, b = st.columns(2)
    with a:
        st.info("📊 **Sales Analytics** — KPIs, trends, store & product-wise sales with interactive filters")
        st.info("🔮 **Demand Forecasting** — Prophet & LightGBM 30-day predictions & model comparisons")
    with b:
        st.info("📦 **Inventory Optimization** — Safety stock & reorder points")
        st.info("🎯 **What-If Scenario** — Price-drop impact simulation")

    st.markdown("---")
    st.markdown("### 🗓️ Build Progress")
    for label, pct in {
        "Day 1 — Setup & Structure": 100,
        "Day 2 — UI & Navigation":   100,
        "Day 3 — Sales Analytics":   100,
        "Day 4 — Filters & Product Viz": 100,
        "Day 5 — Forecasting Page":  100,
        "Day 6 — Forecast Visuals & Metrics": 100,
        "Day 7 — Inventory Dashboard": 100,
        "Day 8 — What-If Analysis":    100,
        "Day 9 — Integration & Testing": 0,
        "Day 10 — Final Dashboard":    0,
    }.items():
        st.markdown(f"**{label}**")
        st.progress(pct / 100)


# ── Sales Analytics ──────────────────────────────────────────────────────────
elif page == "📊 Sales Analytics":
    page_header("📊 Sales Analytics", "Store-wise and product-wise sales KPIs, interactive filters and rankings")

    # ── Interactive Filters ──────────────────────────────────────────────────
    st.markdown("#### 🔍 Filter Controls")
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.5, 2, 2.5])

    all_stores = ["All Stores"] + sorted(df_sales["store"].unique().tolist())
    with f_col1:
        sel_store = st.selectbox("🏬 Store", all_stores, index=0)

    with f_col2:
        all_cats = sorted(df_sales["category"].unique().tolist())
        sel_cats = st.multiselect("🏷️ Category", all_cats, default=all_cats)

    avail_depts = []
    for c in sel_cats:
        avail_depts.extend(DEPT_MAP.get(c, []))

    with f_col3:
        sel_depts = st.multiselect("📦 Department", avail_depts, default=avail_depts)

    min_date = df_sales["date"].min().to_pydatetime().date()
    max_date = df_sales["date"].max().to_pydatetime().date()
    with f_col4:
        date_range = st.date_input(
            "📅 Date Range",
            value=(pd.to_datetime("2015-01-01").date(), max_date),
            min_value=min_date,
            max_value=max_date
        )

    # Apply Filters
    df_filtered = df_sales.copy()
    if sel_store != "All Stores":
        df_filtered = df_filtered[df_filtered["store"] == sel_store]

    if sel_cats:
        df_filtered = df_filtered[df_filtered["category"].isin(sel_cats)]

    if sel_depts:
        df_filtered = df_filtered[df_filtered["department"].isin(sel_depts)]

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
        df_filtered = df_filtered[
            (df_filtered["date"].dt.date >= start_d) &
            (df_filtered["date"].dt.date <= end_d)
        ]

    st.markdown("---")

    # KPI Cards (Filtered)
    total_units   = df_filtered["units"].sum()
    total_revenue = df_filtered["revenue"].sum()
    avg_daily     = df_filtered.groupby("date")["units"].sum().mean() if not df_filtered.empty else 0
    top_store     = df_filtered.groupby("store")["units"].sum().idxmax() if not df_filtered.empty else "N/A"

    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card(f"{total_units/1e6:.2f}M" if total_units >= 1e6 else f"{total_units:,.0f}", "Total Units Sold")
    with k2: metric_card(f"${total_revenue/1e6:.2f}M" if total_revenue >= 1e6 else f"${total_revenue:,.2f}", "Total Revenue")
    with k3: metric_card(f"{avg_daily:,.0f}", "Avg Daily Units")
    with k4: metric_card(top_store, "Top Store")

    st.markdown("---")

    # Daily Sales Trend (Filtered)
    st.markdown("### 📈 Sales Trend (Filtered)")
    daily = df_filtered.groupby("date")["units"].sum().reset_index()
    if not daily.empty:
        daily["rolling_7"] = daily["units"].rolling(7).mean()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily["date"], y=daily["units"], name="Daily Units",
            line=dict(color="#adb5bd", width=1), opacity=0.5))
        fig_trend.add_trace(go.Scatter(
            x=daily["date"], y=daily["rolling_7"], name="7-Day Moving Avg",
            line=dict(color="#e94560", width=2.5)))
        fig_trend.update_layout(
            height=340, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Date", yaxis_title="Units Sold",
            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🗂️ Sales by Department")
        dept_data = df_filtered.groupby("department")["units"].sum().reset_index()
        fig_dept = px.pie(
            dept_data, names="department", values="units",
            color_discrete_sequence=px.colors.qualitative.Prism, hole=0.45)
        fig_dept.update_layout(height=320, paper_bgcolor="white")
        st.plotly_chart(fig_dept, use_container_width=True)

    with col_b:
        st.markdown("### 🏪 Sales by Store")
        store_data = (df_filtered.groupby("store")["units"].sum()
                      .reset_index().sort_values("units", ascending=True))
        fig_store = px.bar(
            store_data, x="units", y="store", orientation="h",
            color="units", color_continuous_scale=["#16213e", "#e94560"])
        fig_store.update_layout(
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Units Sold", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_store, use_container_width=True)

    st.markdown("---")

    # Product-Wise Sales Visualizations
    st.markdown("### 🏆 Product-Wise Performance & Velocity Leaderboard")
    
    prod_filter = df_products.copy()
    if sel_cats:
        prod_filter = prod_filter[prod_filter["category"].isin(sel_cats)]
    if sel_depts:
        prod_filter = prod_filter[prod_filter["department"].isin(sel_depts)]

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### Top Products by Annual Sales Volume")
        top_vol = prod_filter.sort_values("annual_units", ascending=False).head(10)
        fig_vol = px.bar(
            top_vol, x="annual_units", y="item_name", orientation="h",
            color="category", color_discrete_map={"FOODS": "#e94560", "HOUSEHOLD": "#0f3460", "HOBBIES": "#f39c12"},
            text="annual_units")
        fig_vol.update_layout(height=350, plot_bgcolor="white", yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_p2:
        st.markdown("#### Top Products by Annual Revenue ($)")
        top_rev = prod_filter.sort_values("annual_revenue", ascending=False).head(10)
        fig_rev = px.bar(
            top_rev, x="annual_revenue", y="item_name", orientation="h",
            color="category", color_discrete_map={"FOODS": "#e94560", "HOUSEHOLD": "#0f3460", "HOBBIES": "#f39c12"},
            text="annual_revenue")
        fig_rev.update_layout(height=350, plot_bgcolor="white", yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_rev, use_container_width=True)

    with st.expander("📋 View Full Product Sales Table & Export"):
        st.dataframe(prod_filter[["item_id", "item_name", "category", "department", "base_price", "annual_units", "annual_revenue"]].sort_values("annual_revenue", ascending=False), use_container_width=True)
        csv_data = prod_filter.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Product Performance CSV",
            data=csv_data,
            file_name="m5_product_performance.csv",
            mime="text/csv",
        )


# ── Demand Forecasting (Day 6 Integrated Visualizations) ──────────────────────
elif page == "🔮 Demand Forecasting":
    page_header("🔮 Demand Forecasting", "Historical sales profiling, Prophet & LightGBM 30-day forecast visualizations")

    # Forecasting Configuration Controls
    st.markdown("#### ⚙️ Forecast & Scope Configuration")
    fc1, fc2, fc3, fc4 = st.columns([1.5, 1.5, 1.5, 2])
    
    with fc1:
        sel_fc_store = st.selectbox("🏬 Target Store", ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"], index=0)
    with fc2:
        sel_horizon = st.selectbox("📅 Forecast Horizon", ["7 Days", "14 Days", "21 Days", "28 Days", "30 Days"], index=4)
    with fc3:
        sel_model = st.selectbox("🤖 Display Forecast Model", ["Both (Comparison)", "LightGBM (Tweedie GBDT)", "Facebook Prophet", "Ensemble (Blended)"], index=0)
    with fc4:
        lookback_window = st.select_slider("⏪ Historical Context Window", options=["30 Days", "60 Days", "90 Days", "180 Days", "Full History"], value="60 Days")

    st.markdown("---")

    # Load 30-Day Forecast Series
    h_days = int(sel_horizon.split()[0])
    df_fc = load_forecast_series(store_id=sel_fc_store, horizon_days=h_days)
    
    # Store Evaluation Telemetry
    store_metrics = MODEL_EVALUATION_METRICS.get(sel_fc_store, MODEL_EVALUATION_METRICS["CA_1"])
    lgb_wape = store_metrics["lightgbm"]["wape"]
    prophet_wape = store_metrics["prophet"]["wape"]
    total_actual = df_fc["actual_units"].sum()
    total_lgb = df_fc["lightgbm_forecast"].sum()
    
    # Model KPI Metric Cards
    ev1, ev2, ev3, ev4 = st.columns(4)
    with ev1: metric_card(f"{total_actual:,.0f} units", f"Actual Test Demand ({sel_horizon})")
    with ev2: metric_card(f"{lgb_wape:.2f}%", "LightGBM Error (WAPE)")
    with ev3: metric_card(f"{prophet_wape:.2f}%", "Prophet Error (WAPE)")
    champion_badge = "LightGBM 🏆" if store_metrics["lightgbm"]["winner"] else "Prophet 🏆"
    with ev4: metric_card(champion_badge, f"Store Best Model ({sel_fc_store})")

    st.markdown("---")

    # Historical Sales Context
    df_store = df_sales[df_sales["store"] == sel_fc_store].groupby("date")["units"].sum().reset_index()
    max_d = df_store["date"].max()
    window_days_map = {"30 Days": 30, "60 Days": 60, "90 Days": 90, "180 Days": 180, "Full History": 9999}
    df_hist_tail = df_store[df_store["date"] >= (max_d - pd.Timedelta(days=window_days_map[lookback_window]))]

    # Forecast Visualizations Chart (Day 6 Deliverable)
    st.markdown(f"### 📈 Predicted vs Actual Demand Trajectory ({sel_fc_store})")
    fig_pred = go.Figure()

    # Historical line
    fig_pred.add_trace(go.Scatter(
        x=df_hist_tail["date"], y=df_hist_tail["units"],
        name="Historical Actuals", mode="lines",
        line=dict(color="#adb5bd", width=1.5)))

    # Ground Truth Actual in test horizon
    fig_pred.add_trace(go.Scatter(
        x=df_fc["date"], y=df_fc["actual_units"],
        name="Out-of-Sample Actuals", mode="lines+markers",
        line=dict(color="#1a1a2e", width=2.5), marker=dict(size=4)))

    # LightGBM Forecast
    if sel_model in ["Both (Comparison)", "LightGBM (Tweedie GBDT)"]:
        fig_pred.add_trace(go.Scatter(
            x=df_fc["date"], y=df_fc["lightgbm_forecast"],
            name="LightGBM GBDT Forecast", mode="lines",
            line=dict(color="#e94560", width=2.5, dash="solid")))

    # Prophet Forecast & Uncertainty Band
    if sel_model in ["Both (Comparison)", "Facebook Prophet"]:
        fig_pred.add_trace(go.Scatter(
            x=df_fc["date"], y=df_fc["prophet_upper"],
            name="Prophet Upper Band (95% CI)",
            line=dict(color="rgba(15,52,96,0.2)", width=1, dash="dot"), showlegend=False))
        fig_pred.add_trace(go.Scatter(
            x=df_fc["date"], y=df_fc["prophet_lower"],
            name="Prophet 95% Confidence Band",
            fill="tonexty", fillcolor="rgba(15,52,96,0.12)",
            line=dict(color="rgba(15,52,96,0.2)", width=1, dash="dot")))
        fig_pred.add_trace(go.Scatter(
            x=df_fc["date"], y=df_fc["prophet_forecast"],
            name="Prophet Forecast", mode="lines",
            line=dict(color="#0f3460", width=2.5, dash="dash")))

    # Ensemble Forecast
    if sel_model == "Ensemble (Blended)":
        fig_pred.add_trace(go.Scatter(
            x=df_fc["date"], y=df_fc["ensemble_forecast"],
            name="Ensemble (50/50 Blend)", mode="lines",
            line=dict(color="#27ae60", width=3, dash="solid")))

    fig_pred.update_layout(
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Date", yaxis_title="Units Demanded",
        legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("#### 🎯 Prediction Residual Errors (LightGBM)")
        df_fc["residual"] = df_fc["lightgbm_forecast"] - df_fc["actual_units"]
        fig_res = px.bar(
            df_fc, x="date", y="residual",
            color="residual",
            color_continuous_scale=["#e94560", "#adb5bd", "#0f3460"],
            title=f"Residual Error (Predicted - Actual) for {sel_fc_store}")
        fig_res.update_layout(height=300, plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig_res, use_container_width=True)

    with res_col2:
        st.markdown("#### 📊 Cumulative Demand Volume Tracking")
        df_fc["cum_actual"] = df_fc["actual_units"].cumsum()
        df_fc["cum_lgb"] = df_fc["lightgbm_forecast"].cumsum()
        df_fc["cum_prophet"] = df_fc["prophet_forecast"].cumsum()
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=df_fc["date"], y=df_fc["cum_actual"], name="Actual Cumulative", line=dict(color="#1a1a2e", width=2)))
        fig_cum.add_trace(go.Scatter(x=df_fc["date"], y=df_fc["cum_lgb"], name="LightGBM Cumulative", line=dict(color="#e94560", width=2, dash="dash")))
        fig_cum.add_trace(go.Scatter(x=df_fc["date"], y=df_fc["cum_prophet"], name="Prophet Cumulative", line=dict(color="#0f3460", width=2, dash="dot")))
        fig_cum.update_layout(height=300, plot_bgcolor="white", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_cum, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Comprehensive Model Evaluation Matrix across All 10 Stores")
    df_metrics_summary = get_store_metrics_summary()
    st.dataframe(df_metrics_summary, use_container_width=True)


# ── Inventory Optimization (Day 7 Deliverable) ──────────────────────────────
elif page == "📦 Inventory Optimization":
    page_header("📦 Inventory Optimization & Replenishment",
                "Dynamic safety stock, reorder point (ROP), EOQ, and supplier replenishment recommendations")

    # Inventory Control Center Filters
    st.markdown("#### ⚙️ Inventory Parameters & Store Selection")
    inv1, inv2, inv3, inv4 = st.columns([1.5, 1.5, 1.5, 1.5])
    
    with inv1:
        sel_inv_store = st.selectbox("🏬 Target Store", ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"], index=0, key="inv_store")
    with inv2:
        sl_options = {0.90: "90% (Low Buffer)", 0.95: "95% (Standard Balanced)", 0.98: "98% (High Availability)", 0.99: "99% (Mission Critical)"}
        sel_sl = st.selectbox("🎯 Target Service Level", options=list(sl_options.keys()), format_func=lambda x: sl_options[x], index=1)
    with inv3:
        sel_order_cost = st.slider("📑 Order Cost / PO ($)", min_value=15.0, max_value=100.0, value=35.0, step=5.0)
    with inv4:
        sel_holding_rate = st.slider("🏷️ Annual Holding Cost Rate (%)", min_value=10.0, max_value=40.0, value=22.0, step=2.0)

    # Compute Recommendations
    df_inv = generate_inventory_recommendations(
        df_products=df_products,
        store_id=sel_inv_store,
        service_level=sel_sl,
        order_cost_override=sel_order_cost,
        holding_cost_pct_override=sel_holding_rate
    )

    # Secondary Filter for Department / Status
    flt_col1, flt_col2 = st.columns(2)
    with flt_col1:
        status_opts = ["All Statuses"] + sorted(df_inv["stock_status"].unique().tolist())
        sel_status = st.selectbox("🚥 Filter by Stock Health", status_opts, index=0)
    with flt_col2:
        inv_cat_opts = ["All Categories"] + sorted(df_inv["category"].unique().tolist())
        sel_inv_cat = st.selectbox("🏷️ Filter by Category", inv_cat_opts, index=0)

    df_inv_filtered = df_inv.copy()
    if sel_status != "All Statuses":
        df_inv_filtered = df_inv_filtered[df_inv_filtered["stock_status"] == sel_status]
    if sel_inv_cat != "All Categories":
        df_inv_filtered = df_inv_filtered[df_inv_filtered["category"] == sel_inv_cat]

    st.markdown("---")

    # High-level Inventory KPI Metrics
    total_skus = len(df_inv)
    crit_count = len(df_inv[df_inv["stock_status"] == "🔴 Critical Risk"])
    reorder_count = len(df_inv[df_inv["stock_status"] == "🟡 Reorder Needed"])
    total_val = df_inv["stock_value"].sum()
    total_reorder_cost = df_inv["est_order_cost"].sum()

    ik1, ik2, ik3, ik4 = st.columns(4)
    with ik1: metric_card(f"${total_val:,.2f}", f"On-Hand Inventory Value ({sel_inv_store})")
    with ik2: metric_card(str(crit_count), "🔴 Critical Stockout Alerts")
    with ik3: metric_card(str(reorder_count), "🟡 Replenishment Orders Due")
    with ik4: metric_card(f"${total_reorder_cost:,.2f}", "Recommended PO Spend")

    st.markdown("---")

    # Visualizations: Health Breakdown & Stock vs ROP
    v_col1, v_col2 = st.columns([1.2, 2.0])

    with v_col1:
        st.markdown("### 📊 Inventory Health Distribution")
        status_counts = df_inv["stock_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        color_discrete_map = {
            "🔴 Critical Risk": "#e94560",
            "🟡 Reorder Needed": "#f39c12",
            "🟢 Healthy": "#27ae60",
            "🔵 Overstocked": "#0f3460"
        }
        fig_donut = px.pie(
            status_counts, names="Status", values="Count",
            color="Status", color_discrete_map=color_discrete_map,
            hole=0.55
        )
        fig_donut.update_layout(height=340, margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig_donut, use_container_width=True)

    with v_col2:
        st.markdown("### 📦 Stock on Hand vs Safety Stock & ROP")
        # Bar comparison for display
        df_chart = df_inv_filtered.head(10).sort_values("current_stock", ascending=False)
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            name="Current Stock", x=df_chart["item_name"], y=df_chart["current_stock"],
            marker_color="#0f3460"))
        fig_bars.add_trace(go.Bar(
            name="Reorder Point (ROP)", x=df_chart["item_name"], y=df_chart["reorder_point"],
            marker_color="#f39c12"))
        fig_bars.add_trace(go.Bar(
            name="Safety Stock (SS)", x=df_chart["item_name"], y=df_chart["safety_stock"],
            marker_color="#e94560"))
        fig_bars.update_layout(
            barmode="group", height=340, plot_bgcolor="white",
            xaxis_tickangle=-25, legend=dict(orientation="h", y=1.12),
            yaxis_title="Units"
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    st.markdown("---")

    # Comprehensive Reorder Recommendations Table
    st.markdown("### 📋 Automated Reorder Recommendations & Purchase Orders")
    display_cols = [
        "item_id", "item_name", "category", "unit_price", "current_stock",
        "daily_demand", "lead_time_days", "safety_stock", "reorder_point",
        "stock_status", "reorder_units", "est_order_cost"
    ]
    st.dataframe(df_inv_filtered[display_cols].sort_values("reorder_units", ascending=False), use_container_width=True)

    # Export Purchase Order CSV
    po_csv = df_inv_filtered[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Replenishment Purchase Order (CSV)",
        data=po_csv,
        file_name=f"m5_replenishment_po_{sel_inv_store}.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Deep Dive SKU Inspector
    st.markdown("### 🔬 SKU Inventory Replenishment Deep-Dive")
    sku_list = df_inv["item_name"].tolist()
    sel_sku_name = st.selectbox("Select SKU to Inspect", sku_list, index=0)
    sku_row = df_inv[df_inv["item_name"] == sel_sku_name].iloc[0]

    d1, d2, d3, d4, d5 = st.columns(5)
    with d1: st.metric("Current Stock", f"{sku_row['current_stock']} units")
    with d2: st.metric("Safety Stock (SS)", f"{sku_row['safety_stock']} units")
    with d3: st.metric("Reorder Point (ROP)", f"{sku_row['reorder_point']} units")
    with d4: st.metric("Economic Order Qty (EOQ)", f"{sku_row['eoq']} units")
    with d5: st.metric("Days of Supply", f"{sku_row['days_of_supply']} days")

    # Simulated Sawtooth Inventory Cycle Diagram
    sim_days = 45
    timeline = pd.date_range("2016-04-25", periods=sim_days, freq="D")
    stock_traj = []
    c_stock = sku_row["current_stock"]
    rate = max(1, int(round(sku_row["daily_demand"])))
    reorder_placed_day = None
    lt = sku_row["lead_time_days"]
    order_qty = sku_row["eoq"]

    for d_idx in range(sim_days):
        stock_traj.append(c_stock)
        c_stock = max(0, c_stock - rate)
        if c_stock <= sku_row["reorder_point"] and reorder_placed_day is None:
            reorder_placed_day = d_idx
        if reorder_placed_day is not None and (d_idx - reorder_placed_day) == lt:
            c_stock += order_qty
            reorder_placed_day = None

    fig_sawtooth = go.Figure()
    fig_sawtooth.add_trace(go.Scatter(
        x=timeline, y=stock_traj, name="Projected On-Hand Stock",
        line=dict(color="#0f3460", width=2.5)))
    fig_sawtooth.add_hline(
        y=sku_row["reorder_point"], line_dash="dash", line_color="#f39c12",
        annotation_text="Reorder Point (ROP)", annotation_position="top left")
    fig_sawtooth.add_hline(
        y=sku_row["safety_stock"], line_dash="dot", line_color="#e94560",
        annotation_text="Safety Stock (Buffer)", annotation_position="bottom left")
    fig_sawtooth.update_layout(
        title=f"Replenishment Cycle (Sawtooth Curve) for {sku_row['item_name']}",
        height=320, plot_bgcolor="white", xaxis_title="Date", yaxis_title="Units in Stock"
    )
    st.plotly_chart(fig_sawtooth, use_container_width=True)

# ── What-If Scenario Analysis (Day 8 Deliverable) ───────────────────────────
elif page == "🎯 What-If Scenario":
    page_header("🎯 What-If Scenario & Price Elasticity Simulation",
                "Simulate demand surges, revenue trade-offs, and supply-chain buffer requirements under price adjustments")

    # Quick Scenario Preset Bar
    st.markdown("#### ⚡ Quick Simulation Presets")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    if "price_slider_val" not in st.session_state:
        st.session_state.price_slider_val = -10.0

    with q_col1:
        if st.button("🏷️ 10% Price Reduction (Promo)", use_container_width=True):
            st.session_state.price_slider_val = -10.0
            st.rerun()
    with q_col2:
        if st.button("🔥 20% Clearance Sale", use_container_width=True):
            st.session_state.price_slider_val = -20.0
            st.rerun()
    with q_col3:
        if st.button("📈 5% Inflation Adjustment", use_container_width=True):
            st.session_state.price_slider_val = 5.0
            st.rerun()
    with q_col4:
        if st.button("🔄 Reset to Baseline (0%)", use_container_width=True):
            st.session_state.price_slider_val = 0.0
            st.rerun()

    st.markdown("---")

    # Simulation Controls
    st.markdown("#### ⚙️ Scenario Configuration Parameters")
    sc1, sc2, sc3, sc4 = st.columns([1.5, 2.0, 1.5, 1.5])

    with sc1:
        sel_sc_store = st.selectbox("🏬 Target Store", ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"], index=0, key="sc_store")
    with sc2:
        price_drop = st.slider(
            "💰 Price Adjustment (%)",
            min_value=-30.0, max_value=20.0,
            value=float(st.session_state.price_slider_val),
            step=1.0, format="%.1f%%"
        )
    with sc3:
        sel_regime = st.selectbox("📊 Elasticity Sensitivity Regime", ["Moderate", "Conservative", "Aggressive"], index=0)
    with sc4:
        sel_sc_cat = st.selectbox("🏷️ Category Focus", ["All Categories", "FOODS", "HOUSEHOLD", "HOBBIES"], index=0)

    # Generate baseline forecast series and simulate price change
    base_fc_30 = load_forecast_series(store_id=sel_sc_store, horizon_days=30)
    sim_result = simulate_price_scenario(
        base_forecast_df=base_fc_30,
        price_change_pct=price_drop,
        elasticity_regime=sel_regime,
        category_focus=sel_sc_cat
    )

    summary = sim_result["summary"]
    df_sim_traj = sim_result["trajectory_df"]
    df_cat_matrix = sim_result["category_breakdown"]

    st.markdown("---")

    # High-level Scenario Impact Cards
    k_sc1, k_sc2, k_sc3, k_sc4 = st.columns(4)
    with k_sc1:
        metric_card(f"{summary['price_change_pct']:+.1f}% (${summary['simulated_price']:.2f})", "Simulated Price Shift")
    with k_sc2:
        lift_sign = "+" if summary['pct_unit_lift'] >= 0 else ""
        metric_card(f"{lift_sign}{summary['pct_unit_lift']:.1f}% ({summary['unit_lift_total']:+,.0f} units)", "Projected Demand Lift")
    with k_sc3:
        rev_sign = "+" if summary['rev_delta'] >= 0 else ""
        metric_card(f"{rev_sign}${summary['rev_delta']:,.2f} ({rev_sign}{summary['pct_rev_delta']:.1f}%)", "Total Revenue Impact")
    with k_sc4:
        m_sign = "+" if summary['margin_delta'] >= 0 else ""
        metric_card(f"{m_sign}${summary['margin_delta']:,.2f} ({m_sign}{summary['pct_margin_delta']:.1f}%)", "Net Gross Margin Impact")

    st.markdown("---")

    # Main Visual: Baseline Forecast vs Price-Adjusted Forecast
    st.markdown(f"### 📈 30-Day Demand Trajectory: Baseline vs {summary['price_change_pct']:+.1f}% Price Shift")
    fig_sim = go.Figure()

    # Baseline LightGBM Forecast
    fig_sim.add_trace(go.Scatter(
        x=df_sim_traj["date"], y=df_sim_traj["base_forecast"],
        name="Baseline Demand Forecast", mode="lines",
        line=dict(color="#1a1a2e", width=2.5, dash="dash")
    ))

    # Simulated Trajectory
    sim_line_color = "#e94560" if price_drop < 0 else "#27ae60"
    fig_sim.add_trace(go.Scatter(
        x=df_sim_traj["date"], y=df_sim_traj["simulated_demand"],
        name=f"Price Adjusted Demand ({price_drop:+.1f}%)", mode="lines+markers",
        line=dict(color=sim_line_color, width=3),
        marker=dict(size=4)
    ))

    # Shaded Uplift Region
    fig_sim.add_trace(go.Scatter(
        x=df_sim_traj["date"], y=df_sim_traj["simulated_demand"],
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig_sim.add_trace(go.Scatter(
        x=df_sim_traj["date"], y=df_sim_traj["base_forecast"],
        fill="tonexty",
        fillcolor="rgba(233, 69, 96, 0.15)" if price_drop < 0 else "rgba(39, 174, 96, 0.15)",
        line=dict(width=0),
        name="Demand Volume Shift (Uplift / Drop)",
        hoverinfo="skip"
    ))

    fig_sim.update_layout(
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Date", yaxis_title="Daily Units Demanded",
        legend=dict(orientation="h", y=1.12, x=0)
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown("---")

    # Daily Financial & Volume Impact Visualizations
    col_fin1, col_fin2 = st.columns(2)

    with col_fin1:
        st.markdown("#### 💵 Daily Revenue Comparison (Base vs Promo)")
        fig_rev_cmp = go.Figure()
        fig_rev_cmp.add_trace(go.Bar(
            x=df_sim_traj["date"], y=df_sim_traj["base_revenue"],
            name="Baseline Revenue ($)", marker_color="#0f3460"
        ))
        fig_rev_cmp.add_trace(go.Bar(
            x=df_sim_traj["date"], y=df_sim_traj["simulated_revenue"],
            name="Simulated Revenue ($)", marker_color="#e94560" if price_drop < 0 else "#27ae60"
        ))
        fig_rev_cmp.update_layout(
            barmode="group", height=320, plot_bgcolor="white",
            legend=dict(orientation="h", y=1.15), yaxis_title="Revenue ($)"
        )
        st.plotly_chart(fig_rev_cmp, use_container_width=True)

    with col_fin2:
        st.markdown("#### 📦 Cumulative Unit Volume Lift")
        df_sim_traj["cum_lift"] = df_sim_traj["unit_lift"].cumsum()
        fig_cum_lift = go.Figure()
        fig_cum_lift.add_trace(go.Scatter(
            x=df_sim_traj["date"], y=df_sim_traj["cum_lift"],
            mode="lines+markers", name="Cumulative Additional Units",
            line=dict(color="#e94560", width=2.5),
            fill="tozeroy", fillcolor="rgba(233, 69, 96, 0.1)"
        ))
        fig_cum_lift.update_layout(
            height=320, plot_bgcolor="white",
            xaxis_title="Date", yaxis_title="Net Cumulative Units",
            legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig_cum_lift, use_container_width=True)

    st.markdown("---")

    # Category Price Elasticity Sensitivity Matrix
    st.markdown("### 🗂️ Category Elasticity Sensitivity Breakdown")
    st.dataframe(df_cat_matrix, use_container_width=True)

    # Supply Chain Warning Callout
    if price_drop < 0:
        st.warning(f"""
        ⚠️ **Supply Chain Stockout Prevention Notice:**  
        A **{abs(price_drop):.1f}% price reduction** produces a projected **+{summary['pct_unit_lift']:.1f}% surge** in consumer demand (+{summary['unit_lift_total']:,} units over 30 days).  
        To prevent stockouts and service level degradation, procurement must place an **immediate replenishment buffer of at least {summary['extra_buffer_units_needed']:,} additional units** across suppliers before promotional launch.
        """)
    else:
        st.info(f"""
        ℹ️ **Inventory Optimization Notice:**  
        A **+{price_drop:.1f}% price increase** results in a projected **{summary['pct_unit_lift']:.1f}% reduction** in demand velocity.  
        Supplier reorder cycles should be adjusted downward to avoid excess inventory accumulation.
        """)
