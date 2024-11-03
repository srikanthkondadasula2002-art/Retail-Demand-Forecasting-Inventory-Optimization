import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    st.caption("v1.0 · Day 3 Build")


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


# ── Simulated M5 Sales Data ──────────────────────────────────────────────────
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
    rows = []
    for d in dates:
        dow_boost = 1.38 if d.dayofweek >= 5 else 1.0
        for s in stores:
            for c in categories:
                units   = int(base[s] * cat_split[c] * dow_boost * np.random.uniform(0.85, 1.15))
                revenue = round(units * np.random.uniform(2.5, 8.5), 2)
                rows.append({"date": d, "store": s, "category": c,
                              "units": units, "revenue": revenue})
    return pd.DataFrame(rows)


df_sales = load_sales_data()
page     = st.session_state.page


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
        st.info("📊 **Sales Analytics** — KPIs, trends, store & product-wise sales")
        st.info("🔮 **Demand Forecasting** — Prophet & LightGBM 30-day predictions")
    with b:
        st.info("📦 **Inventory Optimization** — Safety stock & reorder points")
        st.info("🎯 **What-If Scenario** — Price-drop impact simulation")

    st.markdown("---")
    st.markdown("### 🗓️ Build Progress")
    for label, pct in {
        "Day 1 — Setup & Structure": 100,
        "Day 2 — UI & Navigation":   100,
        "Day 3 — Sales Analytics":   100,
        "Day 4 — Filters":             0,
        "Day 5 — Forecasting Page":    0,
        "Day 6 — Forecast Visuals":    0,
        "Day 7 — Inventory Dashboard": 0,
        "Day 8 — What-If Analysis":    0,
        "Day 9 — Integration & Testing":0,
        "Day 10 — Final Dashboard":    0,
    }.items():
        st.markdown(f"**{label}**")
        st.progress(pct / 100)


# ── Sales Analytics ───────────────────────────────────────────────────────────
elif page == "📊 Sales Analytics":
    page_header("📊 Sales Analytics", "Store-wise and product-wise sales KPIs and trends")

    # KPI Cards
    total_units   = df_sales["units"].sum()
    total_revenue = df_sales["revenue"].sum()
    avg_daily     = df_sales.groupby("date")["units"].sum().mean()
    top_store     = df_sales.groupby("store")["units"].sum().idxmax()

    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card(f"{total_units/1e6:.1f}M",  "Total Units Sold")
    with k2: metric_card(f"${total_revenue/1e6:.1f}M", "Total Revenue")
    with k3: metric_card(f"{avg_daily:,.0f}",         "Avg Daily Units")
    with k4: metric_card(top_store,                   "Top Store")

    st.markdown("---")

    # Daily Sales Trend
    st.markdown("### 📈 Daily Sales Trend")
    daily = df_sales.groupby("date")["units"].sum().reset_index()
    daily["rolling_7"] = daily["units"].rolling(7).mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily["date"], y=daily["units"], name="Daily Units",
        line=dict(color="#adb5bd", width=1), opacity=0.5))
    fig_trend.add_trace(go.Scatter(
        x=daily["date"], y=daily["rolling_7"], name="7-Day MA",
        line=dict(color="#e94560", width=2)))
    fig_trend.update_layout(
        height=350, plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Date", yaxis_title="Units Sold",
        legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Sales by Category
    with col_a:
        st.markdown("### 🗂️ Sales by Category")
        cat_data = df_sales.groupby("category")["units"].sum().reset_index()
        fig_cat = px.pie(
            cat_data, names="category", values="units",
            color_discrete_sequence=["#e94560", "#0f3460", "#16213e"], hole=0.4)
        fig_cat.update_layout(height=320, paper_bgcolor="white")
        st.plotly_chart(fig_cat, use_container_width=True)

    # Sales by Store
    with col_b:
        st.markdown("### 🏪 Sales by Store")
        store_data = (df_sales.groupby("store")["units"].sum()
                      .reset_index().sort_values("units", ascending=True))
        fig_store = px.bar(
            store_data, x="units", y="store", orientation="h",
            color="units", color_continuous_scale=["#16213e", "#e94560"])
        fig_store.update_layout(
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Units Sold", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_store, use_container_width=True)

    # Monthly Revenue Trend
    st.markdown("### 📅 Monthly Revenue Trend")
    df_sales["month"] = df_sales["date"].dt.to_period("M").astype(str)
    monthly = df_sales.groupby(["month", "category"])["revenue"].sum().reset_index()
    fig_monthly = px.bar(
        monthly, x="month", y="revenue", color="category",
        color_discrete_sequence=["#e94560", "#0f3460", "#16213e"], barmode="stack")
    fig_monthly.update_layout(
        height=350, plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Month", yaxis_title="Revenue ($)",
        legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_monthly, use_container_width=True)


# ── Placeholder Pages ─────────────────────────────────────────────────────────
elif page == "🔮 Demand Forecasting":
    page_header("🔮 Demand Forecasting", "Prophet & LightGBM 30-day demand predictions")
    st.info("🔧 Coming on Day 5 & 6 — Forecasting visualizations will be added here.")

elif page == "📦 Inventory Optimization":
    page_header("📦 Inventory Optimization", "Safety stock levels and reorder point recommendations")
    st.info("🔧 Coming on Day 7 — Inventory optimization dashboard will be added here.")

elif page == "🎯 What-If Scenario":
    page_header("🎯 What-If Scenario Analysis", "Simulate price-drop impact on demand forecasts")
    st.info("🔧 Coming on Day 8 — Price-drop scenario analysis will be added here.")
