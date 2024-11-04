import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

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
    st.caption("v1.0 · Day 4 Build")


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
        "Day 4 — Filters & Product Viz": 100,
        "Day 5 — Forecasting Page":    0,
        "Day 6 — Forecast Visuals":    0,
        "Day 7 — Inventory Dashboard": 0,
        "Day 8 — What-If Analysis":    0,
        "Day 9 — Integration & Testing":0,
        "Day 10 — Final Dashboard":    0,
    }.items():
        st.markdown(f"**{label}**")
        st.progress(pct / 100)


# ── Sales Analytics (with Day 4 Interactive Filters) ─────────────────────────
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

    # Dynamic department filter based on category selection
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

    # Sales by Department Breakdown
    with col_a:
        st.markdown("### 🗂️ Sales by Department")
        dept_data = df_filtered.groupby("department")["units"].sum().reset_index()
        fig_dept = px.pie(
            dept_data, names="department", values="units",
            color_discrete_sequence=px.colors.qualitative.Prism, hole=0.45)
        fig_dept.update_layout(height=320, paper_bgcolor="white")
        st.plotly_chart(fig_dept, use_container_width=True)

    # Sales by Store Comparison
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

    # ── Product-Wise Sales Visualizations (Day 4 Deliverable) ────────────────
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

    # Detailed Table with Download
    with st.expander("📋 View Full Product Sales Table & Export"):
        st.dataframe(prod_filter[["item_id", "item_name", "category", "department", "base_price", "annual_units", "annual_revenue"]].sort_values("annual_revenue", ascending=False), use_container_width=True)
        csv_data = prod_filter.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Product Performance CSV",
            data=csv_data,
            file_name="m5_product_performance.csv",
            mime="text/csv",
        )


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
