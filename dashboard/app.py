import streamlit as st

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Main background */
    .main { background-color: #f8f9fa; }

    /* Nav buttons */
    .nav-btn {
        display: block;
        width: 100%;
        padding: 10px 16px;
        margin: 4px 0;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 8px;
        color: #e0e0e0;
        font-size: 14px;
        cursor: pointer;
        text-align: left;
        transition: background 0.2s;
    }
    .nav-btn:hover { background: rgba(255,255,255,0.18); }
    .nav-btn.active {
        background: linear-gradient(90deg, #e94560, #0f3460);
        border-color: #e94560;
        font-weight: bold;
    }

    /* Page header */
    .page-header {
        background: linear-gradient(90deg, #1a1a2e, #0f3460);
        padding: 20px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        color: white;
    }
    .page-header h1 { margin: 0; font-size: 26px; color: white; }
    .page-header p  { margin: 4px 0 0; font-size: 14px; color: #adb5bd; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #e94560;
        text-align: center;
    }

    /* Hide default Streamlit menu */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state for navigation ────────────────────────────────────────────
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

    for page in PAGES:
        is_active = st.session_state.page == page
        btn_class = "nav-btn active" if is_active else "nav-btn"
        if st.button(page, key=f"nav_{page}", use_container_width=True):
            st.session_state.page = page
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
    st.caption("v1.0 · Day 2 Build")


# ── Page Router ──────────────────────────────────────────────────────────────
page = st.session_state.page


def page_header(title: str, subtitle: str):
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ── Overview Page ────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    page_header("🏠 Project Overview", "Retail Demand Forecasting & Inventory Optimization — M5 Walmart Dataset")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>3,049</h2><p>Products</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>10</h2><p>Stores</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>3</h2><p>States</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>1,941</h2><p>Days of Data</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 Dashboard Modules")

    c1, c2 = st.columns(2)
    with c1:
        st.info("📊 **Sales Analytics** — KPIs, trends, store & product-wise sales")
        st.info("🔮 **Demand Forecasting** — Prophet & LightGBM 30-day predictions")
    with c2:
        st.info("📦 **Inventory Optimization** — Safety stock & reorder points")
        st.info("🎯 **What-If Scenario** — Price-drop impact simulation")

    st.markdown("---")
    st.markdown("### 🗓️ Build Progress")
    progress_data = {
        "Day 1 — Setup & Structure": 100,
        "Day 2 — UI & Navigation": 100,
        "Day 3 — Sales Analytics": 0,
        "Day 4 — Filters": 0,
        "Day 5 — Forecasting Page": 0,
        "Day 6 — Forecast Visuals": 0,
        "Day 7 — Inventory Dashboard": 0,
        "Day 8 — What-If Analysis": 0,
        "Day 9 — Integration & Testing": 0,
        "Day 10 — Final Dashboard": 0,
    }
    for label, pct in progress_data.items():
        st.markdown(f"**{label}**")
        st.progress(pct / 100)


# ── Placeholder Pages ────────────────────────────────────────────────────────
elif page == "📊 Sales Analytics":
    page_header("📊 Sales Analytics", "Store-wise and product-wise sales KPIs and trends")
    st.info("🔧 Coming on Day 3 — Sales KPI cards and trend charts will be added here.")

elif page == "🔮 Demand Forecasting":
    page_header("🔮 Demand Forecasting", "Prophet & LightGBM 30-day demand predictions")
    st.info("🔧 Coming on Day 5 & 6 — Forecasting visualizations will be added here.")

elif page == "📦 Inventory Optimization":
    page_header("📦 Inventory Optimization", "Safety stock levels and reorder point recommendations")
    st.info("🔧 Coming on Day 7 — Inventory optimization dashboard will be added here.")

elif page == "🎯 What-If Scenario":
    page_header("🎯 What-If Scenario Analysis", "Simulate price-drop impact on demand forecasts")
    st.info("🔧 Coming on Day 8 — Price-drop scenario analysis will be added here.")
