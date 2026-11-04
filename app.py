"""
Retail Demand Forecasting & Inventory Optimization - Sales Analytics Dashboard
Features:
- Interactive Sales KPI cards with period comparisons
- Longitudinal sales trend charts with toggleable rolling averages (7D / 30D / Raw)
- Store-by-store sales comparison trendlines
- Category & department composition area charts
- Intra-week day-of-week seasonality analysis
- SNAP policy stimulus impact comparison
- Top product volume & revenue leaderboards with CSV export
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Retail Sales Analytics Dashboard | Walmart M5",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished KPI metric cards and clean layout
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.82rem;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .sub-green { color: #16a34a; font-weight: 600; }
    .sub-muted { color: #64748b; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

@st.cache_data(show_spinner="Loading precomputed sales telemetry...")
def load_data():
    dept_path = os.path.join(PROCESSED_DIR, "store_dept_daily.parquet")
    prod_path = os.path.join(PROCESSED_DIR, "product_performance.parquet")
    
    # Auto-generate if missing
    if not os.path.exists(dept_path) or not os.path.exists(prod_path):
        from src.prepare_dashboard_data import prepare_all
        prepare_all()
        
    df_daily = pd.read_parquet(dept_path)
    df_prod = pd.read_parquet(prod_path)
    return df_daily, df_prod

df_daily, df_prod = load_data()

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/ca/Walmart_logo.svg", width=180)
st.sidebar.title("Filters & Controls")

# Date range selection
min_date = df_daily['date'].min().date()
max_date = df_daily['date'].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# State filter
all_states = ["All"] + sorted(df_daily['state_id'].unique().tolist())
selected_state = st.sidebar.selectbox("State", all_states, index=0)

# Store filter (dynamically filtered by state)
if selected_state != "All":
    available_stores = sorted(df_daily[df_daily['state_id'] == selected_state]['store_id'].unique().tolist())
else:
    available_stores = sorted(df_daily['store_id'].unique().tolist())
all_stores = ["All"] + available_stores
selected_store = st.sidebar.selectbox("Store", all_stores, index=0)

# Category filter
all_cats = ["All"] + sorted(df_daily['cat_id'].unique().tolist())
selected_cat = st.sidebar.selectbox("Category", all_cats, index=0)

# Department filter (dynamically filtered by category)
if selected_cat != "All":
    available_depts = sorted(df_daily[df_daily['cat_id'] == selected_cat]['dept_id'].unique().tolist())
else:
    available_depts = sorted(df_daily['dept_id'].unique().tolist())
all_depts = ["All"] + available_depts
selected_dept = st.sidebar.selectbox("Department", all_depts, index=0)

# Trend Smoothing Options
smoothing_option = st.sidebar.radio(
    "Trend Smoothing Window",
    ["7-Day Moving Average", "30-Day Moving Average", "Raw Daily"],
    index=0
)

# Apply filters to daily time series
filtered_daily = df_daily[
    (df_daily['date'].dt.date >= start_date) & 
    (df_daily['date'].dt.date <= end_date)
]
if selected_state != "All":
    filtered_daily = filtered_daily[filtered_daily['state_id'] == selected_state]
if selected_store != "All":
    filtered_daily = filtered_daily[filtered_daily['store_id'] == selected_store]
if selected_cat != "All":
    filtered_daily = filtered_daily[filtered_daily['cat_id'] == selected_cat]
if selected_dept != "All":
    filtered_daily = filtered_daily[filtered_daily['dept_id'] == selected_dept]

# Apply filters to product performance
filtered_prod = df_prod.copy()
if selected_state != "All":
    filtered_prod = filtered_prod[filtered_prod['state_id'] == selected_state]
if selected_store != "All":
    filtered_prod = filtered_prod[filtered_prod['store_id'] == selected_store]
if selected_cat != "All":
    filtered_prod = filtered_prod[filtered_prod['cat_id'] == selected_cat]
if selected_dept != "All":
    filtered_prod = filtered_prod[filtered_prod['dept_id'] == selected_dept]

# ==========================================
# MAIN HEADER
# ==========================================
st.title("🛒 Walmart Retail Sales Analytics Dashboard")
st.markdown("Interactive performance monitoring, demand trend discovery, and seasonal diagnostics based on the M5 dataset.")

# ==========================================
# KPI METRICS CARDS
# ==========================================
total_units = int(filtered_daily['units'].sum())
num_days = max(1, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)
avg_daily_sales = total_units / num_days

# Approximate revenue from filtered products
total_revenue = float(filtered_prod['est_revenue'].sum())
avg_item_price = float(filtered_prod['avg_sell_price'].mean()) if len(filtered_prod) > 0 else 0.0

# Active SKUs and Sparsity
active_skus = int(filtered_prod['item_id'].nunique())
overall_sparsity = float(filtered_prod['sparsity_pct'].mean()) if len(filtered_prod) > 0 else 68.2

# Weekend surge calculation
daily_ts = filtered_daily.groupby('date')['units'].sum().reset_index()
daily_ts['weekday'] = daily_ts['date'].dt.day_name()
weekend_mask = daily_ts['weekday'].isin(['Saturday', 'Sunday'])
weekend_avg = daily_ts[weekend_mask]['units'].mean() if weekend_mask.any() else 0
weekday_avg = daily_ts[~weekend_mask]['units'].mean() if (~weekend_mask).any() else 1
weekend_lift = ((weekend_avg - weekday_avg) / weekday_avg) * 100 if weekday_avg > 0 else 0.0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Units Sold</div>
        <div class="metric-value">{total_units:,.0f}</div>
        <div class="metric-sub sub-muted">Across {num_days} selected days</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Est. Gross Revenue</div>
        <div class="metric-value">${total_revenue:,.0f}</div>
        <div class="metric-sub sub-muted">Avg Item Price: ${avg_item_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Daily Velocity</div>
        <div class="metric-value">{avg_daily_sales:,.0f}</div>
        <div class="metric-sub sub-muted">Units sold / day</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Catalog</div>
        <div class="metric-value">{active_skus:,}</div>
        <div class="metric-sub sub-muted">Distinct SKUs analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Weekend Surge</div>
        <div class="metric-value">+{weekend_lift:.1f}%</div>
        <div class="metric-sub sub-green">Lift over weekday rate</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# TABS FOR DEEP-DIVE ANALYTICS
# ==========================================
tab_trends, tab_stores, tab_categories, tab_products = st.tabs([
    "📈 Longitudinal Sales Trends",
    "🏪 Store Performance Benchmark",
    "📦 Category & Department Mix",
    "🏆 Top Products & Leaderboard"
])

# ------------------------------------------
# TAB 1: LONGITUDINAL SALES TRENDS
# ------------------------------------------
with tab_trends:
    st.subheader("Historical Demand Trajectory")
    
    # Aggregated time series based on filters
    trend_df = filtered_daily.groupby('date')['units'].sum().reset_index().sort_values('date')
    
    # Apply rolling smoothing
    if smoothing_option == "7-Day Moving Average":
        trend_df['smoothed'] = trend_df['units'].rolling(7, min_periods=1).mean()
        y_col = 'smoothed'
        y_name = "7-Day Rolling Avg"
    elif smoothing_option == "30-Day Moving Average":
        trend_df['smoothed'] = trend_df['units'].rolling(30, min_periods=1).mean()
        y_col = 'smoothed'
        y_name = "30-Day Rolling Avg"
    else:
        trend_df['smoothed'] = trend_df['units']
        y_col = 'smoothed'
        y_name = "Raw Daily Units"

    fig_trend = go.Figure()
    
    # Raw subtle line if smoothed
    if smoothing_option != "Raw Daily":
        fig_trend.add_trace(go.Scatter(
            x=trend_df['date'],
            y=trend_df['units'],
            mode='lines',
            name='Daily Actuals',
            line=dict(color='rgba(148, 163, 184, 0.4)', width=1),
            hoverinfo='skip'
        ))
        
    fig_trend.add_trace(go.Scatter(
        x=trend_df['date'],
        y=trend_df[y_col],
        mode='lines',
        name=y_name,
        line=dict(color='#2563eb', width=2.5),
        hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Units</b>: %{y:,.0f}<extra></extra>"
    ))

    fig_trend.update_layout(
        title=f"Sales Trajectory: {selected_state} / {selected_store} ({smoothing_option})",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Secondary Seasonal Diagnostics: Day-of-Week & SNAP Impact
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📅 Intra-Week Demand Rhythm")
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_df = trend_df.copy()
        dow_df['weekday'] = dow_df['date'].dt.day_name()
        dow_avg = dow_df.groupby('weekday')['units'].mean().reindex(dow_order).reset_index()
        
        fig_dow = px.bar(
            dow_avg,
            x='weekday',
            y='units',
            labels={'weekday': 'Day of Week', 'units': 'Avg Daily Units'},
            title="Average Daily Units by Day of Week",
            color='units',
            color_continuous_scale="Blues"
        )
        fig_dow.update_layout(height=350, showlegend=False, template="plotly_white")
        st.plotly_chart(fig_dow, use_container_width=True)

    with c2:
        st.subheader("🏛️ SNAP Policy Stimulus Lift")
        # SNAP analysis for the filtered subset
        snap_df = filtered_daily.groupby(['date', 'state_id'])['units'].sum().reset_index()
        cal_sub = df_daily[['date', 'snap_CA', 'snap_TX', 'snap_WI']].drop_duplicates()
        snap_df = snap_df.merge(cal_sub, on='date', how='left')
        
        snap_rows = []
        for st_code in ['CA', 'TX', 'WI']:
            if selected_state != "All" and selected_state != st_code:
                continue
            sub_st = snap_df[snap_df['state_id'] == st_code]
            if len(sub_st) > 0:
                is_snap = sub_st[f'snap_{st_code}'] == 1
                snap_mean = sub_st[is_snap]['units'].mean() if is_snap.any() else 0
                nonsnap_mean = sub_st[~is_snap]['units'].mean() if (~is_snap).any() else 0
                lift = ((snap_mean - nonsnap_mean) / nonsnap_mean) * 100 if nonsnap_mean > 0 else 0
                snap_rows.append({'State': st_code, 'Day Type': 'SNAP Days', 'Avg Sales': snap_mean, 'Lift': lift})
                snap_rows.append({'State': st_code, 'Day Type': 'Non-SNAP Days', 'Avg Sales': nonsnap_mean, 'Lift': lift})
                
        if snap_rows:
            fig_snap = px.bar(
                pd.DataFrame(snap_rows),
                x='State',
                y='Avg Sales',
                color='Day Type',
                barmode='group',
                title="SNAP vs Non-SNAP Average Daily Sales",
                color_discrete_map={'SNAP Days': '#1e40af', 'Non-SNAP Days': '#94a3b8'}
            )
            fig_snap.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig_snap, use_container_width=True)
        else:
            st.info("SNAP comparison not applicable for current filter selection.")

# ------------------------------------------
# TAB 2: STORE BENCHMARKING
# ------------------------------------------
with tab_stores:
    st.subheader("Store Sales Volume & Revenue Comparison")
    
    # Store-by-store totals
    store_totals = df_daily.groupby(['state_id', 'store_id'])['units'].sum().reset_index()
    store_rev = df_prod.groupby('store_id')['est_revenue'].sum().reset_index()
    store_totals = store_totals.merge(store_rev, on='store_id', how='left').sort_values('units', ascending=False)
    
    col_s1, col_s2 = st.columns([3, 2])
    with col_s1:
        # Multi-line comparative store trajectory
        st.markdown("**Longitudinal Store Trajectories (30-Day Rolling Average)**")
        store_ts = df_daily.groupby(['date', 'store_id'])['units'].sum().unstack()
        store_ts_smooth = store_ts.rolling(30, min_periods=7).mean().reset_index()
        store_melt = pd.melt(store_ts_smooth, id_vars=['date'], value_name='units', var_name='store_id')
        
        # Filter dates
        store_melt = store_melt[
            (store_melt['date'].dt.date >= start_date) & 
            (store_melt['date'].dt.date <= end_date)
        ]
        
        fig_store_lines = px.line(
            store_melt,
            x='date',
            y='units',
            color='store_id',
            title="Store Comparison Over Time",
            labels={'units': '30D Smoothed Sales', 'date': 'Date', 'store_id': 'Store'},
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_store_lines.update_layout(height=450, template="plotly_white")
        st.plotly_chart(fig_store_lines, use_container_width=True)

    with col_s2:
        st.markdown("**Store Performance Ranking**")
        fig_store_bar = px.bar(
            store_totals,
            x='units',
            y='store_id',
            orientation='h',
            color='state_id',
            title="Total Units Sold by Store",
            labels={'units': 'Units Sold', 'store_id': 'Store ID', 'state_id': 'State'},
            color_discrete_map={'CA': '#2563eb', 'TX': '#d97706', 'WI': '#059669'}
        )
        fig_store_bar.update_layout(height=450, yaxis={'categoryorder':'total ascending'}, template="plotly_white")
        st.plotly_chart(fig_store_bar, use_container_width=True)

# ------------------------------------------
# TAB 3: CATEGORY & DEPARTMENT MIX
# ------------------------------------------
with tab_categories:
    st.subheader("Product Category & Department Demand Breakdown")
    
    cat_daily = filtered_daily.groupby(['date', 'cat_id'])['units'].sum().reset_index()
    
    c_cat1, c_cat2 = st.columns(2)
    with c_cat1:
        fig_cat_area = px.area(
            cat_daily,
            x='date',
            y='units',
            color='cat_id',
            title="Sales Volume Composition Over Time",
            labels={'units': 'Units Sold', 'date': 'Date', 'cat_id': 'Category'},
            color_discrete_sequence=['#2563eb', '#10b981', '#f59e0b']
        )
        fig_cat_area.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_cat_area, use_container_width=True)

    with c_cat2:
        dept_totals = filtered_daily.groupby(['cat_id', 'dept_id'])['units'].sum().reset_index().sort_values('units', ascending=False)
        fig_dept_bar = px.bar(
            dept_totals,
            x='dept_id',
            y='units',
            color='cat_id',
            title="Total Sales by Department",
            labels={'units': 'Units Sold', 'dept_id': 'Department', 'cat_id': 'Category'},
            color_discrete_sequence=['#2563eb', '#10b981', '#f59e0b']
        )
        fig_dept_bar.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_dept_bar, use_container_width=True)

# ------------------------------------------
# TAB 4: TOP PRODUCTS LEADERBOARD
# ------------------------------------------
with tab_products:
    st.subheader("Top Product Velocity & Revenue Drivers")
    
    top_n = st.slider("Select Number of Top Products to Display", min_value=5, max_value=50, value=15, step=5)
    
    prod_grouped = filtered_prod.groupby(['item_id', 'dept_id', 'cat_id']).agg({
        'total_units': 'sum',
        'est_revenue': 'sum',
        'avg_sell_price': 'mean',
        'sparsity_pct': 'mean'
    }).reset_index()
    
    top_vol = prod_grouped.sort_values('total_units', ascending=False).head(top_n)
    top_rev = prod_grouped.sort_values('est_revenue', ascending=False).head(top_n)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        fig_top_vol = px.bar(
            top_vol,
            x='total_units',
            y='item_id',
            orientation='h',
            color='cat_id',
            title=f"Top {top_n} Products by Total Volume",
            labels={'total_units': 'Units Sold', 'item_id': 'Item ID', 'cat_id': 'Category'},
            color_discrete_sequence=['#2563eb', '#10b981', '#f59e0b']
        )
        fig_top_vol.update_layout(height=480, yaxis={'categoryorder':'total ascending'}, template="plotly_white")
        st.plotly_chart(fig_top_vol, use_container_width=True)

    with p_col2:
        fig_top_rev = px.bar(
            top_rev,
            x='est_revenue',
            y='item_id',
            orientation='h',
            color='cat_id',
            title=f"Top {top_n} Products by Estimated Revenue",
            labels={'est_revenue': 'Est. Revenue ($)', 'item_id': 'Item ID', 'cat_id': 'Category'},
            color_discrete_sequence=['#2563eb', '#10b981', '#f59e0b']
        )
        fig_top_rev.update_layout(height=480, yaxis={'categoryorder':'total ascending'}, template="plotly_white")
        st.plotly_chart(fig_top_rev, use_container_width=True)

    st.markdown("**Product Performance Table**")
    display_df = prod_grouped.sort_values('est_revenue', ascending=False).head(100).rename(columns={
        'item_id': 'Item ID',
        'dept_id': 'Department',
        'cat_id': 'Category',
        'total_units': 'Units Sold',
        'est_revenue': 'Est. Revenue ($)',
        'avg_sell_price': 'Avg Price ($)',
        'sparsity_pct': 'Zero-Demand (%)'
    })
    st.dataframe(
        display_df.style.format({
            'Units Sold': '{:,.0f}',
            'Est. Revenue ($)': '${:,.2f}',
            'Avg Price ($)': '${:,.2f}',
            'Zero-Demand (%)': '{:.1f}%'
        }),
        use_container_width=True,
        height=320
    )
    
    # CSV export
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Top Products to CSV",
        data=csv,
        file_name='walmart_top_products_analytics.csv',
        mime='text/csv'
    )
