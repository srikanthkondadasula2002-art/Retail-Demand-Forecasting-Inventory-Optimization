"""
Retail Demand Forecasting & Inventory Optimization - Executive Dashboard
Modules:
1. 📊 Sales Analytics & KPIs (Historical demand, trends, store benchmarks, category mix, product leaderboard)
2. 🔮 Demand Forecasting (Prophet vs LightGBM 28-day predictions vs ground truth actuals)
3. 📦 Inventory Optimization (Safety Stock, Reorder Point, Economic Order Quantity)
4. 🏷️ Price Drop Scenario Simulation (What-if price elasticity & demand shock simulation)
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Retail Demand Forecasting & Inventory Optimization | Walmart M5",
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
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

@st.cache_data(show_spinner="Loading precomputed sales telemetry...")
def load_data():
    dept_path = os.path.join(PROCESSED_DIR, "store_dept_daily.parquet")
    prod_path = os.path.join(PROCESSED_DIR, "product_performance.parquet")
    
    if not os.path.exists(dept_path) or not os.path.exists(prod_path):
        from src.prepare_dashboard_data import prepare_all
        prepare_all()
        
    df_daily = pd.read_parquet(dept_path)
    df_prod = pd.read_parquet(prod_path)
    return df_daily, df_prod

@st.cache_data(show_spinner="Loading forecasting model telemetry...")
def load_forecast_data():
    fc_path = os.path.join(PROCESSED_DIR, "forecasts_integrated.parquet")
    if os.path.exists(fc_path):
        df_fc = pd.read_parquet(fc_path)
        df_fc["date"] = pd.to_datetime(df_fc["date"])
        return df_fc
    return pd.DataFrame()

df_daily, df_prod = load_data()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/ca/Walmart_logo.svg", width=180)
st.sidebar.title("Navigation")
dashboard_mode = st.sidebar.radio(
    "Select Dashboard View:",
    [
        "📊 Sales Analytics & KPIs",
        "🔮 Demand Forecasting (Prophet vs LightGBM)",
        "📦 Inventory Optimization",
        "🏷️ Price Drop Scenario Simulation"
    ],
    index=0
)

# ==============================================================================
# VIEW 1: SALES ANALYTICS & KPIS
# ==============================================================================
if dashboard_mode == "📊 Sales Analytics & KPIs":
    st.sidebar.markdown("---")
    st.sidebar.title("Analytics Filters")

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

    # Main Header
    st.title("🛒 Walmart Retail Sales Analytics Dashboard")
    st.markdown("Interactive performance monitoring, demand trend discovery, and seasonal diagnostics based on the M5 dataset.")

    # KPI Metrics Cards
    total_units = int(filtered_daily['units'].sum())
    num_days = max(1, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)
    avg_daily_sales = total_units / num_days

    total_revenue = float(filtered_prod['est_revenue'].sum())
    avg_item_price = float(filtered_prod['avg_sell_price'].mean()) if len(filtered_prod) > 0 else 0.0

    active_skus = int(filtered_prod['item_id'].nunique())
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

    # Tabs for Deep Dive
    tab_trends, tab_stores, tab_categories, tab_products = st.tabs([
        "📈 Longitudinal Sales Trends",
        "🏪 Store Performance Benchmark",
        "📦 Category & Department Mix",
        "🏆 Top Products & Leaderboard"
    ])

    with tab_trends:
        st.subheader("Sales Volume Over Time")
        trend_df = filtered_daily.groupby('date')['units'].sum().reset_index().sort_values('date')
        if smoothing_option == "7-Day Moving Average":
            trend_df['display_units'] = trend_df['units'].rolling(7, min_periods=1).mean()
            chart_title = "Daily Sales Units (7-Day Moving Average)"
        elif smoothing_option == "30-Day Moving Average":
            trend_df['display_units'] = trend_df['units'].rolling(30, min_periods=1).mean()
            chart_title = "Daily Sales Units (30-Day Moving Average)"
        else:
            trend_df['display_units'] = trend_df['units']
            chart_title = "Daily Sales Units (Raw Daily Observations)"

        fig_trend = px.line(
            trend_df,
            x='date',
            y='display_units',
            title=chart_title,
            labels={'display_units': 'Units Sold', 'date': 'Date'},
            color_discrete_sequence=['#2563eb']
        )
        fig_trend.update_layout(height=420, hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig_trend, use_container_width=True)

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.subheader("Intra-Week Day-of-Week Seasonality")
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_df = filtered_daily.copy()
            dow_df['weekday'] = dow_df['date'].dt.day_name()
            dow_agg = dow_df.groupby('weekday')['units'].mean().reindex(dow_order).reset_index()
            fig_dow = px.bar(
                dow_agg,
                x='weekday',
                y='units',
                title="Average Daily Units Sold by Day of Week",
                labels={'units': 'Average Units', 'weekday': 'Day of Week'},
                color='units',
                color_continuous_scale='Blues'
            )
            fig_dow.update_layout(height=360, coloraxis_showscale=False, template="plotly_white")
            st.plotly_chart(fig_dow, use_container_width=True)

        with col_w2:
            st.subheader("SNAP Policy Stimulus Lift")
            snap_df = filtered_daily.copy()
            state_code = selected_state if selected_state != "All" else "CA"
            snap_col = f"snap_{state_code}" if f"snap_{state_code}" in snap_df.columns else "snap_CA"
            snap_agg = snap_df.groupby(snap_col)['units'].mean().reset_index()
            snap_agg['Status'] = snap_agg[snap_col].map({0: 'Non-SNAP Day', 1: 'SNAP Active (Days 1-10)'})
            fig_snap = px.bar(
                snap_agg,
                x='Status',
                y='units',
                title=f"SNAP Benefit Demand Lift ({state_code})",
                labels={'units': 'Mean Daily Sales (Units)'},
                color='Status',
                color_discrete_map={'Non-SNAP Day': '#94a3b8', 'SNAP Active (Days 1-10)': '#10b981'}
            )
            fig_snap.update_layout(height=360, showlegend=False, template="plotly_white")
            st.plotly_chart(fig_snap, use_container_width=True)

    with tab_stores:
        st.subheader("Store Sales Volume & Trajectory Comparison")
        store_daily = filtered_daily.groupby(['date', 'store_id'])['units'].sum().reset_index()
        store_daily['smoothed_units'] = store_daily.groupby('store_id')['units'].transform(
            lambda x: x.rolling(14, min_periods=1).mean()
        )
        fig_stores = px.line(
            store_daily,
            x='date',
            y='smoothed_units',
            color='store_id',
            title="14-Day Smoothed Sales Trajectory by Store",
            labels={'smoothed_units': '14-Day Avg Units', 'store_id': 'Store'},
            template="plotly_white"
        )
        fig_stores.update_layout(height=450, hovermode="x unified")
        st.plotly_chart(fig_stores, use_container_width=True)

    with tab_categories:
        st.subheader("Category & Department Composition")
        cat_trend = filtered_daily.groupby(['date', 'cat_id'])['units'].sum().reset_index()
        fig_cat = px.area(
            cat_trend,
            x='date',
            y='units',
            color='cat_id',
            title="Longitudinal Category Volume Share",
            labels={'units': 'Units Sold', 'cat_id': 'Category'},
            color_discrete_map={'FOODS': '#10b981', 'HOUSEHOLD': '#3b82f6', 'HOBBIES': '#f59e0b'},
            template="plotly_white"
        )
        fig_cat.update_layout(height=450, hovermode="x unified")
        st.plotly_chart(fig_cat, use_container_width=True)

    with tab_products:
        st.subheader("Top Product Leaderboards")
        top_n = st.slider("Select Top N Products", min_value=5, max_value=30, value=10)
        prod_grouped = filtered_prod.groupby(['item_id', 'dept_id', 'cat_id']).agg({
            'total_units': 'sum',
            'est_revenue': 'sum',
            'avg_sell_price': 'mean',
            'sparsity_pct': 'mean'
        }).reset_index()

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            top_vol = prod_grouped.sort_values('total_units', ascending=False).head(top_n)
            fig_top_vol = px.bar(
                top_vol,
                x='total_units',
                y='item_id',
                orientation='h',
                color='cat_id',
                title=f"Top {top_n} Products by Unit Volume",
                labels={'total_units': 'Units Sold', 'item_id': 'Item ID', 'cat_id': 'Category'},
                color_discrete_sequence=['#10b981', '#2563eb', '#f59e0b']
            )
            fig_top_vol.update_layout(height=450, yaxis={'categoryorder':'total ascending'}, template="plotly_white")
            st.plotly_chart(fig_top_vol, use_container_width=True)

        with c_p2:
            top_rev = prod_grouped.sort_values('est_revenue', ascending=False).head(top_n)
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
            fig_top_rev.update_layout(height=450, yaxis={'categoryorder':'total ascending'}, template="plotly_white")
            st.plotly_chart(fig_top_rev, use_container_width=True)

# ==============================================================================
# VIEW 2: DEMAND FORECASTING (PROPHET VS LIGHTGBM)
# ==============================================================================
elif dashboard_mode == "🔮 Demand Forecasting (Prophet vs LightGBM)":
    st.title("🔮 28-Day Demand Forecasting: Prophet vs. LightGBM")
    st.markdown("Out-of-sample demand evaluation across 10 Walmart retail locations over the 28-day blind test horizon (`2016-04-25` to `2016-05-22`).")

    df_fc = load_forecast_data()
    if df_fc.empty:
        st.error("Forecast telemetry not found. Please run `python src/integrate_forecasts_db.py` to populate data.")
    else:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Forecasting Filters")

        all_stores = ["All Stores (Network Total)"] + sorted(df_fc["store_id"].unique().tolist())
        selected_fc_store = st.sidebar.selectbox("Store Selection", all_stores, index=0)

        model_options = ["Both Models (Comparison)", "LightGBM Tweedie GBDT", "Facebook Prophet"]
        selected_model = st.sidebar.radio("Model Display", model_options, index=0)
        show_ci = st.sidebar.checkbox("Show Prophet 80% Uncertainty Band", value=True)

        # Filter dataset
        if selected_fc_store == "All Stores (Network Total)":
            daily_act = df_fc[df_fc["model_name"] == "LightGBM_Tweedie"].groupby("date")["actual_units"].sum().reset_index()
            daily_lgb = df_fc[df_fc["model_name"] == "LightGBM_Tweedie"].groupby("date")["forecast_units"].sum().reset_index()
            daily_prp = df_fc[df_fc["model_name"] == "Prophet"].groupby("date").agg({
                "forecast_units": "sum", "lower_bound": "sum", "upper_bound": "sum"
            }).reset_index()

            plot_df = pd.DataFrame({
                "date": daily_act["date"],
                "actual_units": daily_act["actual_units"],
                "lightgbm_forecast": daily_lgb["forecast_units"],
                "prophet_forecast": daily_prp["forecast_units"],
                "prophet_lower": daily_prp["lower_bound"],
                "prophet_upper": daily_prp["upper_bound"]
            })
        else:
            lgb_sub = df_fc[(df_fc["store_id"] == selected_fc_store) & (df_fc["model_name"] == "LightGBM_Tweedie")].sort_values("date")
            prp_sub = df_fc[(df_fc["store_id"] == selected_fc_store) & (df_fc["model_name"] == "Prophet")].sort_values("date")

            plot_df = pd.DataFrame({
                "date": lgb_sub["date"].values,
                "actual_units": lgb_sub["actual_units"].values,
                "lightgbm_forecast": lgb_sub["forecast_units"].values,
                "prophet_forecast": prp_sub["forecast_units"].values,
                "prophet_lower": prp_sub["lower_bound"].values,
                "prophet_upper": prp_sub["upper_bound"].values
            })

        # Calculate metrics
        tot_act = plot_df["actual_units"].sum()
        tot_lgb = plot_df["lightgbm_forecast"].sum()
        tot_prp = plot_df["prophet_forecast"].sum()

        lgb_wape = (np.abs(plot_df["lightgbm_forecast"] - plot_df["actual_units"]).sum() / tot_act) * 100
        prp_wape = (np.abs(plot_df["prophet_forecast"] - plot_df["actual_units"]).sum() / tot_act) * 100
        lgb_mae = np.mean(np.abs(plot_df["lightgbm_forecast"] - plot_df["actual_units"]))
        prp_mae = np.mean(np.abs(plot_df["prophet_forecast"] - plot_df["actual_units"]))

        # KPI Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Actual Demand</div>
                <div class="metric-value">{tot_act:,.0f}</div>
                <div class="metric-sub sub-muted">28-day ground truth units</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">LightGBM Forecast</div>
                <div class="metric-value">{tot_lgb:,.0f}</div>
                <div class="metric-sub sub-green">WAPE: {lgb_wape:.2f}% | MAE: {lgb_mae:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Prophet Forecast</div>
                <div class="metric-value">{tot_prp:,.0f}</div>
                <div class="metric-sub sub-muted">WAPE: {prp_wape:.2f}% | MAE: {prp_mae:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            winner = "🏆 LightGBM Tweedie" if lgb_wape <= prp_wape else "🏆 Facebook Prophet"
            delta = abs(lgb_wape - prp_wape)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Champion Model</div>
                <div class="metric-value" style="font-size: 1.35rem;">{winner}</div>
                <div class="metric-sub sub-green">Lead Margin: {delta:.2f}% WAPE</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plot 1: Actual vs Forecasted Demand Trajectory
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=plot_df["date"],
            y=plot_df["actual_units"],
            name="Ground Truth Actual Demand",
            line=dict(color="#0f172a", width=3),
            mode="lines+markers",
            marker=dict(size=6)
        ))

        if selected_model in ["Both Models (Comparison)", "LightGBM Tweedie GBDT"]:
            fig_curve.add_trace(go.Scatter(
                x=plot_df["date"],
                y=plot_df["lightgbm_forecast"],
                name="LightGBM Tweedie (GBDT)",
                line=dict(color="#10b981", width=2.5, dash="dash"),
                mode="lines+markers",
                marker=dict(size=5, symbol="square")
            ))

        if selected_model in ["Both Models (Comparison)", "Facebook Prophet"]:
            if show_ci and "prophet_upper" in plot_df.columns:
                fig_curve.add_trace(go.Scatter(
                    x=pd.concat([plot_df["date"], plot_df["date"][::-1]]),
                    y=pd.concat([plot_df["prophet_upper"], plot_df["prophet_lower"][::-1]]),
                    fill='toself',
                    fillcolor='rgba(99, 102, 241, 0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name="Prophet 80% CI Bounds"
                ))

            fig_curve.add_trace(go.Scatter(
                x=plot_df["date"],
                y=plot_df["prophet_forecast"],
                name="Facebook Prophet",
                line=dict(color="#6366f1", width=2.5, dash="dot"),
                mode="lines+markers",
                marker=dict(size=5, symbol="triangle-up")
            ))

        fig_curve.update_layout(
            title=f"Actual vs. Predicted Demand Trajectory ({selected_fc_store})",
            xaxis_title="Forecast Horizon Date",
            yaxis_title="Units Sold",
            hovermode="x unified",
            template="plotly_white",
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # Plot 2 & 3: Residuals and Store-by-Store Benchmark
        fc_col1, fc_col2 = st.columns(2)
        with fc_col1:
            fig_res = go.Figure()
            if selected_model in ["Both Models (Comparison)", "LightGBM Tweedie GBDT"]:
                fig_res.add_trace(go.Bar(
                    x=plot_df["date"],
                    y=plot_df["lightgbm_forecast"] - plot_df["actual_units"],
                    name="LightGBM Residuals",
                    marker_color="#10b981",
                    opacity=0.8
                ))
            if selected_model in ["Both Models (Comparison)", "Facebook Prophet"]:
                fig_res.add_trace(go.Bar(
                    x=plot_df["date"],
                    y=plot_df["prophet_forecast"] - plot_df["actual_units"],
                    name="Prophet Residuals",
                    marker_color="#6366f1",
                    opacity=0.8
                ))
            fig_res.update_layout(
                title="Daily Forecast Residuals (Predicted - Actual Units)",
                xaxis_title="Date",
                yaxis_title="Residual Error (Units)",
                barmode="group",
                template="plotly_white",
                height=380
            )
            st.plotly_chart(fig_res, use_container_width=True)

        with fc_col2:
            store_benchmarks = []
            for s in sorted(df_fc["store_id"].unique()):
                s_lgb = df_fc[(df_fc["store_id"] == s) & (df_fc["model_name"] == "LightGBM_Tweedie")]
                s_prp = df_fc[(df_fc["store_id"] == s) & (df_fc["model_name"] == "Prophet")]
                act_s = s_lgb["actual_units"].sum()
                lgb_w = (s_lgb["absolute_error"].sum() / act_s) * 100
                prp_w = (s_prp["absolute_error"].sum() / act_s) * 100
                store_benchmarks.append({
                    "store_id": s,
                    "LightGBM WAPE (%)": lgb_w,
                    "Prophet WAPE (%)": prp_w
                })

            df_sb = pd.DataFrame(store_benchmarks)
            fig_sb = go.Figure()
            fig_sb.add_trace(go.Bar(
                x=df_sb["store_id"],
                y=df_sb["LightGBM WAPE (%)"],
                name="LightGBM Tweedie",
                marker_color="#10b981"
            ))
            fig_sb.add_trace(go.Bar(
                x=df_sb["store_id"],
                y=df_sb["Prophet WAPE (%)"],
                name="Facebook Prophet",
                marker_color="#6366f1"
            ))
            fig_sb.update_layout(
                title="Store-by-Store Accuracy Benchmark (WAPE % - Lower is Better)",
                xaxis_title="Store ID",
                yaxis_title="WAPE (%)",
                barmode="group",
                template="plotly_white",
                height=380
            )
            st.plotly_chart(fig_sb, use_container_width=True)

        # Data Ledger Table
        st.markdown("### 📋 28-Day Detailed Forecast Ledger")
        display_table = plot_df.copy()
        display_table["LightGBM Delta"] = display_table["lightgbm_forecast"] - display_table["actual_units"]
        display_table["Prophet Delta"] = display_table["prophet_forecast"] - display_table["actual_units"]
        display_table["Winning Model"] = np.where(
            np.abs(display_table["LightGBM Delta"]) <= np.abs(display_table["Prophet Delta"]),
            "LightGBM", "Prophet"
        )

        st.dataframe(
            display_table.style.format({
                "actual_units": "{:,.0f}",
                "lightgbm_forecast": "{:,.1f}",
                "prophet_forecast": "{:,.1f}",
                "LightGBM Delta": "{:+,.1f}",
                "Prophet Delta": "{:+,.1f}"
            }),
            use_container_width=True,
            height=280
        )

        csv_bytes = display_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export 28-Day Forecast Predictions (CSV)",
            data=csv_bytes,
            file_name=f"walmart_forecast_predictions_{selected_fc_store}.csv",
            mime="text/csv"
        )

# ==============================================================================
# VIEW 3: INVENTORY OPTIMIZATION PLACEHOLDER (POPULATED IN NEXT MILESTONE)
# ==============================================================================
elif dashboard_mode == "📦 Inventory Optimization":
    st.title("📦 Retail Inventory Optimization Engine")
    st.markdown("Automated calculation of Safety Stock, Reorder Points (ROP), and Economic Order Quantities (EOQ).")
    st.info("Inventory optimization parameters and recommendations will load here.")

# ==============================================================================
# VIEW 4: PRICE DROP SCENARIO SIMULATION PLACEHOLDER (POPULATED IN NEXT MILESTONE)
# ==============================================================================
elif dashboard_mode == "🏷️ Price Drop Scenario Simulation":
    st.title("🏷️ What-If Scenario Analysis: Price Drop Simulation")
    st.markdown("Simulate retail price elasticity, demand surges, revenue impact, and stockout risk.")
    st.info("Interactive scenario inputs will load here.")
