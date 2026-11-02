"""
Exploratory Data Analysis (EDA) Pipeline
Analyzes product-wise and store-wise sales dynamics on the Walmart M5 dataset.
Generates publication-quality figures and summary metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
FIGURES_DIR = os.path.join("reports", "figures")
METRICS_PATH = os.path.join("reports", "eda_summary_metrics.json")
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

def load_data():
    print("[1/6] Loading datasets...", flush=True)
    df_cal = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
    df_val = pd.read_csv(os.path.join(DATA_DIR, "sales_train_validation.csv"))
    df_prices = pd.read_csv(os.path.join(DATA_DIR, "sell_prices.csv"))
    return df_cal, df_val, df_prices

def run_product_analysis(df_val, df_prices):
    print("[2/6] Running Product-wise Analysis...", flush=True)
    d_cols = [c for c in df_val.columns if c.startswith('d_')]
    
    # 1. Product total sales across all 10 stores
    item_sales = df_val.groupby(['item_id', 'dept_id', 'cat_id'])[d_cols].sum().sum(axis=1).reset_index(name='total_units')
    
    # Calculate average selling price per item across all stores
    avg_price = df_prices.groupby('item_id')['sell_price'].mean().reset_index(name='avg_sell_price')
    item_sales = item_sales.merge(avg_price, on='item_id', how='left')
    item_sales['est_revenue'] = item_sales['total_units'] * item_sales['avg_sell_price']
    
    # Top 20 by Volume & Top 20 by Revenue
    top_volume = item_sales.sort_values('total_units', ascending=False).head(20)
    top_revenue = item_sales.sort_values('est_revenue', ascending=False).head(20)
    
    # Intermittency / Sparsity per category
    cat_summary = []
    for cat, grp in df_val.groupby('cat_id'):
        mat = grp[d_cols].values
        zeros = np.count_nonzero(mat == 0)
        total_pts = mat.size
        cat_summary.append({
            'category': cat,
            'items': grp['item_id'].nunique(),
            'total_series': len(grp),
            'total_volume': int(mat.sum()),
            'sparsity_pct': round((zeros / total_pts) * 100, 2),
            'mean_daily_sales_per_series': round(float(mat.mean()), 3),
            'p95_daily_sales': round(float(np.percentile(mat, 95)), 2)
        })
        
    dept_summary = []
    for dept, grp in df_val.groupby(['cat_id', 'dept_id']):
        mat = grp[d_cols].values
        dept_summary.append({
            'category': dept[0],
            'department': dept[1],
            'items': grp['item_id'].nunique(),
            'total_volume': int(mat.sum()),
            'sparsity_pct': round((np.count_nonzero(mat == 0) / mat.size) * 100, 2)
        })

    # Plot Category & Department Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cat_df = pd.DataFrame(cat_summary)
    sns.barplot(data=cat_df, x='category', y='total_volume', ax=axes[0], palette='Blues_r')
    axes[0].set_title('Total Unit Sales by Product Category', fontsize=14, weight='bold')
    axes[0].set_ylabel('Total Units Sold')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height()):,}", 
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')
                         
    dept_df = pd.DataFrame(dept_summary).sort_values('total_volume', ascending=False)
    sns.barplot(data=dept_df, x='department', y='total_volume', hue='category', dodge=False, ax=axes[1], palette='Set2')
    axes[1].set_title('Total Unit Sales by Department', fontsize=14, weight='bold')
    axes[1].set_ylabel('Total Units Sold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure2_category_department_breakdown.png"), dpi=200)
    plt.close()

    # Plot Top 15 Products by Volume and Revenue
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.barplot(data=top_volume.head(15), y='item_id', x='total_units', hue='cat_id', dodge=False, ax=axes[0], palette='viridis')
    axes[0].set_title('Top 15 Products by Total Unit Volume', fontsize=13, weight='bold')
    axes[0].set_xlabel('Total Units Sold Across 10 Stores')
    axes[0].set_ylabel('Item ID')
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    sns.barplot(data=top_revenue.head(15), y='item_id', x='est_revenue', hue='cat_id', dodge=False, ax=axes[1], palette='magma')
    axes[1].set_title('Top 15 Products by Estimated Total Revenue ($)', fontsize=13, weight='bold')
    axes[1].set_xlabel('Estimated Revenue ($)')
    axes[1].set_ylabel('')
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:,.0f}".format(x)))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure3_top_products_volume_revenue.png"), dpi=200)
    plt.close()

    # Plot Sparsity / Intermittent Demand distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sparsity_data = pd.DataFrame(cat_summary)
    sns.barplot(data=sparsity_data, x='category', y='sparsity_pct', palette='rocket', ax=ax)
    ax.set_title('Intermittent Demand: Zero-Sales Ratio (%) by Category', fontsize=14, weight='bold')
    ax.set_ylabel('Percentage of Days with Zero Sales (%)')
    ax.set_ylim(0, 100)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, weight='bold', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure5_intermittent_demand_sparsity.png"), dpi=200)
    plt.close()

    return item_sales, cat_summary, dept_summary

def run_store_analysis(df_cal, df_val, df_prices):
    print("[3/6] Running Store-wise Analysis...", flush=True)
    d_cols = [c for c in df_val.columns if c.startswith('d_')]
    
    # Store aggregated daily sales
    store_daily = df_val.groupby(['state_id', 'store_id'])[d_cols].sum()
    
    # Total volume & estimated revenue per store
    avg_price_map = df_prices.groupby(['store_id', 'item_id'])['sell_price'].mean().to_dict()
    
    # Approximate revenue per store
    store_metrics = []
    for store, grp in df_val.groupby('store_id'):
        state = grp['state_id'].iloc[0]
        vol = grp[d_cols].sum().sum()
        # Item avg price in this store
        prices = df_prices[df_prices['store_id'] == store].groupby('item_id')['sell_price'].mean()
        item_vols = grp.groupby('item_id')[d_cols].sum().sum(axis=1)
        rev = (item_vols * prices).sum()
        store_metrics.append({
            'store_id': store,
            'state_id': state,
            'total_volume': int(vol),
            'est_revenue': round(float(rev), 2),
            'mean_daily_store_volume': round(float(vol / len(d_cols)), 1)
        })
    store_df = pd.DataFrame(store_metrics).sort_values('total_volume', ascending=False)
    
    # Longitudinal Store Sales Trends (30-day moving average)
    print("[4/6] Generating Longitudinal Store Sales Trends...", flush=True)
    dates = pd.to_datetime(df_cal.iloc[:len(d_cols)]['date'])
    store_daily_T = store_daily.reset_index(level=0, drop=True).T
    store_daily_T.index = dates
    store_smooth = store_daily_T.rolling(30, min_periods=7).mean()

    fig, ax = plt.subplots(figsize=(16, 7))
    palette = sns.color_palette("tab10", n_colors=10)
    for idx, col in enumerate(store_smooth.columns):
        ax.plot(store_smooth.index, store_smooth[col], label=col, color=palette[idx], linewidth=1.8)
    ax.set_title('Store-Level Sales Trajectory (30-Day Moving Average, 2011–2016)', fontsize=15, weight='bold')
    ax.set_ylabel('Daily Unit Sales (Smoothed)')
    ax.set_xlabel('Date')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.legend(title='Store ID', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure1_store_sales_trends.png"), dpi=200)
    plt.close()

    # Day of Week Seasonality by Store
    print("[5/6] Analyzing Day of Week & Calendar Seasonality...", flush=True)
    cal_subset = df_cal.iloc[:len(d_cols)].copy()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    dow_store = store_daily_T.copy()
    dow_store['weekday'] = cal_subset['weekday'].values
    dow_avg = dow_store.groupby('weekday').mean().reindex(dow_order)

    fig, ax = plt.subplots(figsize=(14, 6))
    for col in dow_avg.columns:
        ax.plot(dow_avg.index, dow_avg[col], marker='o', label=col, linewidth=2)
    ax.set_title('Intra-Week Seasonality: Average Daily Sales by Day of Week across Stores', fontsize=14, weight='bold')
    ax.set_ylabel('Average Daily Sales')
    ax.set_xlabel('Day of Week')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.legend(title='Store ID', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure4_day_of_week_seasonality.png"), dpi=200)
    plt.close()

    # SNAP Impact Analysis by State
    print("[6/6] Quantifying SNAP Policy Stimulus Impact...", flush=True)
    snap_results = {}
    for state in ['CA', 'TX', 'WI']:
        snap_col = f'snap_{state}'
        state_stores = store_df[store_df['state_id'] == state]['store_id'].tolist()
        state_sales_series = store_daily_T[state_stores].sum(axis=1).values
        
        is_snap = cal_subset[snap_col].values == 1
        sales_snap = state_sales_series[is_snap].mean()
        sales_non_snap = state_sales_series[~is_snap].mean()
        lift_pct = ((sales_snap - sales_non_snap) / sales_non_snap) * 100
        
        snap_results[state] = {
            'snap_days_mean': round(float(sales_snap), 1),
            'non_snap_days_mean': round(float(sales_non_snap), 1),
            'lift_pct': round(float(lift_pct), 2)
        }
        
    snap_plot_df = pd.DataFrame([
        {'State': st, 'Day Type': 'SNAP Day', 'Average Daily Sales': data['snap_days_mean']}
        for st, data in snap_results.items()
    ] + [
        {'State': st, 'Day Type': 'Non-SNAP Day', 'Average Daily Sales': data['non_snap_days_mean']}
        for st, data in snap_results.items()
    ])
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=snap_plot_df, x='State', y='Average Daily Sales', hue='Day Type', palette=['#2b5c8f', '#d95f02'], ax=ax)
    ax.set_title('Impact of SNAP Benefits on Daily Sales (CA, TX, WI)', fontsize=14, weight='bold')
    ax.set_ylabel('Average Daily Sales (State Total)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    for i, state in enumerate(['CA', 'TX', 'WI']):
        lift = snap_results[state]['lift_pct']
        y_pos = max(snap_results[state]['snap_days_mean'], snap_results[state]['non_snap_days_mean']) + 300
        ax.text(i, y_pos, f"+{lift:.1f}% Lift", ha='center', weight='bold', color='#1b7837', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure6_snap_stimulus_impact.png"), dpi=200)
    plt.close()

    return store_df, snap_results

def main():
    df_cal, df_val, df_prices = load_data()
    item_sales, cat_summary, dept_summary = run_product_analysis(df_val, df_prices)
    store_df, snap_results = run_store_analysis(df_cal, df_val, df_prices)
    
    # Save combined metrics
    summary = {
        'total_time_series': len(df_val),
        'total_units_sold': int(df_val.loc[:, df_val.columns.str.startswith('d_')].sum().sum()),
        'category_metrics': cat_summary,
        'department_metrics': dept_summary,
        'store_metrics': store_df.to_dict(orient='records'),
        'snap_impact': snap_results
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n[SUCCESS] EDA successfully completed!")
    print(f"Figures saved to: {FIGURES_DIR}/")
    print(f"Summary metrics saved to: {METRICS_PATH}")

if __name__ == "__main__":
    main()
