"""
Precomputes aggregated datasets for sub-second Streamlit dashboard performance.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def prepare_all():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("[1/2] Processing store-department daily time series...")
    dept_parquet = os.path.join(PROCESSED_DIR, "store_dept_daily.parquet")
    if not os.path.exists(dept_parquet):
        df_val = pd.read_csv(os.path.join(DATA_DIR, "sales_train_validation.csv"))
        d_cols = [c for c in df_val.columns if c.startswith('d_')]
        agg = df_val.groupby(['state_id', 'store_id', 'cat_id', 'dept_id'])[d_cols].sum().reset_index()
        agg_long = pd.melt(agg, id_vars=['state_id', 'store_id', 'cat_id', 'dept_id'], var_name='d', value_name='units')
        
        df_cal = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
        agg_long = agg_long.merge(
            df_cal[['d', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'snap_CA', 'snap_TX', 'snap_WI', 'event_name_1', 'event_type_1']], 
            on='d', 
            how='left'
        )
        agg_long['date'] = pd.to_datetime(agg_long['date'])
        agg_long.to_parquet(dept_parquet, index=False)
        print(f"Saved {len(agg_long):,} rows to {dept_parquet}")
    else:
        print("Found existing store_dept_daily.parquet.")

    print("[2/2] Processing product rankings & price metrics...")
    prod_parquet = os.path.join(PROCESSED_DIR, "product_performance.parquet")
    if not os.path.exists(prod_parquet):
        df_val = pd.read_csv(os.path.join(DATA_DIR, "sales_train_validation.csv"))
        d_cols = [c for c in df_val.columns if c.startswith('d_')]
        
        # Calculate item total units and zeros per item
        item_stats = df_val[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].copy()
        sales_mat = df_val[d_cols].values
        item_stats['total_units'] = sales_mat.sum(axis=1)
        item_stats['zero_days'] = np.count_nonzero(sales_mat == 0, axis=1)
        item_stats['total_days'] = len(d_cols)
        item_stats['sparsity_pct'] = (item_stats['zero_days'] / item_stats['total_days']) * 100
        
        # Avg price per item & store
        df_prices = pd.read_csv(os.path.join(DATA_DIR, "sell_prices.csv"))
        avg_prices = df_prices.groupby(['store_id', 'item_id'])['sell_price'].mean().reset_index(name='avg_sell_price')
        
        item_stats = item_stats.merge(avg_prices, on=['store_id', 'item_id'], how='left')
        item_stats['avg_sell_price'] = item_stats['avg_sell_price'].fillna(0)
        item_stats['est_revenue'] = item_stats['total_units'] * item_stats['avg_sell_price']
        
        item_stats.to_parquet(prod_parquet, index=False)
        print(f"Saved {len(item_stats):,} rows to {prod_parquet}")
    else:
        print("Found existing product_performance.parquet.")

if __name__ == "__main__":
    prepare_all()

