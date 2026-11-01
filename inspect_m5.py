import os
import pandas as pd
import numpy as np

DATA_DIR = "data"

def inspect_calendar():
    print("=" * 60)
    print("INSPECTING: calendar.csv")
    print("=" * 60)
    cal_path = os.path.join(DATA_DIR, "calendar.csv")
    df_cal = pd.read_csv(cal_path)
    print(f"Shape: {df_cal.shape}")
    print(f"Memory Usage: {df_cal.memory_usage().sum() / 1024**2:.2f} MB")
    print("\nColumns & Data Types:")
    print(df_cal.dtypes)
    print("\nFirst 3 rows:")
    print(df_cal.head(3))
    print("\nDate Range:")
    print(f"Min date: {df_cal['date'].min()}, Max date: {df_cal['date'].max()}")
    print(f"Total days: {len(df_cal)}, 'd' identifiers from {df_cal['d'].iloc[0]} to {df_cal['d'].iloc[-1]}")
    
    print("\nMissing values:")
    print(df_cal.isnull().sum())
    
    print("\nUnique Events (event_name_1):", df_cal['event_name_1'].dropna().unique())
    print("\nEvent Types (event_type_1):", df_cal['event_type_1'].value_counts().to_dict())
    print("\nSecondary Events (event_name_2):", df_cal['event_name_2'].dropna().unique())
    print("\nSNAP days count:")
    print(f"CA: {df_cal['snap_CA'].sum()}, TX: {df_cal['snap_TX'].sum()}, WI: {df_cal['snap_WI'].sum()}")
    return df_cal

def inspect_sales():
    print("\n" + "=" * 60)
    print("INSPECTING: sales_train_validation.csv & sales_train_evaluation.csv")
    print("=" * 60)
    val_path = os.path.join(DATA_DIR, "sales_train_validation.csv")
    df_val = pd.read_csv(val_path)
    print(f"sales_train_validation shape: {df_val.shape}")
    
    eval_path = os.path.join(DATA_DIR, "sales_train_evaluation.csv")
    df_eval = pd.read_csv(eval_path)
    print(f"sales_train_evaluation shape: {df_eval.shape}")
    
    print(f"Validation columns: {df_val.columns[:10].tolist()} ... {df_val.columns[-5:].tolist()}")
    print(f"Evaluation columns: {df_eval.columns[:10].tolist()} ... {df_eval.columns[-5:].tolist()}")
    
    d_cols_val = [c for c in df_val.columns if c.startswith('d_')]
    d_cols_eval = [c for c in df_eval.columns if c.startswith('d_')]
    print(f"Days in validation: {len(d_cols_val)} ({d_cols_val[0]} to {d_cols_val[-1]})")
    print(f"Days in evaluation: {len(d_cols_eval)} ({d_cols_eval[0]} to {d_cols_eval[-1]})")
    
    print("\nHierarchical breakdown:")
    print(f"Unique item_id: {df_val['item_id'].nunique()}")
    print(f"Unique dept_id: {df_val['dept_id'].nunique()} -> {df_val['dept_id'].unique().tolist()}")
    print(f"Unique cat_id: {df_val['cat_id'].nunique()} -> {df_val['cat_id'].unique().tolist()}")
    print(f"Unique store_id: {df_val['store_id'].nunique()} -> {df_val['store_id'].unique().tolist()}")
    print(f"Unique state_id: {df_val['state_id'].nunique()} -> {df_val['state_id'].unique().tolist()}")
    
    print("\nStores per State:")
    print(df_val.groupby('state_id')['store_id'].nunique())
    
    print("\nItems per Category:")
    print(df_val.groupby('cat_id')['item_id'].nunique())
    
    print("\nItems per Department:")
    print(df_val.groupby(['cat_id', 'dept_id'])['item_id'].nunique())
    
    # Sparsity / Zero analysis on validation sales
    sales_matrix = df_val[d_cols_val].values
    total_entries = sales_matrix.size
    zero_entries = np.count_nonzero(sales_matrix == 0)
    sparsity = (zero_entries / total_entries) * 100
    total_sales_units = sales_matrix.sum()
    mean_daily_sales = sales_matrix.mean()
    max_single_sale = sales_matrix.max()
    
    print("\nSparsity & Sales Statistics (Validation):")
    print(f"Total time series: {len(df_val):,}")
    print(f"Total data points (30490 x 1913): {total_entries:,}")
    print(f"Zero sales count: {zero_entries:,} ({sparsity:.2f}% zero values - intermittent demand)")
    print(f"Total units sold across all items & days: {total_sales_units:,}")
    print(f"Mean unit sales per item/day: {mean_daily_sales:.3f}")
    print(f"Max unit sales in a single day: {max_single_sale}")
    
    return df_val

def inspect_prices():
    print("\n" + "=" * 60)
    print("INSPECTING: sell_prices.csv")
    print("=" * 60)
    prices_path = os.path.join(DATA_DIR, "sell_prices.csv")
    df_prices = pd.read_csv(prices_path)
    print(f"sell_prices shape: {df_prices.shape}")
    print(f"Memory Usage: {df_prices.memory_usage().sum() / 1024**2:.2f} MB")
    print("\nColumns & Types:")
    print(df_prices.dtypes)
    print("\nFirst 5 rows:")
    print(df_prices.head())
    print("\nPrice Statistics:")
    print(df_prices['sell_price'].describe())
    print(f"\nMissing values in sell_prices:")
    print(df_prices.isnull().sum())
    print(f"\nNumber of unique stores: {df_prices['store_id'].nunique()}")
    print(f"Number of unique items: {df_prices['item_id'].nunique()}")
    print(f"Unique wm_yr_wk range: {df_prices['wm_yr_wk'].min()} to {df_prices['wm_yr_wk'].max()}")
    return df_prices

def inspect_sample_submission():
    print("\n" + "=" * 60)
    print("INSPECTING: sample_submission.csv")
    print("=" * 60)
    sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
    df_sub = pd.read_csv(sub_path)
    print(f"sample_submission shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")
    print(f"Number of rows: {len(df_sub)} (expected 30,490 validation + 30,490 evaluation = 60,980)")
    print("\nSample rows:")
    print(df_sub.head(3))
    print(df_sub.tail(3))
    return df_sub

if __name__ == "__main__":
    inspect_calendar()
    inspect_sales()
    inspect_prices()
    inspect_sample_submission()

