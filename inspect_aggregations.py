import os
import pandas as pd
import numpy as np

DATA_DIR = "data"

df_cal = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
df_val = pd.read_csv(os.path.join(DATA_DIR, "sales_train_validation.csv"))
df_prices = pd.read_csv(os.path.join(DATA_DIR, "sell_prices.csv"))

d_cols = [c for c in df_val.columns if c.startswith('d_')]

# Sales by Category
cat_sales = df_val.groupby('cat_id')[d_cols].sum().sum(axis=1)
cat_pct = (cat_sales / cat_sales.sum()) * 100

# Category sparsity
cat_sparsity = {}
for cat, group in df_val.groupby('cat_id'):
    mat = group[d_cols].values
    zeros = np.count_nonzero(mat == 0)
    cat_sparsity[cat] = (zeros / mat.size) * 100

# Sales by State
state_sales = df_val.groupby('state_id')[d_cols].sum().sum(axis=1)
state_pct = (state_sales / state_sales.sum()) * 100

# Sales by Store
store_sales = df_val.groupby(['state_id', 'store_id'])[d_cols].sum().sum(axis=1)

# Time Series aggregate
total_daily = df_val[d_cols].sum(axis=0).values
cal_val = df_cal.iloc[:len(d_cols)].copy()
cal_val['total_sales'] = total_daily

# Day of week sales
dow_avg = cal_val.groupby('weekday')['total_sales'].mean().loc[
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
]

# Check Christmas sales
xmas = cal_val[cal_val['date'].str.endswith('-12-25')][['date', 'total_sales']]

# Price summary by category
prices_with_cat = df_prices.merge(df_val[['item_id', 'cat_id']].drop_duplicates(), on='item_id', how='left')
price_by_cat = prices_with_cat.groupby('cat_id')['sell_price'].describe()

print("--- CATEGORY SALES & SPARSITY ---")
for cat in cat_sales.index:
    print(f"{cat}: Total Units={cat_sales[cat]:,}, Share={cat_pct[cat]:.2f}%, Zero Days={cat_sparsity[cat]:.2f}%")

print("\n--- STATE SALES ---")
for st in state_sales.index:
    print(f"{st}: Total Units={state_sales[st]:,}, Share={state_pct[st]:.2f}%")

print("\n--- STORE SALES ---")
for idx, val in store_sales.items():
    print(f"{idx[0]} - {idx[1]}: Total Units={val:,} ({(val/cat_sales.sum())*100:.2f}%)")

print("\n--- DAY OF WEEK AVERAGE SALES (ALL STORES COMBINED) ---")
for day, val in dow_avg.items():
    print(f"{day}: {val:,.0f} units/day")

print("\n--- CHRISTMAS SALES (WALMART STORE CLOSURES) ---")
print(xmas.to_string(index=False))

print("\n--- PRICE SUMMARY BY CATEGORY ---")
print(price_by_cat[['mean', 'min', '50%', 'max']])

