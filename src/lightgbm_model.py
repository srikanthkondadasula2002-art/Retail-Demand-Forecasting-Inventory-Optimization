"""
LightGBM Demand Forecasting Pipeline
Trains gradient boosted decision trees with Tweedie deviance objective for retail sales forecasting.
Features:
- Anti-leakage autoregressive lags (>= 28 days)
- Rolling window demand velocity & volatility statistics
- Calendar, holiday, weekend, and SNAP interaction regressors
- Chronological train/val/test splits
- Out-of-sample 28-day evaluation (d_1914 to d_1941) benchmarked against ground truth
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from typing import Dict, Tuple, List

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PRED_DIR = os.path.join(DATA_DIR, "predictions")
FIGURES_DIR = os.path.join("reports", "figures")
METRICS_PATH = os.path.join("reports", "lightgbm_evaluation_metrics.json")
PRED_OUTPUT_PATH = os.path.join(PRED_DIR, "lightgbm_forecast_28d.csv")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

def load_and_prepare_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads historical store-department telemetry, calendar, and evaluation ground truth."""
    print("[1/6] Ingesting telemetry and calendar metadata...", flush=True)
    df_cal = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
    df_cal['date'] = pd.to_datetime(df_cal['date'])
    
    # Store-dept daily sales
    df_daily = pd.read_parquet(os.path.join(PROCESSED_DIR, "store_dept_daily.parquet"))
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    
    # Ground truth actuals for d_1914 to d_1941
    eval_path = os.path.join(DATA_DIR, "sales_train_evaluation.csv")
    eval_cols = ['store_id', 'cat_id', 'dept_id'] + [f'd_{i}' for i in range(1914, 1942)]
    df_eval_raw = pd.read_csv(eval_path, usecols=eval_cols)
    
    # Store-level ground truth
    val_days = [f'd_{i}' for i in range(1914, 1942)]
    eval_store = df_eval_raw.groupby('store_id')[val_days].sum().reset_index()
    eval_store_long = pd.melt(eval_store, id_vars=['store_id'], var_name='d', value_name='y_true')
    eval_store_long = eval_store_long.merge(df_cal[['d', 'date']], on='d', how='left')

    return df_daily, df_cal, eval_store_long

def engineer_features(df_daily: pd.DataFrame, df_cal: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Builds anti-leakage lag and rolling features on long-format time series."""
    print("[2/6] Engineering autoregressive lags, rolling stats, and calendar features...", flush=True)
    
    # Create composite series identifier
    df_daily['series_id'] = df_daily['store_id'] + '_' + df_daily['dept_id']
    df_daily = df_daily.sort_values(['series_id', 'date']).reset_index(drop=True)

    # 1. Base Lags (>= 28 days to prevent data leakage during test inference)
    for lag in [28, 29, 30, 35, 42, 56]:
        df_daily[f'lag_{lag}'] = df_daily.groupby('series_id')['units'].shift(lag).astype('float32')

    # 2. Rolling Aggregations on Shifted Sales (shift 28)
    shifted = df_daily.groupby('series_id')['units'].shift(28)
    for w in [7, 14, 28, 90]:
        roll = shifted.groupby(df_daily['series_id']).rolling(w, min_periods=3)
        df_daily[f'roll_mean_{w}'] = roll.mean().reset_index(level=0, drop=True).astype('float32')
        df_daily[f'roll_std_{w}'] = roll.std().reset_index(level=0, drop=True).astype('float32')

    # 3. Calendar & Event Signals
    df_daily['day_of_week'] = df_daily['date'].dt.dayofweek.astype('int8')
    df_daily['day_of_month'] = df_daily['date'].dt.day.astype('int8')
    df_daily['month'] = df_daily['date'].dt.month.astype('int8')
    df_daily['year'] = df_daily['date'].dt.year.astype('int16')
    df_daily['is_weekend'] = df_daily['day_of_week'].isin([5, 6]).astype('int8')
    df_daily['is_christmas'] = ((df_daily['month'] == 12) & (df_daily['day_of_month'] == 25)).astype('int8')

    # 4. State-specific SNAP Active Indicator
    df_daily['snap_active'] = (
        (df_daily['state_id'] == 'CA') & (df_daily['snap_CA'] == 1) |
        (df_daily['state_id'] == 'TX') & (df_daily['snap_TX'] == 1) |
        (df_daily['state_id'] == 'WI') & (df_daily['snap_WI'] == 1)
    ).astype('int8')
    
    # SNAP x FOODS interaction
    df_daily['snap_foods_interaction'] = (df_daily['snap_active'] * (df_daily['cat_id'] == 'FOODS')).astype('int8')

    # 5. Categorical encodings
    for cat_col in ['store_id', 'state_id', 'cat_id', 'dept_id']:
        df_daily[cat_col] = df_daily[cat_col].astype('category')

    feature_cols = [
        'store_id', 'state_id', 'cat_id', 'dept_id',
        'lag_28', 'lag_29', 'lag_30', 'lag_35', 'lag_42', 'lag_56',
        'roll_mean_7', 'roll_mean_14', 'roll_mean_28', 'roll_mean_90',
        'roll_std_7', 'roll_std_28',
        'day_of_week', 'day_of_month', 'month', 'year', 'is_weekend', 'is_christmas',
        'snap_active', 'snap_foods_interaction'
    ]

    return df_daily, feature_cols

def prepare_forecast_matrix(df_daily: pd.DataFrame, df_cal: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Prepares the 28-day out-of-sample forecast matrix (d_1914 to d_1941) with known future calendar & lag features."""
    future_cal = df_cal[(df_cal['date'] >= '2016-04-25') & (df_cal['date'] <= '2016-05-22')].copy()
    
    # Base grid: 70 series x 28 days
    series_meta = df_daily[['series_id', 'store_id', 'state_id', 'cat_id', 'dept_id']].drop_duplicates()
    future_grid = []
    
    for _, s_row in series_meta.iterrows():
        grid = future_cal[['d', 'date', 'snap_CA', 'snap_TX', 'snap_WI']].copy()
        grid['series_id'] = s_row['series_id']
        grid['store_id'] = s_row['store_id']
        grid['state_id'] = s_row['state_id']
        grid['cat_id'] = s_row['cat_id']
        grid['dept_id'] = s_row['dept_id']
        grid['units'] = np.nan
        future_grid.append(grid)
        
    df_future = pd.concat(future_grid, ignore_index=True)
    
    # Combine with historical to compute lags into the future
    combined = pd.concat([df_daily, df_future], ignore_index=True).sort_values(['series_id', 'date']).reset_index(drop=True)
    
    # Compute lags & rollings on combined
    for lag in [28, 29, 30, 35, 42, 56]:
        combined[f'lag_{lag}'] = combined.groupby('series_id')['units'].shift(lag).astype('float32')

    shifted = combined.groupby('series_id')['units'].shift(28)
    for w in [7, 14, 28, 90]:
        roll = shifted.groupby(combined['series_id']).rolling(w, min_periods=3)
        combined[f'roll_mean_{w}'] = roll.mean().reset_index(level=0, drop=True).astype('float32')
        combined[f'roll_std_{w}'] = roll.std().reset_index(level=0, drop=True).astype('float32')

    # Calendar features
    combined['day_of_week'] = combined['date'].dt.dayofweek.astype('int8')
    combined['day_of_month'] = combined['date'].dt.day.astype('int8')
    combined['month'] = combined['date'].dt.month.astype('int8')
    combined['year'] = combined['date'].dt.year.astype('int16')
    combined['is_weekend'] = combined['day_of_week'].isin([5, 6]).astype('int8')
    combined['is_christmas'] = ((combined['month'] == 12) & (combined['day_of_month'] == 25)).astype('int8')

    combined['snap_active'] = (
        (combined['state_id'] == 'CA') & (combined['snap_CA'] == 1) |
        (combined['state_id'] == 'TX') & (combined['snap_TX'] == 1) |
        (combined['state_id'] == 'WI') & (combined['snap_WI'] == 1)
    ).astype('int8')
    combined['snap_foods_interaction'] = (combined['snap_active'] * (combined['cat_id'] == 'FOODS')).astype('int8')

    for cat_col in ['store_id', 'state_id', 'cat_id', 'dept_id']:
        combined[cat_col] = combined[cat_col].astype('category')

    test_matrix = combined[(combined['date'] >= '2016-04-25') & (combined['date'] <= '2016-05-22')].copy()
    return test_matrix

def train_lightgbm_model(
    df_daily: pd.DataFrame, 
    feature_cols: List[str]
) -> Tuple[lgb.Booster, pd.DataFrame]:
    """Fits LightGBM with Tweedie deviance objective and early stopping."""
    print("[3/6] Splitting train/validation partitions and fitting LightGBM Tweedie model...", flush=True)
    
    # Drop rows with NaN in features (due to lag_56)
    valid_data = df_daily.dropna(subset=['lag_56', 'roll_mean_90']).copy()
    
    # Chronological Split:
    # Train: start through 2016-03-27 (up to d_1885)
    # Validation: 2016-03-28 to 2016-04-24 (d_1886 to d_1913, 28 days)
    train_mask = valid_data['date'] <= '2016-03-27'
    val_mask = (valid_data['date'] > '2016-03-27') & (valid_data['date'] <= '2016-04-24')

    X_train, y_train = valid_data.loc[train_mask, feature_cols], valid_data.loc[train_mask, 'units']
    X_val, y_val = valid_data.loc[val_mask, feature_cols], valid_data.loc[val_mask, 'units']

    print(f"  -> Train sample size: {len(X_train):,} rows", flush=True)
    print(f"  -> Val sample size  : {len(X_val):,} rows", flush=True)

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=['store_id', 'state_id', 'cat_id', 'dept_id'])
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.15,
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'min_data_in_leaf': 20,
        'verbosity': -1,
        'seed': 42
    }

    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=600,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )

    print(f"  -> Best iteration: {model.best_iteration}", flush=True)
    return model, valid_data

def evaluate_and_benchmark(
    model: lgb.Booster,
    test_matrix: pd.DataFrame,
    eval_store_long: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[Dict, pd.DataFrame]:
    """Generates 28-day out-of-sample predictions and benchmarks against ground truth."""
    print("[4/6] Generating 28-day out-of-sample predictions & evaluating against actuals...", flush=True)
    
    test_matrix['pred_units'] = model.predict(test_matrix[feature_cols])
    test_matrix['pred_units'] = np.clip(test_matrix['pred_units'], 0, None)  # Demand non-negative
    
    # Store-level aggregation of predictions
    store_preds = test_matrix.groupby(['store_id', 'date'])['pred_units'].sum().reset_index()

    merged = store_preds.merge(
        eval_store_long,
        left_on=['store_id', 'date'],
        right_on=['store_id', 'date'],
        how='inner'
    )

    metrics = {}
    total_abs_err = 0.0
    total_true = 0.0
    sq_errors = []

    for store, grp in merged.groupby('store_id'):
        y_true = grp['y_true'].values
        y_pred = grp['pred_units'].values
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        wape = float((np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100)
        
        metrics[store] = {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'wape_pct': round(wape, 2),
            'total_actual': int(np.sum(y_true)),
            'total_forecast': int(np.sum(y_pred))
        }
        total_abs_err += np.sum(np.abs(y_true - y_pred))
        total_true += np.sum(y_true)
        sq_errors.extend((y_true - y_pred)**2)

    overall_mae = float(total_abs_err / len(sq_errors))
    overall_rmse = float(np.sqrt(np.mean(sq_errors)))
    overall_wape = float((total_abs_err / total_true) * 100)

    metrics['OVERALL_ALL_STORES'] = {
        'mae': round(overall_mae, 2),
        'rmse': round(overall_rmse, 2),
        'wape_pct': round(overall_wape, 2),
        'total_actual': int(total_true),
        'total_forecast': int(merged['pred_units'].sum())
    }

    return metrics, merged

def plot_visualizations(model: lgb.Booster, feature_cols: List[str], eval_merged: pd.DataFrame, df_daily: pd.DataFrame):
    """Saves Feature Importance and Forecast vs Actuals plots."""
    print("[5/6] Generating publication-quality charts...", flush=True)
    
    # 1. Feature Importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 7))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature', palette='viridis')
    plt.title('LightGBM Feature Importance (Gain)', fontsize=14, weight='bold')
    plt.xlabel('Split Gain')
    plt.ylabel('Feature')
    plt.tight_layout()
    f1_path = os.path.join(FIGURES_DIR, "figure8_lightgbm_feature_importance.png")
    plt.savefig(f1_path, dpi=200)
    plt.close()
    print(f"  -> Saved feature importance to: {f1_path}", flush=True)

    # 2. Forecast vs Actuals for CA_3 and TX_2
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    store_daily_hist = df_daily.groupby(['store_id', 'date'])['units'].sum().reset_index()

    for i, target_store in enumerate(['CA_3', 'TX_2']):
        store_eval = eval_merged[eval_merged['store_id'] == target_store].sort_values('date')
        hist_sub = store_daily_hist[
            (store_daily_hist['store_id'] == target_store) & 
            (store_daily_hist['date'] >= '2016-02-01')
        ]
        
        axes[i].plot(hist_sub['date'], hist_sub['units'], label='Historical Sales', color='#94a3b8', linewidth=1.5)
        axes[i].plot(store_eval['date'], store_eval['y_true'], label='Ground Truth Actuals', color='#0f172a', linewidth=2.2, marker='o', markersize=4)
        axes[i].plot(store_eval['date'], store_eval['pred_units'], label='LightGBM 28-Day Forecast', color='#16a34a', linewidth=2.2, linestyle='--', marker='^', markersize=4)
        
        axes[i].set_title(f'LightGBM Store {target_store} Demand Forecast vs Actuals', fontsize=13, weight='bold')
        axes[i].set_ylabel('Daily Unit Sales')
        axes[i].legend(loc='upper left', frameon=True)
        axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    plt.xlabel('Date')
    plt.tight_layout()
    f2_path = os.path.join(FIGURES_DIR, "figure9_lightgbm_forecast_actuals.png")
    plt.savefig(f2_path, dpi=200)
    plt.close()
    print(f"  -> Saved forecast plot to: {f2_path}", flush=True)

def main():
    df_daily, df_cal, eval_store_long = load_and_prepare_dataset()
    df_daily, feature_cols = engineer_features(df_daily, df_cal)
    
    test_matrix = prepare_forecast_matrix(df_daily, df_cal, feature_cols)
    model, _ = train_lightgbm_model(df_daily, feature_cols)
    
    metrics, eval_merged = evaluate_and_benchmark(model, test_matrix, eval_store_long, feature_cols)
    
    # Print scorecard
    print("\n" + "=" * 68)
    print("LIGHTGBM 28-DAY DEMAND FORECAST EVALUATION (d_1914 to d_1941)")
    print("=" * 68)
    print(f"{'Store / Entity':<18} | {'MAE':>10} | {'RMSE':>10} | {'WAPE (%)':>10} | {'Total Forecast':>14}")
    print("-" * 68)
    for k, v in metrics.items():
        if k == 'OVERALL_ALL_STORES':
            print("-" * 68)
        print(f"{k:<18} | {v['mae']:>10,.1f} | {v['rmse']:>10,.1f} | {v['wape_pct']:>9.2f}% | {v['total_forecast']:>14,}")
    print("=" * 68)

    # Save metrics JSON & predictions CSV
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[6/6] Saved metrics to: {METRICS_PATH}", flush=True)

    export_df = eval_merged[['store_id', 'date', 'pred_units', 'y_true']].rename(columns={'pred_units': 'forecast_units', 'y_true': 'actual_units'})
    export_df.to_csv(PRED_OUTPUT_PATH, index=False)
    print(f"Saved predictions to: {PRED_OUTPUT_PATH}", flush=True)

    plot_visualizations(model, feature_cols, eval_merged, df_daily)
    print("\n[SUCCESS] LightGBM forecasting pipeline successfully executed!")

if __name__ == "__main__":
    main()

