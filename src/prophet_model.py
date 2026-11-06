"""
Facebook Prophet Retail Demand Forecasting Pipeline
Trains Prophet models on store and category aggregations incorporating:
- Multi-scale seasonality (Weekly, Yearly)
- Event/Holiday effects (Thanksgiving, Christmas, SuperBowl, etc.)
- Exogenous regressors (State SNAP disbursements)
- 28-day out-of-sample forecast evaluation (d_1914 to d_1941)
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
from typing import Dict, Tuple
from prophet import Prophet

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PRED_DIR = os.path.join(DATA_DIR, "predictions")
FIGURES_DIR = os.path.join("reports", "figures")
METRICS_PATH = os.path.join("reports", "prophet_evaluation_metrics.json")
PRED_OUTPUT_PATH = os.path.join(PRED_DIR, "prophet_forecast_28d.csv")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

def prepare_holiday_dataframe(df_cal: pd.DataFrame) -> pd.DataFrame:
    """Extracts holiday and special event schedule for Prophet."""
    events = df_cal[['date', 'event_name_1', 'event_type_1']].dropna(subset=['event_name_1']).copy()
    events = events.rename(columns={'date': 'ds', 'event_name_1': 'holiday'})
    events['lower_window'] = -1
    events['upper_window'] = 1
    
    # Add explicit Christmas Day store closure event
    df_cal_dt = pd.to_datetime(df_cal['date'])
    xmas = df_cal[(df_cal_dt.dt.month == 12) & (df_cal_dt.dt.day == 25)][['date']].copy()
    xmas = xmas.rename(columns={'date': 'ds'})
    xmas['holiday'] = 'Christmas_Store_Closure'
    xmas['lower_window'] = 0
    xmas['upper_window'] = 0

    combined = pd.concat([events[['holiday', 'ds', 'lower_window', 'upper_window']], xmas], ignore_index=True)
    combined['ds'] = pd.to_datetime(combined['ds'])
    return combined

def load_data():
    print("[1/5] Loading historical and evaluation telemetry...", flush=True)
    df_cal = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
    df_cal['date'] = pd.to_datetime(df_cal['date'])

    # Training data from processed store_dept_daily.parquet
    df_daily = pd.read_parquet(os.path.join(PROCESSED_DIR, "store_dept_daily.parquet"))
    df_daily['date'] = pd.to_datetime(df_daily['date'])

    # Ground truth evaluation data for validation period d_1914 to d_1941
    eval_path = os.path.join(DATA_DIR, "sales_train_evaluation.csv")
    eval_cols = ['store_id', 'cat_id', 'dept_id'] + [f'd_{i}' for i in range(1914, 1942)]
    df_eval_raw = pd.read_csv(eval_path, usecols=eval_cols)
    
    # Aggregate ground truth to store-level and total-level
    val_days = [f'd_{i}' for i in range(1914, 1942)]
    eval_store = df_eval_raw.groupby('store_id')[val_days].sum().reset_index()
    eval_store_long = pd.melt(eval_store, id_vars=['store_id'], var_name='d', value_name='y_true')
    eval_store_long = eval_store_long.merge(df_cal[['d', 'date']], on='d', how='left')

    return df_cal, df_daily, eval_store_long

def train_and_forecast_series(
    df_train: pd.DataFrame,
    holidays_df: pd.DataFrame,
    df_cal: pd.DataFrame,
    state_id: str,
    series_name: str,
    forecast_days: int = 28
) -> pd.DataFrame:
    """Trains a tuned Prophet model on a single time series with regressors."""
    # Build Prophet training dataframe
    prophet_df = df_train[['date', 'units']].rename(columns={'date': 'ds', 'units': 'y'}).copy()
    
    # Merge SNAP regressor
    snap_col = f'snap_{state_id}' if state_id in ['CA', 'TX', 'WI'] else 'snap_CA'
    snap_map = df_cal[['date', snap_col]].rename(columns={'date': 'ds', snap_col: 'snap_active'})
    prophet_df = prophet_df.merge(snap_map, on='ds', how='left')
    prophet_df['snap_active'] = prophet_df['snap_active'].fillna(0)

    # Initialize Prophet with retail-specific configurations
    m = Prophet(
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Multiplicative scales seasonality with trend growth
        changepoint_prior_scale=0.05,
        holidays_prior_scale=10.0
    )
    m.add_regressor('snap_active', mode='multiplicative')
    
    # Fit model
    m.fit(prophet_df)

    # Make future dataframe for 28-day horizon
    future = m.make_future_dataframe(periods=forecast_days, freq='D')
    future = future.merge(snap_map, on='ds', how='left')
    future['snap_active'] = future['snap_active'].fillna(0)

    # Predict
    forecast = m.predict(future)
    forecast['series_name'] = series_name
    forecast['yhat'] = forecast['yhat'].clip(lower=0)  # Demand cannot be negative
    return forecast

def evaluate_predictions(forecast_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """Computes RMSE, MAE, and WAPE across the 28-day forecast horizon."""
    # Filter 28-day forecast horizon
    pred_window = forecast_df[forecast_df['ds'] >= '2016-04-25'].copy()
    
    merged = pred_window.merge(
        ground_truth_df,
        left_on=['ds', 'series_name'],
        right_on=['date', 'store_id'],
        how='inner'
    )
    
    metrics = {}
    total_abs_err = 0.0
    total_true = 0.0
    sq_errors = []

    for store, grp in merged.groupby('series_name'):
        y_true = grp['y_true'].values
        y_pred = grp['yhat'].values
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
        'total_forecast': int(merged['yhat'].sum())
    }
    return metrics, merged

def main():
    df_cal, df_daily, eval_store_long = load_data()
    holidays_df = prepare_holiday_dataframe(df_cal)
    
    print("[2/5] Training Prophet models across all 10 Walmart stores...", flush=True)
    all_forecasts = []
    
    # Store-to-state mapping
    store_state_map = df_daily[['store_id', 'state_id']].drop_duplicates().set_index('store_id')['state_id'].to_dict()
    
    # Aggregate training data to store-daily level
    store_daily_train = df_daily.groupby(['store_id', 'date'])['units'].sum().reset_index()

    for idx, (store_id, state_id) in enumerate(store_state_map.items()):
        print(f"  -> Fitting store {store_id} ({state_id}) [{idx+1}/10]...", flush=True)
        sub_df = store_daily_train[store_daily_train['store_id'] == store_id]
        fcst = train_and_forecast_series(
            df_train=sub_df,
            holidays_df=holidays_df,
            df_cal=df_cal,
            state_id=state_id,
            series_name=store_id,
            forecast_days=28
        )
        all_forecasts.append(fcst[['ds', 'series_name', 'yhat', 'yhat_lower', 'yhat_upper']])

    combined_forecast = pd.concat(all_forecasts, ignore_index=True)

    print("[3/5] Evaluating 28-day out-of-sample forecast accuracy against ground truth...", flush=True)
    metrics, eval_merged = evaluate_predictions(combined_forecast, eval_store_long)

    # Print summary metrics table
    print("\n" + "=" * 68)
    print("PROPHET 28-DAY DEMAND FORECAST EVALUATION (d_1914 to d_1941)")
    print("=" * 68)
    print(f"{'Store / Entity':<18} | {'MAE':>10} | {'RMSE':>10} | {'WAPE (%)':>10} | {'Total Forecast':>14}")
    print("-" * 68)
    for k, v in metrics.items():
        if k == 'OVERALL_ALL_STORES':
            print("-" * 68)
        print(f"{k:<18} | {v['mae']:>10,.1f} | {v['rmse']:>10,.1f} | {v['wape_pct']:>9.2f}% | {v['total_forecast']:>14,}")
    print("=" * 68)

    # Save metrics JSON
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[4/5] Saved evaluation metrics to: {METRICS_PATH}")

    # Export 28-day forecast CSV
    pred_28d = combined_forecast[combined_forecast['ds'] >= '2016-04-25'].copy()
    pred_28d.to_csv(PRED_OUTPUT_PATH, index=False)
    print(f"Saved 28-day predictions to: {PRED_OUTPUT_PATH}")

    # Plot Forecast vs Actuals for Flagship Store CA_3 and Store TX_2
    print("[5/5] Generating publication-quality Forecast vs Actuals plots...", flush=True)
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    
    for i, target_store in enumerate(['CA_3', 'TX_2']):
        store_eval = eval_merged[eval_merged['series_name'] == target_store].sort_values('ds')
        
        # Historical last 60 days
        hist_sub = store_daily_train[
            (store_daily_train['store_id'] == target_store) & 
            (store_daily_train['date'] >= '2016-02-01')
        ]
        
        axes[i].plot(hist_sub['date'], hist_sub['units'], label='Historical Sales', color='#64748b', linewidth=1.5)
        axes[i].plot(store_eval['ds'], store_eval['y_true'], label='Ground Truth Actuals (d_1914-d_1941)', color='#0f172a', linewidth=2.2, marker='o', markersize=4)
        axes[i].plot(store_eval['ds'], store_eval['yhat'], label='Prophet 28-Day Forecast', color='#2563eb', linewidth=2.2, linestyle='--', marker='s', markersize=4)
        axes[i].fill_between(store_eval['ds'], store_eval['yhat_lower'], store_eval['yhat_upper'], color='#bfdbfe', alpha=0.5, label='80% Confidence Interval')
        
        wape = metrics[target_store]['wape_pct']
        axes[i].set_title(f'Store {target_store} Demand Forecast vs Ground Truth (WAPE: {wape:.2f}%)', fontsize=13, weight='bold')
        axes[i].set_ylabel('Daily Unit Sales')
        axes[i].legend(loc='upper left', frameon=True)
        axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    plt.xlabel('Date')
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "figure7_prophet_forecast_actuals.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved forecast figure to: {plot_path}")
    print("\n[SUCCESS] Prophet model pipeline executed successfully!")

if __name__ == "__main__":
    main()
