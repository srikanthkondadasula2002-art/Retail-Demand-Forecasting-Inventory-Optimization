"""
Figure 15 Generator: Demand Forecasting Dashboard Visualizations
Creates publication-grade composite figure showcasing 28-day actual vs predicted demand,
model accuracy comparisons (LightGBM vs Prophet), and residual error distributions.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PROJECT_ROOT, PROCESSED_DIR, FIGURES_DIR

# Set publication styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#2d3748",
    "axes.linewidth": 1.2,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7
})


def generate_figure15():
    """Generates Figure 15: Demand Forecasting Visualizations."""
    data_path = PROCESSED_DIR / "forecasts_integrated.parquet"
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.3, wspace=0.25)

    # -------------------------------------------------------------
    # Ax 1: Network Total 28-Day Actual vs Predicted Trajectory
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])

    # Aggregate network total by date & model
    net_lgb = df[df["model_name"] == "LightGBM_Tweedie"].groupby("date").agg({
        "actual_units": "sum",
        "forecast_units": "sum"
    }).reset_index()

    net_prp = df[df["model_name"] == "Prophet"].groupby("date").agg({
        "forecast_units": "sum",
        "lower_bound": "sum",
        "upper_bound": "sum"
    }).reset_index()

    dates = net_lgb["date"]
    ax1.plot(dates, net_lgb["actual_units"], color="#0f172a", marker="o", markersize=5, linewidth=2.5, label="Ground Truth Actual Demand")
    ax1.plot(dates, net_lgb["forecast_units"], color="#10b981", marker="s", markersize=4, linewidth=2.0, linestyle="--", label="LightGBM Tweedie Forecast (Overall WAPE: 6.80%)")
    ax1.plot(dates, net_prp["forecast_units"], color="#6366f1", marker="^", markersize=4, linewidth=2.0, linestyle="-.", label="Facebook Prophet Forecast (Overall WAPE: 7.79%)")

    # Fill confidence interval for Prophet
    ax1.fill_between(dates, net_prp["lower_bound"], net_prp["upper_bound"], color="#6366f1", alpha=0.15, label="Prophet 80% Uncertainty Interval")

    ax1.set_title("Network-Wide 28-Day Demand Forecast vs. Ground Truth Actuals (M5 Evaluation Horizon)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Forecast Evaluation Horizon (2016-04-25 to 2016-05-22)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Units Sold Across All 10 Stores", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # -------------------------------------------------------------
    # Ax 2: Store-by-Store Model Accuracy Benchmark (WAPE %)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])

    store_metrics = []
    for store in sorted(df["store_id"].unique()):
        lgb_s = df[(df["store_id"] == store) & (df["model_name"] == "LightGBM_Tweedie")]
        prp_s = df[(df["store_id"] == store) & (df["model_name"] == "Prophet")]
        act = lgb_s["actual_units"].sum()

        lgb_wape = (lgb_s["absolute_error"].sum() / act) * 100
        prp_wape = (prp_s["absolute_error"].sum() / act) * 100
        store_metrics.append({"store_id": store, "LightGBM": lgb_wape, "Prophet": prp_wape})

    df_wape = pd.DataFrame(store_metrics)
    x = np.arange(len(df_wape))
    width = 0.35

    bars1 = ax2.bar(x - width/2, df_wape["LightGBM"], width, label="LightGBM Tweedie", color="#10b981", edgecolor="#065f46")
    bars2 = ax2.bar(x + width/2, df_wape["Prophet"], width, label="Facebook Prophet", color="#6366f1", edgecolor="#4338ca")

    ax2.set_title("Model Forecast Accuracy by Store (WAPE % - Lower is Better)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_wape["store_id"], fontweight="bold")
    ax2.set_ylabel("WAPE (%)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    ax2.set_ylim(0, 16)

    # -------------------------------------------------------------
    # Ax 3: Daily Error / Residuals Over Time
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])

    lgb_err = net_lgb["forecast_units"] - net_lgb["actual_units"]
    prp_err = net_prp["forecast_units"] - net_lgb["actual_units"]

    ax3.axhline(0, color="#1e293b", linewidth=1.2, linestyle="-")
    ax3.bar(dates - pd.Timedelta(hours=4), lgb_err, width=0.35, label="LightGBM Residuals", color="#10b981", alpha=0.8)
    ax3.bar(dates + pd.Timedelta(hours=4), prp_err, width=0.35, label="Prophet Residuals", color="#6366f1", alpha=0.8)

    ax3.set_title("Daily Forecast Residuals (Predicted - Actual Units)", fontsize=13, fontweight="bold", pad=12)
    ax3.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Residual Error (Units)", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    out_fig = FIGURES_DIR / "figure15_demand_forecasting_dashboard.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"[OK] Figure 15 saved to {out_fig.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    generate_figure15()
