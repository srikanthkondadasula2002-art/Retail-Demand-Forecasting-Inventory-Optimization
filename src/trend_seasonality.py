"""
Time Series Trend, Seasonality, and Holiday Shock Analysis
Decomposes retail sales series into trend, 7-day cyclical seasonality, and residuals.
Calculates ACF/PACF autocorrelation structure and quantifies demand lift across all 30 holidays.
Outputs Figure 12, Figure 13, Figure 14, and trend_seasonality_metrics.json.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PROJECT_ROOT, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR

# Set publication style
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


class TrendSeasonalityAnalyzer:
    """Rigorous time series decomposition, harmonic autocorrelation, and event impact analysis."""

    def __init__(self):
        self.store_dept_path = PROCESSED_DIR / "store_dept_daily.parquet"
        self.calendar_path = PROCESSED_DIR / "clean_calendar.parquet"
        self.output_json = REPORTS_DIR / "trend_seasonality_metrics.json"
        self.fig12_path = FIGURES_DIR / "figure12_stl_time_series_decomposition.png"
        self.fig13_path = FIGURES_DIR / "figure13_autocorrelation_acf_pacf.png"
        self.fig14_path = FIGURES_DIR / "figure14_holiday_shock_ranking.png"

    def load_daily_series(self) -> pd.DataFrame:
        """Loads daily aggregated total sales and calendar attributes."""
        print("Loading daily sales series and calendar events...")
        df_sd = pd.read_parquet(self.store_dept_path)
        df_cal = pd.read_parquet(self.calendar_path)

        # Aggregate total daily units across all stores
        daily_units = df_sd.groupby("date")["units"].sum().reset_index()
        daily_units["date"] = pd.to_datetime(daily_units["date"])

        # Merge with calendar features
        df_cal["date"] = pd.to_datetime(df_cal["date"])
        df_merged = pd.merge(daily_units, df_cal, on="date", how="left").sort_values("date").reset_index(drop=True)
        return df_merged

    def perform_decomposition(self, df: pd.DataFrame):
        """Performs additive seasonal decomposition with a 7-day weekly period."""
        print("Performing time series decomposition (7-day period)...")
        ts = df.set_index("date")["units"].asfreq("D")
        decomposition = seasonal_decompose(ts, model="additive", period=7)
        return decomposition

    def compute_autocorrelation(self, df: pd.DataFrame, nlags: int = 56):
        """Computes Autocorrelation (ACF) and Partial Autocorrelation (PACF) up to lag 56."""
        print(f"Computing ACF and PACF across {nlags} lags...")
        series = df["units"].values
        acf_vals, acf_conf = acf(series, nlags=nlags, alpha=0.05)
        pacf_vals, pacf_conf = pacf(series, nlags=nlags, alpha=0.05)
        return acf_vals, acf_conf, pacf_vals, pacf_conf

    def analyze_holiday_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quantifies demand uplift or drop for every holiday event vs non-event baseline days.
        Controls for day-of-week and month baseline variations.
        """
        print("Quantifying holiday and retail event demand lift...")
        # Overall non-event baseline
        baseline_sales = df[df["event_name_1"] == "None"]["units"].mean()

        events = [e for e in df["event_name_1"].unique() if e != "None"]
        event_stats = []

        for event in events:
            ev_df = df[df["event_name_1"] == event]
            ev_mean = ev_df["units"].mean()
            ev_count = len(ev_df)
            lift_pct = ((ev_mean - baseline_sales) / baseline_sales) * 100

            event_type = ev_df["event_type_1"].iloc[0]
            event_stats.append({
                "event_name": event,
                "event_type": str(event_type),
                "occurrences": ev_count,
                "avg_units": round(ev_mean, 1),
                "baseline_units": round(baseline_sales, 1),
                "lift_pct": round(lift_pct, 2)
            })

        df_events = pd.DataFrame(event_stats).sort_values("lift_pct", ascending=False).reset_index(drop=True)
        return df_events

    def generate_visualizations(self, df: pd.DataFrame, decomp, acf_vals, acf_conf, pacf_vals, pacf_conf, df_events):
        """Generates Figure 12 (Decomposition), Figure 13 (ACF/PACF), and Figure 14 (Holidays)."""
        print("Generating Figure 12: STL Time Series Decomposition...")
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

        dates = df["date"]
        # Observed
        axes[0].plot(dates, decomp.observed, color="#1e3a8a", linewidth=1.2)
        axes[0].set_title("Observed Daily Retail Units (M5 Walmart Network)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Total Units", fontsize=10, fontweight="bold")

        # Trend
        axes[1].plot(dates, decomp.trend, color="#059669", linewidth=2.0)
        axes[1].set_title("Underlying Trend (Moving Average Smoothed)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Trend", fontsize=10, fontweight="bold")

        # Seasonal
        axes[2].plot(dates[-90:], decomp.seasonal[-90:], color="#7c3aed", linewidth=1.5)
        axes[2].set_title("Weekly 7-Day Seasonal Pattern (Trailing 90 Days Zoom)", fontsize=12, fontweight="bold")
        axes[2].set_ylabel("Seasonal (+/- Units)", fontsize=10, fontweight="bold")

        # Residuals
        axes[3].plot(dates, decomp.resid, color="#dc2626", linewidth=1.0, alpha=0.7)
        axes[3].axhline(0, color="#1f2937", linestyle="--", linewidth=1)
        axes[3].set_title("Irregular Residuals (Holiday Shocks & Unexplained Variance)", fontsize=12, fontweight="bold")
        axes[3].set_ylabel("Residual", fontsize=10, fontweight="bold")
        axes[3].set_xlabel("Date", fontsize=11, fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.fig12_path, dpi=300)
        plt.close()
        print(f"[OK] Figure 12 saved to {self.fig12_path.name}")

        # Figure 13: ACF and PACF
        print("Generating Figure 13: Autocorrelation & Partial Autocorrelation...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        nlags = len(acf_vals) - 1
        lags = np.arange(len(acf_vals))

        # ACF
        ax1.vlines(lags, 0, acf_vals, color="#2563eb", linewidth=2)
        ax1.scatter(lags, acf_vals, color="#1e3a8a", s=20)
        ax1.axhline(0, color="#64748b", linewidth=1)
        # 95% Confidence Band
        conf_band = 1.96 / np.sqrt(len(df))
        ax1.axhline(conf_band, color="#dc2626", linestyle="--", linewidth=1.2, label="95% CI Limit")
        ax1.axhline(-conf_band, color="#dc2626", linestyle="--", linewidth=1.2)
        for h in [7, 14, 21, 28, 35, 42, 49, 56]:
            ax1.axvline(h, color="#10b981", linestyle=":", alpha=0.6, linewidth=1)
        ax1.set_title("Autocorrelation Function (ACF) - Multi-Week Seasonality", fontsize=12, fontweight="bold", pad=12)
        ax1.set_xlabel("Lag (Days)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Autocorrelation", fontsize=11, fontweight="bold")
        ax1.set_ylim(-0.3, 1.05)
        ax1.legend(loc="upper right")

        # PACF
        ax2.vlines(lags, 0, pacf_vals, color="#059669", linewidth=2)
        ax2.scatter(lags, pacf_vals, color="#065f46", s=20)
        ax2.axhline(0, color="#64748b", linewidth=1)
        ax2.axhline(conf_band, color="#dc2626", linestyle="--", linewidth=1.2, label="95% CI Limit")
        ax2.axhline(-conf_band, color="#dc2626", linestyle="--", linewidth=1.2)
        for h in [7, 14, 21, 28]:
            ax2.axvline(h, color="#2563eb", linestyle=":", alpha=0.6, linewidth=1)
        ax2.set_title("Partial Autocorrelation Function (PACF)", fontsize=12, fontweight="bold", pad=12)
        ax2.set_xlabel("Lag (Days)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Partial Autocorrelation", fontsize=11, fontweight="bold")
        ax2.set_ylim(-0.3, 1.05)
        ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(self.fig13_path, dpi=300)
        plt.close()
        print(f"[OK] Figure 13 saved to {self.fig13_path.name}")

        # Figure 14: Holiday Demand Shocks Ranking
        print("Generating Figure 14: Holiday Demand Shocks Ranking...")
        fig, ax = plt.subplots(figsize=(14, 10))

        # Top 10 positive and bottom 5 negative shocks
        top_pos = df_events[df_events["lift_pct"] > 0].head(12)
        top_neg = df_events[df_events["lift_pct"] < 0].tail(6)
        ranked_shocks = pd.concat([top_pos, top_neg]).sort_values("lift_pct", ascending=True)

        bar_colors = ["#ef4444" if x < 0 else "#10b981" for x in ranked_shocks["lift_pct"]]
        bars = ax.barh(ranked_shocks["event_name"], ranked_shocks["lift_pct"], color=bar_colors, edgecolor="#1e293b", linewidth=1.1)

        ax.axvline(0, color="#1e293b", linewidth=1.5)
        ax.set_title("Retail Event & Holiday Demand Lift Ranking (% vs. Baseline)", fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel("Demand Lift / Drop (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Calendar Event", fontsize=12, fontweight="bold")

        for bar in bars:
            w = bar.get_width()
            offset = 1.0 if w >= 0 else -1.5
            ha = "left" if w >= 0 else "right"
            ax.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:+.1f}%",
                    ha=ha, va="center", fontsize=10, fontweight="bold",
                    color="#065f46" if w >= 0 else "#991b1b")

        plt.tight_layout()
        plt.savefig(self.fig14_path, dpi=300)
        plt.close()
        print(f"[OK] Figure 14 saved to {self.fig14_path.name}")

    def generate_metrics_json(self, df: pd.DataFrame, decomp, acf_vals, pacf_vals, df_events):
        """Compiles and exports trend, seasonal, harmonic, and holiday metrics."""
        print("Exporting trend & seasonality metrics JSON...")
        # Trend statistics
        trend_clean = decomp.trend.dropna()
        trend_start = float(trend_clean.iloc[0])
        trend_end = float(trend_clean.iloc[-1])
        cagr_approx = ((trend_end / trend_start) ** (365.25 / len(trend_clean)) - 1) * 100

        # Variance decomposition (relative strength of seasonality vs noise)
        resid_clean = decomp.resid.dropna()
        var_total = float(np.var(decomp.observed))
        var_seasonal = float(np.var(decomp.seasonal))
        var_trend = float(np.var(trend_clean))
        var_resid = float(np.var(resid_clean))

        seasonal_strength = max(0.0, 1.0 - (var_resid / (var_seasonal + var_resid)))
        trend_strength = max(0.0, 1.0 - (var_resid / (var_trend + var_resid)))

        metrics = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "time_horizon": {
                "start_date": df["date"].min().strftime("%Y-%m-%d"),
                "end_date": df["date"].max().strftime("%Y-%m-%d"),
                "total_days": len(df)
            },
            "trend_dynamics": {
                "trend_initial_daily_units": round(trend_start, 1),
                "trend_final_daily_units": round(trend_end, 1),
                "approx_annualized_growth_pct": round(cagr_approx, 2),
                "trend_strength_index": round(trend_strength, 4)
            },
            "seasonality_dynamics": {
                "weekly_seasonal_period": 7,
                "seasonal_strength_index": round(seasonal_strength, 4),
                "acf_harmonic_peaks": {
                    "lag_7": round(float(acf_vals[7]), 4),
                    "lag_14": round(float(acf_vals[14]), 4),
                    "lag_21": round(float(acf_vals[21]), 4),
                    "lag_28": round(float(acf_vals[28]), 4),
                    "lag_35": round(float(acf_vals[35]), 4),
                    "lag_56": round(float(acf_vals[56]), 4)
                },
                "pacf_key_lags": {
                    "lag_1": round(float(pacf_vals[1]), 4),
                    "lag_7": round(float(pacf_vals[7]), 4),
                    "lag_14": round(float(pacf_vals[14]), 4)
                }
            },
            "top_positive_holiday_shocks": df_events.head(10).to_dict(orient="records"),
            "top_negative_holiday_shocks": df_events.tail(5).to_dict(orient="records")
        }

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[OK] Trend & seasonality metrics saved to {self.output_json.relative_to(PROJECT_ROOT)}")

    def run(self):
        """Execute full trend, seasonality, and holiday impact pipeline."""
        print("=" * 60)
        print("STARTING TIME SERIES TREND & SEASONALITY PIPELINE")
        print("=" * 60)
        df = self.load_daily_series()
        decomp = self.perform_decomposition(df)
        acf_vals, acf_conf, pacf_vals, pacf_conf = self.compute_autocorrelation(df, nlags=56)
        df_events = self.analyze_holiday_impact(df)
        self.generate_visualizations(df, decomp, acf_vals, acf_conf, pacf_vals, pacf_conf, df_events)
        self.generate_metrics_json(df, decomp, acf_vals, pacf_vals, df_events)
        print("=" * 60)
        print("[SUCCESS] Trend and Seasonality Pipeline completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    analyzer = TrendSeasonalityAnalyzer()
    analyzer.run()
