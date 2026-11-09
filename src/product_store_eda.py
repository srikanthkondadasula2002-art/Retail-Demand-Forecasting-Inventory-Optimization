"""
Product and Store Exploratory Data Analysis (EDA)
Implements ABC Inventory Classification (Pareto 80/20 analysis),
Product Velocity Clustering (Fast, Medium, Slow, Lumpy),
and Store Performance Profiling with publication-grade visualizations.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
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


class ProductStoreEDA:
    """Exploratory Data Analysis for Product Classification and Store Performance."""

    def __init__(self):
        self.perf_path = PROCESSED_DIR / "product_performance.parquet"
        self.summary_path = PROCESSED_DIR / "clean_sales_summary.parquet"
        self.output_json = REPORTS_DIR / "abc_classification_summary.json"
        self.fig10_path = FIGURES_DIR / "figure10_abc_pareto_analysis.png"
        self.fig11_path = FIGURES_DIR / "figure11_product_velocity_clusters.png"

    def load_data(self) -> pd.DataFrame:
        """Loads and merges product performance with clean summary stats."""
        print("Loading product performance and summary datasets...")
        df_perf = pd.read_parquet(self.perf_path)
        df_sum = pd.read_parquet(self.summary_path)

        df = pd.merge(
            df_perf,
            df_sum[["item_id", "store_id", "mean_daily_units", "std_daily_units", "zero_days_ratio"]],
            on=["item_id", "store_id"],
            how="left"
        )
        # Compute Coefficient of Variation (CV = std / mean, with epsilon protection)
        df["cv_demand"] = np.where(df["mean_daily_units"] > 0, df["std_daily_units"] / df["mean_daily_units"], 0.0)
        return df

    def perform_abc_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs Pareto ABC Inventory Classification by total revenue:
        - Class A: Top 80% of cumulative revenue
        - Class B: Next 15% (80% to 95%) of cumulative revenue
        - Class C: Bottom 5% (95% to 100%) of cumulative revenue
        """
        print("Performing ABC Pareto Classification...")
        df_sorted = df.sort_values(by="est_revenue", ascending=False).reset_index(drop=True)
        total_rev = df_sorted["est_revenue"].sum()
        df_sorted["cum_revenue"] = df_sorted["est_revenue"].cumsum()
        df_sorted["cum_rev_pct"] = df_sorted["cum_revenue"] / total_rev * 100

        conditions = [
            (df_sorted["cum_rev_pct"] <= 80.0),
            (df_sorted["cum_rev_pct"] <= 95.0)
        ]
        choices = ["Class A", "Class B"]
        df_sorted["abc_class"] = np.select(conditions, choices, default="Class C")
        return df_sorted

    def perform_velocity_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies items into velocity clusters based on zero-demand ratio and volume:
        - Fast: zero_days_ratio < 0.25 (active demand on >= 75% of days)
        - Medium: 0.25 <= zero_days_ratio < 0.60
        - Slow: 0.60 <= zero_days_ratio < 0.85
        - Lumpy / Intermittent: zero_days_ratio >= 0.85
        """
        print("Classifying Product Velocity Clusters...")
        conditions = [
            (df["zero_days_ratio"] < 0.25),
            (df["zero_days_ratio"] < 0.60),
            (df["zero_days_ratio"] < 0.85)
        ]
        choices = ["Fast", "Medium", "Slow"]
        df["velocity_cluster"] = np.select(conditions, choices, default="Lumpy / Intermittent")
        return df

    def generate_visualizations(self, df: pd.DataFrame):
        """Generates Figure 10 (Pareto ABC) and Figure 11 (Velocity Clusters)."""
        print("Generating Figure 10: ABC Pareto Analysis...")
        # Figure 10: Pareto Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Ax 1: Cumulative Revenue Curve
        n_items = len(df)
        x_pct = np.linspace(0, 100, n_items)
        y_rev_pct = df["cum_rev_pct"].values

        ax1.plot(x_pct, y_rev_pct, color="#1e3a8a", linewidth=2.5, label="Cumulative Revenue %")
        ax1.axhline(80, color="#059669", linestyle="--", linewidth=1.5, label="80% Threshold (Class A)")
        ax1.axhline(95, color="#d97706", linestyle="--", linewidth=1.5, label="95% Threshold (Class B)")

        # Find SKU threshold percentages
        sku_pct_a = (df["abc_class"] == "Class A").mean() * 100
        sku_pct_ab = ((df["abc_class"] == "Class A") | (df["abc_class"] == "Class B")).mean() * 100

        ax1.axvline(sku_pct_a, color="#059669", linestyle=":", linewidth=1.2)
        ax1.axvline(sku_pct_ab, color="#d97706", linestyle=":", linewidth=1.2)

        ax1.set_title("Pareto 80/20 Revenue Distribution Curve", fontsize=13, fontweight="bold", pad=12)
        ax1.set_xlabel("% of Total SKUs (Sorted by Revenue)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Cumulative Revenue %", fontsize=11, fontweight="bold")
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 102)
        ax1.legend(loc="lower right", frameon=True)
        ax1.annotate(f"Class A: {sku_pct_a:.1f}% SKUs -> 80% Rev",
                     xy=(sku_pct_a, 80), xytext=(sku_pct_a + 5, 65),
                     arrowprops=dict(facecolor="#059669", shrink=0.05, width=1.2, headwidth=6),
                     fontweight="bold", color="#059669")

        # Ax 2: Class SKU Share vs Revenue Share
        abc_agg = df.groupby("abc_class").agg(
            sku_count=("item_id", "count"),
            total_revenue=("est_revenue", "sum"),
            total_units=("total_units", "sum")
        ).reset_index()
        abc_agg["sku_pct"] = abc_agg["sku_count"] / abc_agg["sku_count"].sum() * 100
        abc_agg["rev_pct"] = abc_agg["total_revenue"] / abc_agg["total_revenue"].sum() * 100

        bar_width = 0.35
        x = np.arange(len(abc_agg))
        bars1 = ax2.bar(x - bar_width/2, abc_agg["sku_pct"], bar_width, label="SKU Count %", color="#64748b")
        bars2 = ax2.bar(x + bar_width/2, abc_agg["rev_pct"], bar_width, label="Revenue %", color="#2563eb")

        ax2.set_title("ABC Classes: SKU Proportion vs. Revenue Contribution", fontsize=13, fontweight="bold", pad=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(abc_agg["abc_class"], fontweight="bold")
        ax2.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right", frameon=True)

        for bar in bars1:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=10)
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.fig10_path, dpi=300)
        plt.close()
        print(f"[OK] Figure 10 saved to {self.fig10_path.name}")

        # Figure 11: Product Velocity Clusters
        print("Generating Figure 11: Product Velocity Clusters...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        palette = {
            "Fast": "#10b981",
            "Medium": "#3b82f6",
            "Slow": "#f59e0b",
            "Lumpy / Intermittent": "#ef4444"
        }

        # Subsample 4,000 points for crisp visual clarity in scatter
        sample_df = df.sample(n=min(4000, len(df)), random_state=42)
        sns.scatterplot(
            data=sample_df,
            x="zero_days_ratio",
            y="mean_daily_units",
            hue="velocity_cluster",
            palette=palette,
            alpha=0.6,
            s=25,
            ax=ax1
        )
        ax1.set_yscale("log")
        ax1.set_title("Demand Sparsity vs. Velocity (Log Mean Daily Units)", fontsize=13, fontweight="bold", pad=12)
        ax1.set_xlabel("Zero-Demand Days Ratio", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Mean Daily Units (Log Scale)", fontsize=11, fontweight="bold")
        ax1.legend(title="Velocity Cluster", loc="upper right", frameon=True)

        # Bar chart: Velocity cluster counts and revenue
        vel_agg = df.groupby("velocity_cluster").agg(
            sku_count=("item_id", "count"),
            rev_sum=("est_revenue", "sum")
        ).reindex(["Fast", "Medium", "Slow", "Lumpy / Intermittent"]).reset_index()
        vel_agg["sku_pct"] = vel_agg["sku_count"] / vel_agg["sku_count"].sum() * 100

        colors = [palette[c] for c in vel_agg["velocity_cluster"]]
        bars = ax2.bar(vel_agg["velocity_cluster"], vel_agg["sku_pct"], color=colors, edgecolor="#1e293b", linewidth=1.1)
        ax2.set_title("SKU Share by Velocity Cluster", fontsize=13, fontweight="bold", pad=12)
        ax2.set_ylabel("% of Total Series", fontsize=11, fontweight="bold")
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.fig11_path, dpi=300)
        plt.close()
        print(f"[OK] Figure 11 saved to {self.fig11_path.name}")

    def generate_summary_json(self, df: pd.DataFrame):
        """Generates comprehensive JSON summary of ABC and velocity metrics."""
        print("Exporting ABC and Store Performance Summary...")
        total_rev = float(df["est_revenue"].sum())
        total_units = int(df["total_units"].sum())
        total_skus = len(df)

        abc_summary = {}
        for c in ["Class A", "Class B", "Class C"]:
            sub = df[df["abc_class"] == c]
            abc_summary[c] = {
                "sku_count": len(sub),
                "sku_percentage": round(len(sub) / total_skus * 100, 2),
                "revenue_sum": round(float(sub["est_revenue"].sum()), 2),
                "revenue_percentage": round(float(sub["est_revenue"].sum()) / total_rev * 100, 2),
                "units_sum": int(sub["total_units"].sum()),
                "units_percentage": round(int(sub["total_units"].sum()) / total_units * 100, 2),
                "avg_unit_price": round(float(sub["est_revenue"].sum() / sub["total_units"].sum()), 2)
            }

        velocity_summary = {}
        for v in ["Fast", "Medium", "Slow", "Lumpy / Intermittent"]:
            sub = df[df["velocity_cluster"] == v]
            velocity_summary[v] = {
                "series_count": len(sub),
                "series_percentage": round(len(sub) / total_skus * 100, 2),
                "avg_zero_ratio": round(float(sub["zero_days_ratio"].mean()), 4),
                "avg_daily_units": round(float(sub["mean_daily_units"].mean()), 2),
                "avg_cv_demand": round(float(sub["cv_demand"].median()), 2)
            }

        store_summary = {}
        store_agg = df.groupby(["store_id", "state_id"]).agg(
            revenue=("est_revenue", "sum"),
            units=("total_units", "sum"),
            avg_sparsity=("sparsity_pct", "mean")
        ).reset_index().sort_values(by="revenue", ascending=False)

        for _, row in store_agg.iterrows():
            store_summary[row["store_id"]] = {
                "state": row["state_id"],
                "revenue": round(float(row["revenue"]), 2),
                "revenue_share_pct": round(float(row["revenue"]) / total_rev * 100, 2),
                "total_units": int(row["units"]),
                "average_sparsity_pct": round(float(row["avg_sparsity"]), 2)
            }

        summary_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_series": total_skus,
            "total_revenue": round(total_rev, 2),
            "total_units": total_units,
            "abc_classification": abc_summary,
            "velocity_clusters": velocity_summary,
            "store_rankings": store_summary
        }

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"[OK] Summary JSON saved to {self.output_json.relative_to(PROJECT_ROOT)}")

    def run(self):
        """Execute full EDA analysis."""
        print("=" * 60)
        print("STARTING PRODUCT & STORE EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        df = self.load_data()
        df = self.perform_abc_classification(df)
        df = self.perform_velocity_clustering(df)
        self.generate_visualizations(df)
        self.generate_summary_json(df)
        print("=" * 60)
        print("[SUCCESS] Product & Store EDA completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    eda = ProductStoreEDA()
    eda.run()
