"""
Retail Inventory Optimization Engine
Computes Safety Stock (SS), Reorder Points (ROP), Economic Order Quantities (EOQ),
and optimal replenishment policies under stochastic demand and uncertain lead times.
Outputs recommendations to parquet and figure16_inventory_optimization_dashboard.png.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PROJECT_ROOT, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR

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


class InventoryOptimizationEngine:
    """Operations Research Engine for stochastic retail replenishment policies."""

    def __init__(self, lead_time_days: int = 7, service_level: float = 0.95,
                 order_cost: float = 50.0, holding_cost_rate: float = 0.25):
        self.lead_time = lead_time_days
        self.service_level = service_level
        self.order_cost = order_cost
        self.holding_cost_rate = holding_cost_rate
        self.output_parquet = PROCESSED_DIR / "inventory_recommendations.parquet"
        self.output_json = REPORTS_DIR / "inventory_optimization_summary.json"
        self.fig16_path = FIGURES_DIR / "figure16_inventory_optimization_dashboard.png"

    def compute_z_score(self, service_level: float) -> float:
        """Calculates standard normal quantile for target cycle service level."""
        return float(stats.norm.ppf(service_level))

    def calculate_policies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes item-level stochastic inventory policies:
        - Safety Stock: SS = Z * sigma_d * sqrt(L)
        - Lead Time Demand: LTD = d_bar * L
        - Reorder Point: ROP = LTD + SS
        - Economic Order Quantity: EOQ = sqrt(2 * D * S / H)
        """
        print(f"Calculating inventory policies (L={self.lead_time}d, CSL={self.service_level*100}%)...")
        z = self.compute_z_score(self.service_level)

        df = df.copy()
        # Protect against zeroes
        df["mean_daily_units"] = df["mean_daily_units"].clip(lower=0.01)
        df["std_daily_units"] = df["std_daily_units"].clip(lower=0.01)
        df["unit_cost"] = (df["avg_sell_price"] * 0.70).clip(lower=0.50) # 30% gross margin assumption
        df["holding_cost_annual"] = df["unit_cost"] * self.holding_cost_rate

        # 1. Safety Stock
        df["safety_stock"] = np.ceil(z * df["std_daily_units"] * np.sqrt(self.lead_time)).astype(int)

        # 2. Lead Time Demand
        df["lead_time_demand"] = np.ceil(df["mean_daily_units"] * self.lead_time).astype(int)

        # 3. Reorder Point (ROP)
        df["reorder_point"] = df["lead_time_demand"] + df["safety_stock"]

        # 4. Annual Demand & EOQ
        df["annual_demand"] = df["mean_daily_units"] * 365.25
        df["eoq_units"] = np.ceil(np.sqrt((2 * df["annual_demand"] * self.order_cost) / df["holding_cost_annual"])).astype(int)

        # 5. Annual Costs
        df["annual_ordering_cost"] = (df["annual_demand"] / df["eoq_units"]) * self.order_cost
        df["annual_holding_cost"] = ((df["eoq_units"] / 2) + df["safety_stock"]) * df["holding_cost_annual"]
        df["annual_total_cost"] = df["annual_ordering_cost"] + df["annual_holding_cost"]
        df["orders_per_year"] = np.round(df["annual_demand"] / df["eoq_units"], 1)

        return df

    def generate_visualizations(self, df: pd.DataFrame):
        """Generates Figure 16: Multi-panel inventory optimization composite."""
        print("Generating Figure 16: Inventory Optimization Dashboard...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

        # Panel 1: Lead Time Demand vs Safety Stock Buffer by Department
        dept_agg = df.groupby(["cat_id", "dept_id"]).agg({
            "lead_time_demand": "sum",
            "safety_stock": "sum"
        }).reset_index().sort_values("lead_time_demand", ascending=True)

        y = np.arange(len(dept_agg))
        ax1.barh(y, dept_agg["lead_time_demand"], color="#3b82f6", label="Lead Time Demand (LTD)")
        ax1.barh(y, dept_agg["safety_stock"], left=dept_agg["lead_time_demand"], color="#10b981", label="Safety Stock Buffer (SS)")
        ax1.set_yticks(y)
        ax1.set_yticklabels(dept_agg["dept_id"], fontweight="bold")
        ax1.set_title("Reorder Point Composition (LTD + Safety Stock) by Department", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel("Units", fontsize=10, fontweight="bold")
        ax1.legend(loc="lower right")

        # Panel 2: Service Level vs Safety Stock Multiplier Curve
        csl_range = np.linspace(0.80, 0.999, 100)
        z_curve = [stats.norm.ppf(c) for c in csl_range]
        base_std = df["std_daily_units"].median() * np.sqrt(self.lead_time)
        ss_curve = [z * base_std for z in z_curve]

        ax2.plot(csl_range * 100, ss_curve, color="#7c3aed", linewidth=2.5)
        ax2.axvline(95, color="#10b981", linestyle="--", label="Target: 95% (Z=1.65)")
        ax2.axvline(98, color="#f59e0b", linestyle="--", label="Target: 98% (Z=2.05)")
        ax2.axvline(99, color="#ef4444", linestyle="--", label="Target: 99% (Z=2.33)")
        ax2.set_title("Safety Stock Non-Linear Growth vs. Target Service Level", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Cycle Service Level (%)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Safety Stock Units (Median SKU)", fontsize=10, fontweight="bold")
        ax2.legend(loc="upper left")

        # Panel 3: EOQ Cost Trade-off Curves for Representative Item
        sample_item = df[df["mean_daily_units"] > 3].iloc[0]
        q_range = np.linspace(5, max(100, int(sample_item["eoq_units"] * 3)), 100)
        d_ann = sample_item["annual_demand"]
        h_cost = sample_item["holding_cost_annual"]
        s_cost = self.order_cost

        ord_costs = (d_ann / q_range) * s_cost
        hld_costs = (q_range / 2) * h_cost
        tot_costs = ord_costs + hld_costs
        eoq_opt = sample_item["eoq_units"]
        min_cost = np.min(tot_costs)

        ax3.plot(q_range, ord_costs, color="#ef4444", linestyle="--", label="Annual Ordering Cost (D/Q * S)")
        ax3.plot(q_range, hld_costs, color="#3b82f6", linestyle="--", label="Annual Holding Cost (Q/2 * H)")
        ax3.plot(q_range, tot_costs, color="#0f172a", linewidth=2.5, label="Total Annual Policy Cost")
        ax3.axvline(eoq_opt, color="#10b981", linestyle=":", linewidth=2, label=f"Optimal EOQ ({eoq_opt} units)")
        ax3.scatter([eoq_opt], [min_cost], color="#10b981", s=80, zorder=5)
        ax3.set_title(f"Economic Order Quantity (EOQ) Cost Trade-Off: {sample_item['item_id']}", fontsize=12, fontweight="bold", pad=10)
        ax3.set_xlabel("Order Quantity (Q)", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Annual Cost ($)", fontsize=10, fontweight="bold")
        ax3.set_ylim(0, min_cost * 2.5)
        ax3.legend(loc="upper right")

        # Panel 4: Distribution of Recommended Order Frequency
        orders_clipped = df["orders_per_year"].clip(upper=52)
        ax4.hist(orders_clipped, bins=25, color="#0284c7", edgecolor="#0369a1", alpha=0.85)
        ax4.axvline(orders_clipped.median(), color="#ef4444", linestyle="--", linewidth=1.8,
                    label=f"Median Frequency: {orders_clipped.median():.1f} orders/year")
        ax4.set_title("Distribution of Annual Replenishment Order Frequencies", fontsize=12, fontweight="bold", pad=10)
        ax4.set_xlabel("Replenishment Orders per Year", fontsize=10, fontweight="bold")
        ax4.set_ylabel("Number of SKUs", fontsize=10, fontweight="bold")
        ax4.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(self.fig16_path, dpi=300)
        plt.close()
        print(f"[OK] Figure 16 saved to {self.fig16_path.name}")

    def export_summary(self, df: pd.DataFrame):
        """Exports parquet dataset and JSON summary."""
        df.to_parquet(self.output_parquet, index=False)
        print(f"[OK] Saved {len(df):,} inventory recommendations to {self.output_parquet.name}")

        summary = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "parameters": {
                "lead_time_days": self.lead_time,
                "target_service_level": self.service_level,
                "order_setup_cost_usd": self.order_cost,
                "annual_holding_cost_rate": self.holding_cost_rate,
                "z_score": round(self.compute_z_score(self.service_level), 3)
            },
            "network_totals": {
                "total_series_evaluated": len(df),
                "total_safety_stock_units": int(df["safety_stock"].sum()),
                "total_lead_time_demand_units": int(df["lead_time_demand"].sum()),
                "total_reorder_point_units": int(df["reorder_point"].sum()),
                "total_annual_holding_cost_usd": round(float(df["annual_holding_cost"].sum()), 2),
                "total_annual_ordering_cost_usd": round(float(df["annual_ordering_cost"].sum()), 2),
                "total_annual_inventory_cost_usd": round(float(df["annual_total_cost"].sum()), 2),
                "median_eoq_order_units": int(df["eoq_units"].median()),
                "median_orders_per_year": round(float(df["orders_per_year"].median()), 1)
            },
            "department_breakdown": df.groupby("dept_id").agg({
                "safety_stock": "sum",
                "reorder_point": "sum",
                "annual_total_cost": "sum"
            }).round(2).to_dict(orient="index")
        }

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Saved inventory optimization summary to {self.output_json.relative_to(PROJECT_ROOT)}")

    def run(self):
        """Execute inventory optimization calculations."""
        print("=" * 60)
        print("STARTING RETAIL INVENTORY OPTIMIZATION ENGINE")
        print("=" * 60)
        perf_path = PROCESSED_DIR / "product_performance.parquet"
        sum_path = PROCESSED_DIR / "clean_sales_summary.parquet"

        df_perf = pd.read_parquet(perf_path)
        df_sum = pd.read_parquet(sum_path)

        df = pd.merge(
            df_perf,
            df_sum[["item_id", "store_id", "mean_daily_units", "std_daily_units"]],
            on=["item_id", "store_id"],
            how="left"
        )

        df_policies = self.calculate_policies(df)
        self.generate_visualizations(df_policies)
        self.export_summary(df_policies)
        print("=" * 60)
        print("[SUCCESS] Inventory Optimization Pipeline Completed Successfully!")
        print("=" * 60)


if __name__ == "__main__":
    optimizer = InventoryOptimizationEngine()
    optimizer.run()
