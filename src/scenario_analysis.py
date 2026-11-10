"""
What-If Scenario Analysis: Retail Price Drop Simulation
Simulates demand response, revenue impact, and supply chain inventory strain
under price reductions (-10% baseline scenario, and -30% to +10% sensitivity sweeps).
Outputs summary to reports/price_drop_simulation_10pct.json and Figure 17.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
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


class PriceScenarioSimulator:
    """Simulates retail price elasticity and demand response across product categories."""

    def __init__(self, baseline_price_drop_pct: float = -10.0):
        self.price_drop_pct = baseline_price_drop_pct
        # Empirical retail category price elasticities
        self.elasticities = {
            "FOODS": -0.85,       # Inelastic grocery staples
            "HOUSEHOLD": -1.15,   # Moderately elastic essentials
            "HOBBIES": -1.45      # Elastic discretionary merchandise
        }
        self.output_json = REPORTS_DIR / "price_drop_simulation_10pct.json"
        self.fig17_path = FIGURES_DIR / "figure17_price_drop_scenario_simulation.png"

    def simulate_price_change(self, df_prod: pd.DataFrame, price_pct_change: float) -> pd.DataFrame:
        """
        Calculates new demand and revenue given a price change percentage:
        % Delta Q = Elasticity * % Delta P
        Q_new = Q_base * (1 + % Delta Q)
        Rev_new = Q_new * P_new
        """
        df = df_prod.copy()
        delta_p_ratio = price_pct_change / 100.0

        # Map elasticity
        df["elasticity"] = df["cat_id"].map(self.elasticities).fillna(-1.0)
        df["demand_lift_pct"] = df["elasticity"] * price_pct_change
        df["simulated_units"] = np.round(df["total_units"] * (1.0 + (df["demand_lift_pct"] / 100.0))).astype(int)

        df["baseline_price"] = df["avg_sell_price"]
        df["simulated_price"] = df["baseline_price"] * (1.0 + delta_p_ratio)

        df["baseline_revenue"] = df["est_revenue"]
        df["simulated_revenue"] = df["simulated_units"] * df["simulated_price"]
        df["revenue_delta"] = df["simulated_revenue"] - df["baseline_revenue"]
        df["revenue_pct_change"] = (df["revenue_delta"] / df["baseline_revenue"]) * 100

        # Inventory strain: safety stock required under new demand
        z = 1.645 # 95% CSL
        lead_time = 7
        # Std scales with square root of volume lift
        lift_multiplier = np.sqrt(1.0 + np.maximum(0, df["demand_lift_pct"] / 100.0))
        df["additional_ss_units"] = np.ceil(z * (df["mean_daily_units"] * 0.8) * np.sqrt(lead_time) * (lift_multiplier - 1.0)).clip(lower=0).astype(int)

        return df

    def run_sensitivity_sweep(self, df_prod: pd.DataFrame) -> pd.DataFrame:
        """Evaluates revenue and volume across price adjustments from -30% to +10%."""
        price_grid = np.linspace(-30, 10, 41)
        sweep_records = []

        base_units = df_prod["total_units"].sum()
        base_rev = df_prod["est_revenue"].sum()

        for p_chg in price_grid:
            sim = self.simulate_price_change(df_prod, p_chg)
            sweep_records.append({
                "price_change_pct": p_chg,
                "total_units": sim["simulated_units"].sum(),
                "unit_lift_pct": ((sim["simulated_units"].sum() - base_units) / base_units) * 100,
                "total_revenue": sim["simulated_revenue"].sum(),
                "revenue_delta": sim["simulated_revenue"].sum() - base_rev,
                "revenue_lift_pct": ((sim["simulated_revenue"].sum() - base_rev) / base_rev) * 100,
                "total_additional_ss_units": sim["additional_ss_units"].sum()
            })

        return pd.DataFrame(sweep_records)

    def generate_visualizations(self, df_sim: pd.DataFrame, df_sweep: pd.DataFrame):
        """Generates Figure 17: Price drop scenario simulation composite."""
        print("Generating Figure 17: Price Drop Scenario Simulation...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

        # -------------------------------------------------------------
        # Ax 1: Department Demand Lift under -10% Price Drop
        # -------------------------------------------------------------
        dept_comp = df_sim.groupby(["cat_id", "dept_id"]).agg({
            "total_units": "sum",
            "simulated_units": "sum"
        }).reset_index().sort_values("total_units", ascending=True)

        y = np.arange(len(dept_comp))
        h = 0.38
        ax1.barh(y - h/2, dept_comp["total_units"] / 1e6, h, label="Baseline Demand (Units)", color="#64748b")
        ax1.barh(y + h/2, dept_comp["simulated_units"] / 1e6, h, label="Simulated Demand (-10% Price)", color="#10b981")
        ax1.set_yticks(y)
        ax1.set_yticklabels(dept_comp["dept_id"], fontweight="bold")
        ax1.set_title("Department Unit Volume Lift Under 10% Price Reduction", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel("Millions of Units Sold", fontsize=10, fontweight="bold")
        ax1.legend(loc="lower right")

        # -------------------------------------------------------------
        # Ax 2: Revenue vs Price Elasticity Sensitivity Curve
        # -------------------------------------------------------------
        ax2.plot(df_sweep["price_change_pct"], df_sweep["total_revenue"] / 1e6, color="#2563eb", linewidth=2.5, label="Simulated Gross Revenue ($M)")
        ax2.axvline(0, color="#64748b", linestyle=":", label="Status Quo (0% Change)")
        ax2.axvline(-10, color="#10b981", linestyle="--", linewidth=1.8, label="-10% Scenario")

        # Optimal price point
        opt_idx = df_sweep["total_revenue"].idxmax()
        opt_p = df_sweep.loc[opt_idx, "price_change_pct"]
        opt_rev = df_sweep.loc[opt_idx, "total_revenue"] / 1e6
        ax2.scatter([opt_p], [opt_rev], color="#f59e0b", s=90, zorder=5, label=f"Max Rev Point ({opt_p:+.1f}%)")

        ax2.set_title("Gross Revenue Sensitivity Curve Across Price Adjustments (-30% to +10%)", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Price Adjustment (%)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Total Gross Revenue ($ Millions)", fontsize=10, fontweight="bold")
        ax2.legend(loc="lower left")

        # -------------------------------------------------------------
        # Ax 3: Category Demand Lift vs Revenue Impact Breakdown
        # -------------------------------------------------------------
        cat_comp = df_sim.groupby("cat_id").agg({
            "demand_lift_pct": "mean",
            "revenue_pct_change": "mean"
        }).reset_index()

        x = np.arange(len(cat_comp))
        w = 0.35
        ax3.bar(x - w/2, cat_comp["demand_lift_pct"], w, label="Unit Demand Lift (%)", color="#10b981")
        ax3.bar(x + w/2, cat_comp["revenue_pct_change"], w, label="Revenue Change (%)", color="#3b82f6")
        ax3.axhline(0, color="#1e293b", linewidth=1)
        ax3.set_xticks(x)
        ax3.set_xticklabels(cat_comp["cat_id"], fontweight="bold")
        ax3.set_title("Category Elasticity Divergence: Unit Lift vs. Net Revenue Delta", fontsize=12, fontweight="bold", pad=10)
        ax3.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold")
        ax3.legend(loc="upper right")

        for i, row in cat_comp.iterrows():
            ax3.text(i - w/2, row["demand_lift_pct"] + 0.3, f"+{row['demand_lift_pct']:.1f}%", ha="center", fontweight="bold", fontsize=9)
            ax3.text(i + w/2, row["revenue_pct_change"] - 0.7 if row["revenue_pct_change"] < 0 else row["revenue_pct_change"] + 0.3,
                     f"{row['revenue_pct_change']:+.1f}%", ha="center", fontweight="bold", fontsize=9)

        # -------------------------------------------------------------
        # Ax 4: Supply Chain Strain: Additional Safety Stock Buffer Required
        # -------------------------------------------------------------
        ax4.plot(df_sweep["price_change_pct"], df_sweep["total_additional_ss_units"] / 1e3, color="#ef4444", linewidth=2.5)
        ax4.axvline(-10, color="#10b981", linestyle="--", label="-10% Scenario")
        ax4.set_title("Supply Chain Strain: Additional Safety Stock Required to Prevent Stockout", fontsize=12, fontweight="bold", pad=10)
        ax4.set_xlabel("Price Adjustment (%)", fontsize=10, fontweight="bold")
        ax4.set_ylabel("Additional Safety Stock (Thousands of Units)", fontsize=10, fontweight="bold")
        ax4.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(self.fig17_path, dpi=300)
        plt.close()
        print(f"[OK] Figure 17 saved to {self.fig17_path.name}")

    def export_summary(self, df_sim: pd.DataFrame, df_sweep: pd.DataFrame):
        """Exports JSON summary of the 10% price reduction scenario."""
        base_units = int(df_sim["total_units"].sum())
        sim_units = int(df_sim["simulated_units"].sum())
        unit_lift = ((sim_units - base_units) / base_units) * 100

        base_rev = float(df_sim["baseline_revenue"].sum())
        sim_rev = float(df_sim["simulated_revenue"].sum())
        rev_delta = sim_rev - base_rev
        rev_pct_change = (rev_delta / base_rev) * 100

        additional_ss = int(df_sim["additional_ss_units"].sum())

        summary = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "scenario": {
                "name": "10% Price Reduction Across Catalog",
                "price_drop_percentage": self.price_drop_pct,
                "category_elasticities": self.elasticities
            },
            "network_impact": {
                "baseline_units": base_units,
                "simulated_units": sim_units,
                "demand_units_lift": sim_units - base_units,
                "demand_lift_pct": round(unit_lift, 2),
                "baseline_gross_revenue_usd": round(base_rev, 2),
                "simulated_gross_revenue_usd": round(sim_rev, 2),
                "revenue_delta_usd": round(rev_delta, 2),
                "revenue_percentage_change": round(rev_pct_change, 2),
                "additional_safety_stock_required_units": additional_ss,
                "stockout_risk_mitigation": "Increase safety stock by additional buffer to maintain 95% CSL"
            },
            "category_breakdown": df_sim.groupby("cat_id").agg({
                "total_units": "sum",
                "simulated_units": "sum",
                "baseline_revenue": "sum",
                "simulated_revenue": "sum"
            }).round(2).to_dict(orient="index")
        }

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Saved price scenario summary to {self.output_json.relative_to(PROJECT_ROOT)}")

    def run(self):
        """Execute price drop simulation."""
        print("=" * 60)
        print("STARTING WHAT-IF PRICE DROP SCENARIO SIMULATOR")
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

        df_sim = self.simulate_price_change(df, self.price_drop_pct)
        df_sweep = self.run_sensitivity_sweep(df)

        self.generate_visualizations(df_sim, df_sweep)
        self.export_summary(df_sim, df_sweep)

        print("=" * 60)
        print("[SUCCESS] Price Drop Scenario Simulation Completed Successfully!")
        print("=" * 60)


if __name__ == "__main__":
    simulator = PriceScenarioSimulator(baseline_price_drop_pct=-10.0)
    simulator.run()
