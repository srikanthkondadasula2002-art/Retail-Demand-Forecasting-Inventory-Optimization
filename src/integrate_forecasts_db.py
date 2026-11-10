"""
Forecast Database Integration Pipeline
Loads Prophet and LightGBM 28-day out-of-sample demand forecasts,
normalizes schemas, computes absolute and percentage errors against ground truth actuals,
and integrates the telemetry into the relational database layer (fct_forecasts).
"""

import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PROJECT_ROOT, DATA_DIR, PREDICTIONS_DIR, PROCESSED_DIR, REPORTS_DIR


class ForecastDatabaseIntegrator:
    """Consolidates forecasting models outputs and ingests them into the DWH fact layer."""

    def __init__(self):
        self.lgb_path = PREDICTIONS_DIR / "lightgbm_forecast_28d.csv"
        self.prp_path = PREDICTIONS_DIR / "prophet_forecast_28d.csv"
        self.output_parquet = PROCESSED_DIR / "forecasts_integrated.parquet"
        self.output_csv = REPORTS_DIR / "sql_analytics" / "08_forecasts_integrated.csv"
        self.output_json = REPORTS_DIR / "forecast_database_integration.json"
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

    def load_and_standardize(self) -> pd.DataFrame:
        """Loads LightGBM and Prophet predictions and unifies them into a standard schema."""
        print("Loading Prophet and LightGBM forecast predictions...")
        df_lgb = pd.read_csv(self.lgb_path)
        df_prp = pd.read_csv(self.prp_path)

        # Standardize LightGBM
        df_lgb["date"] = pd.to_datetime(df_lgb["date"]).dt.strftime("%Y-%m-%d")
        df_lgb["model_name"] = "LightGBM_Tweedie"
        df_lgb["state_id"] = df_lgb["store_id"].str[:2]
        df_lgb["error_units"] = df_lgb["forecast_units"] - df_lgb["actual_units"]
        df_lgb["absolute_error"] = np.abs(df_lgb["error_units"])
        df_lgb["percentage_error"] = (df_lgb["error_units"] / df_lgb["actual_units"]) * 100
        df_lgb["lower_bound"] = np.nan
        df_lgb["upper_bound"] = np.nan

        # Standardize Prophet
        df_prp["date"] = pd.to_datetime(df_prp["ds"]).dt.strftime("%Y-%m-%d")
        df_prp["store_id"] = df_prp["series_name"]
        df_prp["model_name"] = "Prophet"
        df_prp["state_id"] = df_prp["store_id"].str[:2]
        df_prp["forecast_units"] = df_prp["yhat"]
        df_prp["lower_bound"] = df_prp["yhat_lower"]
        df_prp["upper_bound"] = df_prp["yhat_upper"]

        # Map ground-truth actuals from LightGBM table
        actual_lookup = df_lgb.set_index(["store_id", "date"])["actual_units"].to_dict()
        df_prp["actual_units"] = df_prp.apply(lambda r: actual_lookup.get((r["store_id"], r["date"]), np.nan), axis=1)
        df_prp["error_units"] = df_prp["forecast_units"] - df_prp["actual_units"]
        df_prp["absolute_error"] = np.abs(df_prp["error_units"])
        df_prp["percentage_error"] = (df_prp["error_units"] / df_prp["actual_units"]) * 100

        cols = [
            "store_id", "state_id", "date", "model_name",
            "forecast_units", "actual_units", "error_units",
            "absolute_error", "percentage_error", "lower_bound", "upper_bound"
        ]

        unified_df = pd.concat([df_lgb[cols], df_prp[cols]], ignore_index=True)
        unified_df = unified_df.sort_values(["store_id", "date", "model_name"]).reset_index(drop=True)
        return unified_df

    def evaluate_benchmarks(self, df: pd.DataFrame) -> dict:
        """Computes WAPE, MAE, RMSE, and bias across models and individual stores."""
        benchmarks = {}
        for model in ["LightGBM_Tweedie", "Prophet"]:
            sub = df[df["model_name"] == model]
            total_actual = sub["actual_units"].sum()
            total_pred = sub["forecast_units"].sum()
            total_abs_err = sub["absolute_error"].sum()
            wape = (total_abs_err / total_actual) * 100
            mae = sub["absolute_error"].mean()
            rmse = np.sqrt(np.mean(sub["error_units"] ** 2))

            store_metrics = {}
            for store in sorted(sub["store_id"].unique()):
                st_sub = sub[sub["store_id"] == store]
                st_act = st_sub["actual_units"].sum()
                st_pred = st_sub["forecast_units"].sum()
                st_wape = (st_sub["absolute_error"].sum() / st_act) * 100
                st_mae = st_sub["absolute_error"].mean()
                store_metrics[store] = {
                    "actual_units": int(st_act),
                    "forecast_units": round(float(st_pred), 1),
                    "wape_pct": round(float(st_wape), 2),
                    "mae_units": round(float(st_mae), 1)
                }

            benchmarks[model] = {
                "total_actual_units": int(total_actual),
                "total_forecast_units": round(float(total_pred), 1),
                "overall_wape_pct": round(float(wape), 2),
                "overall_mae_units": round(float(mae), 1),
                "overall_rmse_units": round(float(rmse), 1),
                "store_breakdown": store_metrics
            }
        return benchmarks

    def ingest_to_sqlite(self, df: pd.DataFrame):
        """Loads the integrated dataset into an SQLite table representing fct_forecasts."""
        print("Ingesting records into SQLite table 'fct_forecasts'...")
        conn = sqlite3.connect(DATA_DIR / "retail_demand.db")
        df.to_sql("fct_forecasts", conn, if_exists="replace", index=False)
        conn.close()
        print(f"[OK] Ingested {len(df):,} records into data/retail_demand.db -> fct_forecasts")

    def run(self):
        """Execute integration pipeline."""
        print("=" * 60)
        print("INTEGRATING FORECAST OUTPUTS INTO DATABASE LAYER")
        print("=" * 60)

        df = self.load_and_standardize()
        benchmarks = self.evaluate_benchmarks(df)

        # Export Parquet & CSV
        df.to_parquet(self.output_parquet, index=False)
        df.to_csv(self.output_csv, index=False)
        print(f"[OK] Exported {len(df):,} rows to {self.output_parquet.name}")
        print(f"[OK] Exported reporting CSV to {self.output_csv.name}")

        self.ingest_to_sqlite(df)

        # Save metadata JSON
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "forecast_horizon_days": 28,
            "models": ["LightGBM_Tweedie", "Prophet"],
            "benchmarks": benchmarks,
            "integration_status": "SUCCESS"
        }
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"[OK] Integration audit report saved to {self.output_json.relative_to(PROJECT_ROOT)}")
        print("\n--- MODEL BENCHMARK OVERVIEW ---")
        print(f"LightGBM Tweedie Overall WAPE : {benchmarks['LightGBM_Tweedie']['overall_wape_pct']}% (MAE: {benchmarks['LightGBM_Tweedie']['overall_mae_units']:.1f})")
        print(f"Facebook Prophet Overall WAPE : {benchmarks['Prophet']['overall_wape_pct']}% (MAE: {benchmarks['Prophet']['overall_mae_units']:.1f})")
        print("=" * 60)


if __name__ == "__main__":
    integrator = ForecastDatabaseIntegrator()
    integrator.run()
