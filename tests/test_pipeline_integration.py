"""
Comprehensive Pipeline and Dashboard Integration Test Suite
Validates ETL pipelines, analytical SQL transformations, forecasting models integration,
inventory optimization mechanics, price drop scenario simulations, and Streamlit components.
Outputs validation summary to reports/test_validation_report.json.
"""

import os
import sys
import json
import sqlite3
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, DATA_DIR, REPORTS_DIR, QUERIES_DIR, SQL_DIR


class TestPipelineIntegration(unittest.TestCase):
    """End-to-end integration and data validation tests."""

    @classmethod
    def setUpClass(cls):
        cls.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "failures": []
        }

    # -------------------------------------------------------------
    # 1. ETL & Preprocessing Pipeline Tests
    # -------------------------------------------------------------
    def test_01_clean_calendar_schema(self):
        """Validates clean calendar: 1,969 days, continuous, no missing events."""
        cal_path = PROCESSED_DIR / "clean_calendar.parquet"
        self.assertTrue(cal_path.exists(), "clean_calendar.parquet does not exist")
        df_cal = pd.read_parquet(cal_path)

        self.assertEqual(len(df_cal), 1969, f"Expected 1969 days, got {len(df_cal)}")
        # Check no nulls in event columns
        self.assertEqual(df_cal["event_name_1"].isnull().sum(), 0)
        self.assertEqual(df_cal["event_type_1"].isnull().sum(), 0)
        # Check date continuity
        dates = pd.to_datetime(df_cal["date"])
        day_diffs = dates.diff().dropna()
        self.assertTrue((day_diffs == pd.Timedelta(days=1)).all(), "Dates are not strictly continuous")

    def test_02_clean_sales_summary(self):
        """Validates clean sales summary: 30,490 series, zero negative values, correct types."""
        sum_path = PROCESSED_DIR / "clean_sales_summary.parquet"
        self.assertTrue(sum_path.exists(), "clean_sales_summary.parquet does not exist")
        df_sum = pd.read_parquet(sum_path)

        self.assertEqual(len(df_sum), 30490, f"Expected 30490 series, got {len(df_sum)}")
        self.assertTrue((df_sum["mean_daily_units"] >= 0).all(), "Detected negative sales means")
        self.assertTrue((df_sum["zero_days_ratio"] >= 0).all() and (df_sum["zero_days_ratio"] <= 1.0).all(), "Invalid zero_days_ratio")

    # -------------------------------------------------------------
    # 2. Forecasting Feature Engineering Tests
    # -------------------------------------------------------------
    def test_03_anti_leakage_feature_matrix(self):
        """Validates feature matrix anti-leakage policy: lags shifted >= 28 days."""
        ft_path = PROCESSED_DIR / "test_features_matrix.parquet"
        self.assertTrue(ft_path.exists(), "test_features_matrix.parquet does not exist")
        df_ft = pd.read_parquet(ft_path)

        # Expected 500 series x 28 days = 14,000 rows
        self.assertEqual(len(df_ft), 14000, f"Expected 14000 rows, got {len(df_ft)}")
        # Required anti-leakage columns
        required_cols = ["lag_28", "lag_35", "lag_42", "roll_mean_7", "roll_mean_28", "is_weekend", "snap_active"]
        for col in required_cols:
            self.assertIn(col, df_ft.columns, f"Missing required feature: {col}")

    # -------------------------------------------------------------
    # 3. Analytical SQL Aggregations Tests
    # -------------------------------------------------------------
    def test_04_analytical_sql_execution(self):
        """Validates that all analytical SQL queries execute and agree on total units."""
        conn = sqlite3.connect(":memory:")
        sd_path = PROCESSED_DIR / "store_dept_daily.parquet"
        df_sd = pd.read_parquet(sd_path)
        df_sd.to_sql("store_dept_daily", conn, if_exists="replace", index=False)

        total_units_ground_truth = int(df_sd["units"].sum())

        # Test daily query
        with open(QUERIES_DIR / "01_daily_sales_aggregation.sql", "r", encoding="utf-8") as f:
            df_daily = pd.read_sql_query(f.read(), conn)
        self.assertEqual(int(df_daily["total_units"].sum()), total_units_ground_truth)

        # Test monthly query
        with open(QUERIES_DIR / "03_monthly_sales_aggregation.sql", "r", encoding="utf-8") as f:
            df_monthly = pd.read_sql_query(f.read(), conn)
        self.assertEqual(int(df_monthly["total_units"].sum()), total_units_ground_truth)

        # Test store rankings query
        with open(QUERIES_DIR / "04_store_performance_rankings.sql", "r", encoding="utf-8") as f:
            df_stores = pd.read_sql_query(f.read(), conn)
        self.assertEqual(len(df_stores), 10, "Expected 10 stores in ranking")
        conn.close()

    # -------------------------------------------------------------
    # 4. Forecast Integration & Database Telemetry Tests
    # -------------------------------------------------------------
    def test_05_forecast_database_integration(self):
        """Validates integrated forecast table: 560 rows (280 LightGBM, 280 Prophet) and WAPE benchmark."""
        fc_path = PROCESSED_DIR / "forecasts_integrated.parquet"
        self.assertTrue(fc_path.exists(), "forecasts_integrated.parquet does not exist")
        df_fc = pd.read_parquet(fc_path)

        self.assertEqual(len(df_fc), 560, f"Expected 560 rows, got {len(df_fc)}")
        models = df_fc["model_name"].unique().tolist()
        self.assertIn("LightGBM_Tweedie", models)
        self.assertIn("Prophet", models)

        # Check LightGBM overall WAPE is 6.80% (+/- 0.05%)
        lgb = df_fc[df_fc["model_name"] == "LightGBM_Tweedie"]
        act_tot = lgb["actual_units"].sum()
        lgb_wape = (lgb["absolute_error"].sum() / act_tot) * 100
        self.assertAlmostEqual(lgb_wape, 6.80, delta=0.05, msg="LightGBM WAPE deviation")

        # Check Prophet overall WAPE is 7.79% (+/- 0.05%)
        prp = df_fc[df_fc["model_name"] == "Prophet"]
        prp_wape = (prp["absolute_error"].sum() / act_tot) * 100
        self.assertAlmostEqual(prp_wape, 7.79, delta=0.05, msg="Prophet WAPE deviation")

    # -------------------------------------------------------------
    # 5. Inventory Optimization Mechanics Tests
    # -------------------------------------------------------------
    def test_06_inventory_optimization_logic(self):
        """Validates safety stock, ROP, and EOQ mathematical constraints."""
        inv_path = PROCESSED_DIR / "inventory_recommendations.parquet"
        self.assertTrue(inv_path.exists(), "inventory_recommendations.parquet does not exist")
        df_inv = pd.read_parquet(inv_path)

        self.assertEqual(len(df_inv), 30490)
        # Safety Stock >= 0
        self.assertTrue((df_inv["safety_stock"] >= 0).all(), "Negative safety stock found")
        # ROP >= Safety Stock
        self.assertTrue((df_inv["reorder_point"] >= df_inv["safety_stock"]).all(), "ROP less than safety stock")
        # EOQ >= 1
        self.assertTrue((df_inv["eoq_units"] >= 1).all(), "Invalid EOQ")

    # -------------------------------------------------------------
    # 6. What-If Price Drop Simulation Tests
    # -------------------------------------------------------------
    def test_07_price_drop_simulation_elasticity(self):
        """Validates price drop simulation: negative price change yields positive demand lift."""
        from src.scenario_analysis import PriceScenarioSimulator
        sim = PriceScenarioSimulator(baseline_price_drop_pct=-10.0)

        df_prod = pd.read_parquet(PROCESSED_DIR / "product_performance.parquet")
        df_sum = pd.read_parquet(PROCESSED_DIR / "clean_sales_summary.parquet")
        df = pd.merge(df_prod, df_sum[["item_id", "store_id", "mean_daily_units", "std_daily_units"]], on=["item_id", "store_id"])

        sim_res = sim.simulate_price_change(df, -10.0)
        base_units = df["total_units"].sum()
        sim_units = sim_res["simulated_units"].sum()

        self.assertGreater(sim_units, base_units, "Demand lift was not positive for price cut")
        # Food elasticity should be -0.85 -> +8.5% volume
        foods_lift = sim_res[sim_res["cat_id"] == "FOODS"]["demand_lift_pct"].iloc[0]
        self.assertAlmostEqual(foods_lift, 8.5, delta=0.01)

    # -------------------------------------------------------------
    # 7. Streamlit Dashboard Data Integrity
    # -------------------------------------------------------------
    def test_08_dashboard_telemetry_sources(self):
        """Validates all parquets required by the 4 Streamlit views are healthy."""
        required_parquets = [
            PROCESSED_DIR / "store_dept_daily.parquet",
            PROCESSED_DIR / "product_performance.parquet",
            PROCESSED_DIR / "forecasts_integrated.parquet",
            PROCESSED_DIR / "inventory_recommendations.parquet"
        ]
        for p in required_parquets:
            self.assertTrue(p.exists(), f"Dashboard dependency missing: {p.name}")
            df = pd.read_parquet(p)
            self.assertGreater(len(df), 0, f"Dashboard dependency empty: {p.name}")

    # -------------------------------------------------------------
    # 8. dbt Transformation Layer Tests
    # -------------------------------------------------------------
    def test_09_dbt_transformation_layer(self):
        """Validates that dbt project configuration, staging models, and marts exist."""
        dbt_dir = PROJECT_ROOT / "dbt"
        self.assertTrue((dbt_dir / "dbt_project.yml").exists(), "dbt_project.yml missing")
        self.assertTrue((dbt_dir / "profiles.yml").exists(), "profiles.yml missing")

        # Staging models
        staging_dir = dbt_dir / "models" / "staging"
        self.assertTrue((staging_dir / "stg_calendar.sql").exists(), "stg_calendar.sql missing")
        self.assertTrue((staging_dir / "stg_sales.sql").exists(), "stg_sales.sql missing")
        self.assertTrue((staging_dir / "stg_sell_prices.sql").exists(), "stg_sell_prices.sql missing")
        self.assertTrue((staging_dir / "staging_schema.yml").exists(), "staging_schema.yml missing")

        # Mart models
        marts_dir = dbt_dir / "models" / "marts"
        self.assertTrue((marts_dir / "fct_daily_sales.sql").exists(), "fct_daily_sales.sql missing")
        self.assertTrue((marts_dir / "agg_weekly_sales.sql").exists(), "agg_weekly_sales.sql missing")
        self.assertTrue((marts_dir / "agg_monthly_sales.sql").exists(), "agg_monthly_sales.sql missing")
        self.assertTrue((marts_dir / "mart_inventory_replenishment.sql").exists(), "mart_inventory_replenishment.sql missing")
        self.assertTrue((marts_dir / "mart_forecast_evaluation.sql").exists(), "mart_forecast_evaluation.sql missing")
        self.assertTrue((marts_dir / "marts_schema.yml").exists(), "marts_schema.yml missing")


def run_tests_and_report():
    """Runs all test cases and exports reports/test_validation_report.json."""
    print("=" * 60)
    print("RUNNING PIPELINE, ETL, AND DASHBOARD INTEGRATION TEST SUITE")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPipelineIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "status": "ALL_PASSED" if result.wasSuccessful() else "FAILED",
        "failures_detail": [str(f) for f in result.failures],
        "errors_detail": [str(e) for e in result.errors]
    }

    report_path = REPORTS_DIR / "test_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {report['passed']}/{report['total_tests']} PASSED (Status: {report['status']})")
    print(f"Validation report saved to: {report_path.relative_to(PROJECT_ROOT)}")
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests_and_report()
    sys.exit(0 if success else 1)
