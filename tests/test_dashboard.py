"""
Day 9 — Dashboard Integration & Validation Tests
Person 3: Dashboard & Inventory Optimization
Tests all dashboard component engines for correctness and integration.
"""

import sys
import os
import math
import pytest
import pandas as pd
import numpy as np

# Allow imports from dashboard/components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))

from components.forecasting_engine import (
    load_forecast_series,
    get_store_metrics_summary,
    MODEL_EVALUATION_METRICS,
)
from components.inventory_engine import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_eoq,
    evaluate_stock_status,
    generate_inventory_recommendations,
    SERVICE_LEVEL_Z,
)
from components.scenario_engine import (
    simulate_price_scenario,
    CATEGORY_ELASTICITIES,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_products():
    return pd.DataFrame([
        {"item_id": "FOODS_3_090",  "item_name": "Fresh Bananas",       "department": "FOODS_3",     "category": "FOODS",     "base_price": 1.25, "sales_velocity": 450, "annual_units": 164250, "annual_revenue": 205312.5},
        {"item_id": "HOUSEHOLD_1_118", "item_name": "Paper Towels",     "department": "HOUSEHOLD_1", "category": "HOUSEHOLD", "base_price": 12.98, "sales_velocity": 180, "annual_units": 65700,  "annual_revenue": 852786.0},
        {"item_id": "HOBBIES_1_001",   "item_name": "Lego Creator Set", "department": "HOBBIES_1",  "category": "HOBBIES",   "base_price": 24.99, "sales_velocity": 85,  "annual_units": 31025,  "annual_revenue": 775514.75},
    ])


@pytest.fixture(scope="module")
def base_forecast_30():
    return load_forecast_series(store_id="CA_1", horizon_days=30)


# ── Forecasting Engine Tests ──────────────────────────────────────────────────

class TestForecastingEngine:

    def test_load_forecast_series_shape(self, base_forecast_30):
        assert len(base_forecast_30) == 30
        required_cols = {"date", "store", "actual_units", "lightgbm_forecast",
                         "prophet_forecast", "prophet_lower", "prophet_upper", "ensemble_forecast"}
        assert required_cols.issubset(set(base_forecast_30.columns))

    def test_forecast_values_non_negative(self, base_forecast_30):
        for col in ["actual_units", "lightgbm_forecast", "prophet_forecast",
                    "prophet_lower", "prophet_upper", "ensemble_forecast"]:
            assert (base_forecast_30[col] >= 0).all(), f"Negative values found in {col}"

    def test_prophet_confidence_band_ordering(self, base_forecast_30):
        assert (base_forecast_30["prophet_upper"] >= base_forecast_30["prophet_lower"]).all()

    def test_ensemble_is_blend_of_lgb_and_prophet(self, base_forecast_30):
        expected = ((base_forecast_30["lightgbm_forecast"] + base_forecast_30["prophet_forecast"]) / 2).round().astype(int)
        assert (base_forecast_30["ensemble_forecast"] - expected).abs().max() <= 1

    def test_all_stores_have_metrics(self):
        stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
        for s in stores:
            assert s in MODEL_EVALUATION_METRICS, f"Missing metrics for store {s}"

    def test_metrics_summary_dataframe(self):
        df = get_store_metrics_summary()
        assert len(df) == 10
        assert "Store" in df.columns
        assert "Top Model" in df.columns

    def test_forecast_store_isolation(self):
        df_ca1 = load_forecast_series("CA_1", 30)
        df_wi2 = load_forecast_series("WI_2", 30)
        # Different stores should produce different demand volumes
        assert df_ca1["actual_units"].sum() != df_wi2["actual_units"].sum()

    def test_forecast_horizon_respected(self):
        for h in [7, 14, 28, 30]:
            df = load_forecast_series("TX_1", h)
            assert len(df) == h, f"Expected {h} rows, got {len(df)}"


# ── Inventory Engine Tests ────────────────────────────────────────────────────

class TestInventoryEngine:

    def test_safety_stock_positive(self):
        ss = calculate_safety_stock(
            mean_daily_demand=15.0, std_daily_demand=5.0,
            lead_time_days=7, lead_time_std=1.5, service_level=0.95
        )
        assert ss > 0

    def test_safety_stock_increases_with_service_level(self):
        kwargs = dict(mean_daily_demand=20.0, std_daily_demand=6.0,
                      lead_time_days=5, lead_time_std=1.0)
        ss_90 = calculate_safety_stock(**kwargs, service_level=0.90)
        ss_99 = calculate_safety_stock(**kwargs, service_level=0.99)
        assert ss_99 > ss_90

    def test_safety_stock_formula(self):
        # SS = Z * sqrt(L * sigma_d^2 + d_avg^2 * sigma_L^2)
        z = SERVICE_LEVEL_Z[0.95]
        L, sigma_d, d_avg, sigma_L = 7, 5.0, 15.0, 1.5
        expected = math.ceil(z * math.sqrt(L * sigma_d**2 + d_avg**2 * sigma_L**2))
        result = calculate_safety_stock(d_avg, sigma_d, L, sigma_L, 0.95)
        assert result == expected

    def test_reorder_point_includes_safety_stock(self):
        rop = calculate_reorder_point(mean_daily_demand=10.0, lead_time_days=5, safety_stock=30)
        assert rop >= 30 + 10 * 5

    def test_eoq_positive(self):
        eoq = calculate_eoq(annual_demand=5000, order_cost=35.0, holding_cost_unit=2.5)
        assert eoq > 0

    def test_eoq_wilson_formula(self):
        D, S, H = 5000, 35.0, 2.5
        expected = max(10, int(round(math.sqrt((2 * D * S) / H))))
        assert calculate_eoq(D, S, H) == expected

    def test_eoq_zero_demand_fallback(self):
        assert calculate_eoq(0, 35.0, 2.5) == 50

    def test_stock_status_critical(self):
        assert evaluate_stock_status(5, 30, 60, 100) == "🔴 Critical Risk"

    def test_stock_status_reorder(self):
        assert evaluate_stock_status(45, 20, 60, 100) == "🟡 Reorder Needed"

    def test_stock_status_healthy(self):
        assert evaluate_stock_status(100, 20, 60, 100) == "🟢 Healthy"

    def test_stock_status_overstocked(self):
        assert evaluate_stock_status(300, 20, 60, 100) == "🔵 Overstocked"

    def test_generate_recommendations_shape(self, sample_products):
        df = generate_inventory_recommendations(sample_products, store_id="CA_1")
        assert len(df) == len(sample_products)

    def test_generate_recommendations_required_columns(self, sample_products):
        df = generate_inventory_recommendations(sample_products, store_id="TX_1")
        required = {"item_id", "safety_stock", "reorder_point", "eoq",
                    "current_stock", "stock_status", "reorder_units", "est_order_cost"}
        assert required.issubset(set(df.columns))

    def test_reorder_units_zero_when_healthy(self, sample_products):
        df = generate_inventory_recommendations(sample_products, store_id="CA_3", service_level=0.95)
        healthy = df[df["stock_status"] == "🟢 Healthy"]
        if not healthy.empty:
            assert (healthy["reorder_units"] == 0).all()

    def test_est_order_cost_matches_units_times_price(self, sample_products):
        df = generate_inventory_recommendations(sample_products, store_id="WI_1")
        for _, row in df.iterrows():
            expected_cost = round(row["reorder_units"] * row["unit_price"], 2)
            assert abs(row["est_order_cost"] - expected_cost) < 0.01

    def test_service_level_override_affects_safety_stock(self, sample_products):
        df_90 = generate_inventory_recommendations(sample_products, "CA_1", service_level=0.90)
        df_99 = generate_inventory_recommendations(sample_products, "CA_1", service_level=0.99)
        assert df_99["safety_stock"].sum() > df_90["safety_stock"].sum()


# ── Scenario Engine Tests ─────────────────────────────────────────────────────

class TestScenarioEngine:

    def test_price_drop_increases_demand(self, base_forecast_30):
        result = simulate_price_scenario(base_forecast_30, price_change_pct=-10.0)
        assert result["summary"]["unit_lift_total"] > 0

    def test_price_increase_decreases_demand(self, base_forecast_30):
        result = simulate_price_scenario(base_forecast_30, price_change_pct=10.0)
        assert result["summary"]["unit_lift_total"] < 0

    def test_zero_price_change_no_lift(self, base_forecast_30):
        result = simulate_price_scenario(base_forecast_30, price_change_pct=0.0)
        assert result["summary"]["unit_lift_total"] == 0

    def test_trajectory_df_shape(self, base_forecast_30):
        result = simulate_price_scenario(base_forecast_30, price_change_pct=-10.0)
        df = result["trajectory_df"]
        assert len(df) == 30
        assert {"date", "base_forecast", "simulated_demand", "unit_lift",
                "base_revenue", "simulated_revenue"}.issubset(set(df.columns))

    def test_simulated_demand_non_negative(self, base_forecast_30):
        result = simulate_price_scenario(base_forecast_30, price_change_pct=-30.0)
        assert (result["trajectory_df"]["simulated_demand"] >= 0).all()

    def test_category_breakdown_has_three_rows(self, base_forecast_30):
        result = simulate_price_scenario(base_forecast_30, price_change_pct=-10.0)
        assert len(result["category_breakdown"]) == 3

    def test_aggressive_regime_larger_lift_than_conservative(self, base_forecast_30):
        r_cons = simulate_price_scenario(base_forecast_30, -10.0, "Conservative")
        r_aggr = simulate_price_scenario(base_forecast_30, -10.0, "Aggressive")
        assert r_aggr["summary"]["unit_lift_total"] > r_cons["summary"]["unit_lift_total"]

    def test_summary_keys_present(self, base_forecast_30):
        result = simulate_price_scenario(base_forecast_30, price_change_pct=-10.0)
        required_keys = {"price_change_pct", "base_price", "simulated_price",
                         "unit_lift_total", "pct_unit_lift", "rev_delta",
                         "pct_rev_delta", "margin_delta", "extra_buffer_units_needed"}
        assert required_keys.issubset(set(result["summary"].keys()))

    def test_buffer_units_positive_on_price_drop(self, base_forecast_30):
        result = simulate_price_scenario(base_forecast_30, price_change_pct=-15.0)
        assert result["summary"]["extra_buffer_units_needed"] > 0

    def test_foods_category_focus(self, base_forecast_30):
        result = simulate_price_scenario(base_forecast_30, -10.0, "Moderate", "FOODS")
        assert result["summary"]["base_price"] == 3.25

    def test_revenue_delta_sign_consistency(self, base_forecast_30):
        # For a price drop, revenue delta direction depends on elasticity
        result = simulate_price_scenario(base_forecast_30, -10.0, "Moderate", "FOODS")
        s = result["summary"]
        computed_delta = s["total_sim_rev"] - s["total_base_rev"]
        assert abs(computed_delta - s["rev_delta"]) < 1.0


# ── Integration Tests ─────────────────────────────────────────────────────────

class TestEndToEndIntegration:
    """Validates the full pipeline: forecast → inventory → scenario."""

    def test_full_pipeline_ca3(self, sample_products):
        # Step 1: Load forecast
        df_fc = load_forecast_series("CA_3", 30)
        assert len(df_fc) == 30

        # Step 2: Generate inventory recommendations
        df_inv = generate_inventory_recommendations(sample_products, "CA_3", 0.95)
        assert len(df_inv) == 3

        # Step 3: Run scenario simulation
        result = simulate_price_scenario(df_fc, -10.0, "Moderate", "FOODS")
        assert result["summary"]["unit_lift_total"] > 0

    def test_all_stores_inventory_pipeline(self, sample_products):
        stores = ["CA_1", "TX_1", "WI_1"]
        for store in stores:
            df = generate_inventory_recommendations(sample_products, store)
            assert len(df) == len(sample_products), f"Failed for store {store}"
            assert df["stock_status"].notna().all()

    def test_metrics_summary_covers_all_stores(self):
        df = get_store_metrics_summary()
        expected_stores = {"CA_1", "CA_2", "CA_3", "CA_4",
                           "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"}
        assert set(df["Store"].tolist()) == expected_stores
