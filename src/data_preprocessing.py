"""
M5 Walmart Dataset Preprocessing and Data Cleaning Pipeline
Handles extraction, schema downcasting, missing value imputation,
date formatting/continuity validation, and generates a data cleaning audit report.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    CALENDAR_CSV, SALES_TRAIN_VAL_CSV, SELL_PRICES_CSV,
    PROJECT_ROOT, PROCESSED_DIR, REPORTS_DIR, TOTAL_TIME_SERIES, TOTAL_DAYS
)


class M5DataPreprocessor:
    """End-to-end data cleaning, typing optimization, and extraction pipeline."""

    def __init__(self):
        self.audit_log = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_status": "PENDING",
            "calendar_cleaning": {},
            "sales_cleaning": {},
            "prices_cleaning": {},
            "memory_optimizations": {}
        }

    def clean_calendar(self) -> pd.DataFrame:
        """
        Extracts and cleans calendar.csv:
        - Parses date column into datetime64[ns]
        - Verifies monotonic date continuity (1,969 days, no gaps)
        - Imputes missing event values with 'None'
        - Optimizes integer types to int8/int16
        """
        print("\n--- Cleaning Calendar Data ---")
        df_cal = pd.read_csv(CALENDAR_CSV)
        raw_mem_mb = df_cal.memory_usage(deep=True).sum() / (1024 * 1024)

        # Check missing values before imputation
        null_counts_before = df_cal.isnull().sum().to_dict()

        # Handle missing event names and types
        event_cols = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
        for col in event_cols:
            df_cal[col] = df_cal[col].fillna("None").astype("category")

        # Parse and validate dates
        df_cal["date"] = pd.to_datetime(df_cal["date"])
        min_date = df_cal["date"].min()
        max_date = df_cal["date"].max()
        expected_days = (max_date - min_date).days + 1
        actual_days = len(df_cal["date"].unique())
        is_continuous = (expected_days == actual_days) and (actual_days == TOTAL_DAYS)

        # Downcast integer columns
        int8_cols = ["wday", "month", "snap_CA", "snap_TX", "snap_WI"]
        for col in int8_cols:
            df_cal[col] = df_cal[col].astype(np.int8)

        df_cal["year"] = df_cal["year"].astype(np.int16)
        df_cal["wm_yr_wk"] = df_cal["wm_yr_wk"].astype(np.int32)
        df_cal["weekday"] = df_cal["weekday"].astype("category")

        clean_mem_mb = df_cal.memory_usage(deep=True).sum() / (1024 * 1024)
        mem_savings_pct = (raw_mem_mb - clean_mem_mb) / raw_mem_mb * 100

        # Save to processed directory
        out_path = PROCESSED_DIR / "clean_calendar.parquet"
        df_cal.to_parquet(out_path, index=False)
        print(f"[OK] Cleaned calendar saved to {out_path.name} ({clean_mem_mb:.2f} MB, -{mem_savings_pct:.1f}% RAM)")
        print(f"     Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({actual_days} days, Continuous: {is_continuous})")

        self.audit_log["calendar_cleaning"] = {
            "total_days": int(actual_days),
            "min_date": min_date.strftime("%Y-%m-%d"),
            "max_date": max_date.strftime("%Y-%m-%d"),
            "is_continuous": bool(is_continuous),
            "missing_values_handled": {k: int(v) for k, v in null_counts_before.items() if v > 0},
            "raw_memory_mb": round(raw_mem_mb, 2),
            "clean_memory_mb": round(clean_mem_mb, 2),
            "memory_savings_pct": round(mem_savings_pct, 2)
        }
        return df_cal

    def clean_sales(self) -> pd.DataFrame:
        """
        Extracts and cleans sales_train_validation.csv:
        - Downcasts IDs to categoricals
        - Downcasts sales counts (d_1 to d_1913) from int64 to int16
        - Verifies 0 nulls, 0 negative values
        - Computes series-level summary stats (total units, zero ratio, active days)
        """
        print("\n--- Cleaning Sales Data & Extracting Summary ---")
        # Read header and dtypes
        df_sales = pd.read_csv(SALES_TRAIN_VAL_CSV)
        raw_mem_mb = df_sales.memory_usage(deep=True).sum() / (1024 * 1024)

        id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        d_cols = [c for c in df_sales.columns if c.startswith("d_")]

        # Validate no negative or null values
        sales_matrix = df_sales[d_cols].values
        has_nulls = bool(np.isnan(sales_matrix).any())
        has_negatives = bool((sales_matrix < 0).any())

        # Downcast day columns to int16 in-place
        for col in d_cols:
            df_sales[col] = df_sales[col].astype(np.int16)

        # Categorize ID columns
        for col in id_cols:
            df_sales[col] = df_sales[col].astype("category")

        clean_mem_mb = df_sales.memory_usage(deep=True).sum() / (1024 * 1024)
        mem_savings_pct = (raw_mem_mb - clean_mem_mb) / raw_mem_mb * 100

        print(f"[OK] Sales downcast: {raw_mem_mb:.1f} MB -> {clean_mem_mb:.1f} MB (-{mem_savings_pct:.1f}% RAM)")
        print(f"     Null values detected: {has_nulls}, Negative values: {has_negatives}")

        # Compute summary metrics per time series
        print("     Computing series performance summaries...")
        total_units = sales_matrix.sum(axis=1)
        zero_counts = (sales_matrix == 0).sum(axis=1)
        zero_ratio = zero_counts / len(d_cols)
        mean_units = sales_matrix.mean(axis=1)
        std_units = sales_matrix.std(axis=1)

        summary_df = pd.DataFrame({
            "id": df_sales["id"].astype(str),
            "item_id": df_sales["item_id"].astype(str),
            "dept_id": df_sales["dept_id"].astype(str),
            "cat_id": df_sales["cat_id"].astype(str),
            "store_id": df_sales["store_id"].astype(str),
            "state_id": df_sales["state_id"].astype(str),
            "total_units": total_units.astype(np.int32),
            "mean_daily_units": np.round(mean_units, 3).astype(np.float32),
            "std_daily_units": np.round(std_units, 3).astype(np.float32),
            "zero_days_count": zero_counts.astype(np.int16),
            "zero_days_ratio": np.round(zero_ratio, 4).astype(np.float32)
        })

        out_path = PROCESSED_DIR / "clean_sales_summary.parquet"
        summary_df.to_parquet(out_path, index=False)
        print(f"[OK] Clean sales summary saved to {out_path.name} ({len(summary_df):,} series)")

        self.audit_log["sales_cleaning"] = {
            "total_series": len(df_sales),
            "historical_days": len(d_cols),
            "has_nulls": has_nulls,
            "has_negatives": has_negatives,
            "raw_memory_mb": round(raw_mem_mb, 2),
            "clean_memory_mb": round(clean_mem_mb, 2),
            "memory_savings_pct": round(mem_savings_pct, 2),
            "overall_zero_demand_ratio": round(float(zero_counts.sum() / (len(df_sales) * len(d_cols))), 4),
            "total_cumulative_units": int(total_units.sum())
        }
        return df_sales

    def clean_prices(self) -> pd.DataFrame:
        """
        Extracts and cleans sell_prices.csv:
        - Downcasts sell_price to float32
        - Categorizes store_id and item_id
        - Analyzes price ranges and missing values
        """
        print("\n--- Cleaning Sell Prices Data ---")
        df_prices = pd.read_csv(SELL_PRICES_CSV)
        raw_mem_mb = df_prices.memory_usage(deep=True).sum() / (1024 * 1024)

        null_prices = int(df_prices["sell_price"].isnull().sum())
        min_price = float(df_prices["sell_price"].min())
        max_price = float(df_prices["sell_price"].max())
        median_price = float(df_prices["sell_price"].median())

        # Optimize types
        df_prices["store_id"] = df_prices["store_id"].astype("category")
        df_prices["item_id"] = df_prices["item_id"].astype("category")
        df_prices["wm_yr_wk"] = df_prices["wm_yr_wk"].astype(np.int32)
        df_prices["sell_price"] = df_prices["sell_price"].astype(np.float32)

        clean_mem_mb = df_prices.memory_usage(deep=True).sum() / (1024 * 1024)
        mem_savings_pct = (raw_mem_mb - clean_mem_mb) / raw_mem_mb * 100

        print(f"[OK] Sell prices downcast: {raw_mem_mb:.1f} MB -> {clean_mem_mb:.1f} MB (-{mem_savings_pct:.1f}% RAM)")
        print(f"     Price Range: ${min_price:.2f} to ${max_price:.2f} (Median: ${median_price:.2f})")

        self.audit_log["prices_cleaning"] = {
            "total_price_records": len(df_prices),
            "null_prices_count": null_prices,
            "min_price": round(min_price, 2),
            "max_price": round(max_price, 2),
            "median_price": round(median_price, 2),
            "raw_memory_mb": round(raw_mem_mb, 2),
            "clean_memory_mb": round(clean_mem_mb, 2),
            "memory_savings_pct": round(mem_savings_pct, 2)
        }
        return df_prices

    def run(self):
        """Execute full preprocessing pipeline and save audit report."""
        print("=" * 60)
        print("STARTING M5 DATA PREPROCESSING & CLEANING PIPELINE")
        print("=" * 60)

        self.clean_calendar()
        self.clean_sales()
        self.clean_prices()

        self.audit_log["pipeline_status"] = "SUCCESS"
        total_raw_mem = (
            self.audit_log["calendar_cleaning"]["raw_memory_mb"] +
            self.audit_log["sales_cleaning"]["raw_memory_mb"] +
            self.audit_log["prices_cleaning"]["raw_memory_mb"]
        )
        total_clean_mem = (
            self.audit_log["calendar_cleaning"]["clean_memory_mb"] +
            self.audit_log["sales_cleaning"]["clean_memory_mb"] +
            self.audit_log["prices_cleaning"]["clean_memory_mb"]
        )
        self.audit_log["memory_optimizations"] = {
            "total_raw_memory_mb": round(total_raw_mem, 2),
            "total_clean_memory_mb": round(total_clean_mem, 2),
            "total_memory_saved_mb": round(total_raw_mem - total_clean_mem, 2),
            "total_memory_reduction_pct": round((total_raw_mem - total_clean_mem) / total_raw_mem * 100, 2)
        }

        audit_path = REPORTS_DIR / "data_cleaning_audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(self.audit_log, f, indent=2)

        print("\n" + "=" * 60)
        print(f"[SUCCESS] Preprocessing completed successfully!")
        print(f"Total RAM footprint: {total_raw_mem:.1f} MB -> {total_clean_mem:.1f} MB ({self.audit_log['memory_optimizations']['total_memory_reduction_pct']}% reduction)")
        print(f"Audit report saved to: {audit_path.relative_to(PROJECT_ROOT)}")
        print("=" * 60)


if __name__ == "__main__":
    preprocessor = M5DataPreprocessor()
    preprocessor.run()
