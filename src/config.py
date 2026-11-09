"""
Centralized Project Configuration
Defines paths, forecasting parameters, hierarchy taxonomies, and environment defaults.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SQL_DIR = PROJECT_ROOT / "sql"
QUERIES_DIR = SQL_DIR / "queries"
SCHEMA_DIR = SQL_DIR / "schema"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

# Raw Data Files
CALENDAR_CSV = DATA_DIR / "calendar.csv"
SALES_TRAIN_VAL_CSV = DATA_DIR / "sales_train_validation.csv"
SALES_TRAIN_EVAL_CSV = DATA_DIR / "sales_train_evaluation.csv"
SELL_PRICES_CSV = DATA_DIR / "sell_prices.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

# Forecasting Configuration
FORECAST_HORIZON_DAYS = 28
VAL_START_DATE = "2016-04-25"
VAL_END_DATE = "2016-05-22"
VAL_START_DAY = "d_1914"
VAL_END_DAY = "d_1941"

# Hierarchy Definitions
STATES = ["CA", "TX", "WI"]
STORES = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3"
]
CATEGORIES = ["FOODS", "HOUSEHOLD", "HOBBIES"]
DEPARTMENTS = [
    "FOODS_1", "FOODS_2", "FOODS_3",
    "HOUSEHOLD_1", "HOUSEHOLD_2",
    "HOBBIES_1", "HOBBIES_2"
]

TOTAL_TIME_SERIES = 30490
TOTAL_ITEMS = 3049
TOTAL_DAYS = 1969
HISTORICAL_VAL_DAYS = 1913
HISTORICAL_EVAL_DAYS = 1941
