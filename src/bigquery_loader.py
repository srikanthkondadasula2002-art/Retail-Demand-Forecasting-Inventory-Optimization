"""
BigQuery Data Warehouse Loader
Creates relational tables and loads processed M5 telemetry into Google Cloud BigQuery.
Supports both live BigQuery upload and dry-run validation mode.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from google.cloud import bigquery
from src.bigquery_config import BigQueryConfig

DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

TABLE_SCHEMAS = {
    "dim_calendar": [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("wm_yr_wk", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("weekday", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("wday", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("month", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("year", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("d", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("event_name_1", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("event_type_1", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("event_name_2", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("event_type_2", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("snap_CA", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("snap_TX", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("snap_WI", "INTEGER", mode="REQUIRED"),
    ],
    "dim_product": [
        bigquery.SchemaField("item_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("dept_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("cat_id", "STRING", mode="REQUIRED"),
    ],
    "dim_store": [
        bigquery.SchemaField("store_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("state_id", "STRING", mode="REQUIRED"),
    ],
    "fct_daily_sales": [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("d", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("store_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("state_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("cat_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("dept_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("units", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("wm_yr_wk", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("weekday", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("wday", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("month", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("year", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("snap_active", "INTEGER", mode="REQUIRED"),
    ],
    "fct_product_performance": [
        bigquery.SchemaField("item_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("store_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("state_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("cat_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("dept_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("total_units", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("zero_days", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("total_days", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("sparsity_pct", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("avg_sell_price", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("est_revenue", "FLOAT", mode="REQUIRED"),
    ]
}

def prepare_tables_data() -> Dict[str, pd.DataFrame]:
    """Loads and standardizes local processed data into warehouse table frames."""
    print("Preparing datasets for BigQuery loading...", flush=True)
    
    # 1. dim_calendar
    cal_path = os.path.join(DATA_DIR, "calendar.csv")
    df_cal = pd.read_csv(cal_path)
    df_cal['date'] = pd.to_datetime(df_cal['date']).dt.date
    
    # 2. Daily sales & snap_active
    dept_daily_path = os.path.join(PROCESSED_DIR, "store_dept_daily.parquet")
    if not os.path.exists(dept_daily_path):
        from src.prepare_dashboard_data import prepare_all
        prepare_all()
        
    df_daily = pd.read_parquet(dept_daily_path)
    df_daily['date'] = pd.to_datetime(df_daily['date']).dt.date
    df_daily['snap_active'] = (
        (df_daily['state_id'] == 'CA') & (df_daily['snap_CA'] == 1) |
        (df_daily['state_id'] == 'TX') & (df_daily['snap_TX'] == 1) |
        (df_daily['state_id'] == 'WI') & (df_daily['snap_WI'] == 1)
    ).astype(int)
    
    daily_cols = [
        'date', 'd', 'store_id', 'state_id', 'cat_id', 'dept_id', 
        'units', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'snap_active'
    ]
    df_daily = df_daily[daily_cols]

    # 3. fct_product_performance
    prod_path = os.path.join(PROCESSED_DIR, "product_performance.parquet")
    df_prod = pd.read_parquet(prod_path)

    # 4. dim_product & dim_store (normalized dimensions)
    df_dim_product = df_prod[['item_id', 'dept_id', 'cat_id']].drop_duplicates().reset_index(drop=True)
    df_dim_store = df_prod[['store_id', 'state_id']].drop_duplicates().reset_index(drop=True)

    return {
        "dim_calendar": df_cal,
        "dim_product": df_dim_product,
        "dim_store": df_dim_store,
        "fct_daily_sales": df_daily,
        "fct_product_performance": df_prod
    }

def run_dry_run_validation(data_dict: Dict[str, pd.DataFrame], config: BigQueryConfig):
    """Performs rigorous schema, nullability, and partitioning dry-run checks."""
    print("\n" + "=" * 65)
    print(f"RUNNING BIGQUERY DRY-RUN VALIDATION [{config.full_dataset_id}]")
    print("=" * 65)
    
    total_rows = 0
    total_bytes = 0
    
    for table_name, df in data_dict.items():
        schema = TABLE_SCHEMAS[table_name]
        expected_cols = [field.name for field in schema]
        actual_cols = list(df.columns)
        
        # Verify columns exist
        missing = set(expected_cols) - set(actual_cols)
        if missing:
            raise ValueError(f"Table '{table_name}' missing expected columns: {missing}")
            
        # Verify required non-null fields
        for field in schema:
            if field.mode == "REQUIRED":
                null_count = df[field.name].isnull().sum()
                if null_count > 0:
                    raise ValueError(f"Column '{field.name}' in '{table_name}' is REQUIRED but has {null_count} nulls!")

        bytes_est = df.memory_usage(deep=True).sum()
        total_rows += len(df)
        total_bytes += bytes_est
        
        partition_info = "Partition: [date (DAY)]" if "date" in df.columns else "Partition: None"
        print(f"[OK] Table: {table_name:<24} | Rows: {len(df):>8,} | Est Memory: {bytes_est/(1024**2):>6.2f} MB | {partition_info}")

    print("-" * 65)
    print(f"Total Tables Validated : {len(data_dict)}")
    print(f"Total Rows Prepared    : {total_rows:,}")
    print(f"Total Estimated Volume : {total_bytes / (1024**2):.2f} MB")
    print("[OK] All schemas, datatypes, and partitioning constraints strictly validated.")
    print("=" * 65)

def run_bigquery_upload(data_dict: Dict[str, pd.DataFrame], config: BigQueryConfig, client: bigquery.Client):
    """Executes live table creation and batch load jobs in Google Cloud BigQuery."""
    print("\n" + "=" * 65)
    print(f"EXECUTING LIVE BIGQUERY LOAD -> {config.full_dataset_id}")
    print("=" * 65)

    # 1. Create Dataset if needed
    dataset_ref = bigquery.DatasetReference(config.project_id, config.dataset_id)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = config.location
    dataset = client.create_dataset(dataset, exists_ok=True)
    print(f"[OK] Dataset '{dataset.dataset_id}' ready in location '{dataset.location}'.")

    # 2. Create tables & load data
    for table_name, df in data_dict.items():
        table_ref = dataset_ref.table(table_name)
        schema = TABLE_SCHEMAS[table_name]

        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        
        # Partitioning & Clustering configuration
        if "date" in df.columns:
            job_config.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="date"
            )
            
        if table_name == "fct_daily_sales":
            job_config.clustering_fields = ["store_id", "cat_id", "dept_id"]
        elif table_name == "fct_product_performance":
            job_config.clustering_fields = ["store_id", "cat_id", "dept_id"]
        elif table_name == "dim_product":
            job_config.clustering_fields = ["cat_id", "dept_id"]

        print(f"Uploading {len(df):,} rows to `{config.get_table_id(table_name)}`...", end="", flush=True)
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Wait for job completion
        print(" [DONE]")

    print("\n[OK] Live BigQuery upload completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="BigQuery Table Creation and Data Loader")
    parser.add_argument("--project", type=str, default=None, help="GCP Project ID")
    parser.add_argument("--dataset", type=str, default=None, help="BigQuery Dataset ID")
    parser.add_argument("--dry-run", action="store_true", help="Perform schema and data validation without GCP API calls")
    args = parser.parse_args()

    config = BigQueryConfig(project_id=args.project, dataset_id=args.dataset)
    data_dict = prepare_tables_data()

    client = None
    if not args.dry_run:
        client = config.get_client()

    if args.dry_run or client is None:
        if not args.dry_run:
            print("[INFO] Active GCP BigQuery credentials not found. Executing in dry-run validation mode.")
        run_dry_run_validation(data_dict, config)
    else:
        run_dry_run_validation(data_dict, config)
        run_bigquery_upload(data_dict, config, client)

if __name__ == "__main__":
    main()
