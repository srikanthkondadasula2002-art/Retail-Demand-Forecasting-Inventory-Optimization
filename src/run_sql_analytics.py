"""
SQL Analytics Runner
Executes analytical SQL queries against in-memory retail sales dataset.
Outputs tabular CSV results and a comprehensive executive markdown report.
"""

import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PROJECT_ROOT, PROCESSED_DIR, REPORTS_DIR, QUERIES_DIR


def df_to_markdown_simple(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame into a clean Markdown table string without tabulate dependency."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join([str(val) if pd.notnull(val) else "" for val in row]) + " |")
    return "\n".join([header, separator] + rows)


class SQLAnalyticsRunner:
    """Executes the suite of analytical SQL queries and generates reporting tables."""

    def __init__(self):
        self.db_conn = sqlite3.connect(":memory:")
        self.output_dir = REPORTS_DIR / "sql_analytics"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data_to_sqlite(self):
        """Loads parquet datasets into in-memory SQLite tables."""
        print("Loading processed sales aggregations into in-memory SQLite database...")
        sd_path = PROCESSED_DIR / "store_dept_daily.parquet"
        df_sd = pd.read_parquet(sd_path)

        # Write to SQLite
        df_sd.to_sql("store_dept_daily", self.db_conn, if_exists="replace", index=False)
        print(f"[OK] In-memory table 'store_dept_daily' populated ({len(df_sd):,} rows).")

    def run_queries(self):
        """Discovers and runs all .sql files in sql/queries/."""
        query_files = sorted(list(QUERIES_DIR.glob("*.sql")))
        print(f"Found {len(query_files)} analytical SQL queries in {QUERIES_DIR.name}/")

        results = {}
        for qf in query_files:
            query_name = qf.stem
            print(f"\n--- Running [{query_name}] ---")
            with open(qf, "r", encoding="utf-8") as f:
                sql_script = f.read()

            try:
                df_res = pd.read_sql_query(sql_script, self.db_conn)
                print(f"[OK] Query returned {len(df_res):,} rows.")
                # Save CSV
                csv_path = self.output_dir / f"{query_name}.csv"
                df_res.to_csv(csv_path, index=False)
                results[query_name] = df_res
            except Exception as e:
                print(f"[ERROR] Failed to run {query_name}: {e}")

        self.generate_markdown_report(results)

    def generate_markdown_report(self, results: dict):
        """Generates an executive markdown report consolidating all SQL query results."""
        report_path = self.output_dir / "sql_analytics_summary.md"
        print(f"\nGenerating consolidated report at {report_path.relative_to(PROJECT_ROOT)}...")

        md_lines = [
            "# M5 Retail Sales Analytical SQL Reporting Summary",
            "",
            "> **Dataset**: Walmart M5 Forecasting Dataset (1,913 Historical Days)",
            "> **Database**: In-Memory Relational Analytical Engine (Standard ANSI SQL)",
            "",
            "---",
            "",
            "## 1. Store Performance Rankings",
            ""
        ]

        if "04_store_performance_rankings" in results:
            df = results["04_store_performance_rankings"]
            md_lines.append(df_to_markdown_simple(df))
            md_lines.append("")

        md_lines.extend([
            "## 2. Category & Department Contribution Analysis",
            ""
        ])
        if "05_category_velocity_analysis" in results:
            df = results["05_category_velocity_analysis"]
            md_lines.append(df_to_markdown_simple(df))
            md_lines.append("")

        md_lines.extend([
            "## 3. SNAP Policy Benefit Demand Lift",
            ""
        ])
        if "06_snap_impact_analysis" in results:
            df = results["06_snap_impact_analysis"]
            md_lines.append(df_to_markdown_simple(df))
            md_lines.append("")

        md_lines.extend([
            "## 4. Weekend Demand Surge Analysis",
            ""
        ])
        if "07_weekend_surge_metrics" in results:
            df = results["07_weekend_surge_metrics"]
            md_lines.append(df_to_markdown_simple(df))
            md_lines.append("")

        md_lines.extend([
            "## 5. Recent Monthly Rollups & MoM Growth",
            ""
        ])
        if "03_monthly_sales_aggregation" in results:
            df = results["03_monthly_sales_aggregation"].tail(12)
            md_lines.append(df_to_markdown_simple(df))
            md_lines.append("")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"[OK] Consolidated report saved to {report_path.relative_to(PROJECT_ROOT)}")

    def run(self):
        """Execute full analytics pipeline."""
        print("=" * 60)
        print("STARTING ANALYTICAL SQL PIPELINE")
        print("=" * 60)
        self.load_data_to_sqlite()
        self.run_queries()
        print("\n" + "=" * 60)
        print("[SUCCESS] All Analytical SQL Queries Executed Successfully!")
        print("=" * 60)


if __name__ == "__main__":
    runner = SQLAnalyticsRunner()
    runner.run()
