"""
Project Environment and Dataset Setup Verification
Validates directory structure, Python packages, and M5 Walmart datasets.
Outputs summary to reports/environment_setup.json.
"""

import os
import sys
import json
import platform
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    PROJECT_ROOT, DATA_DIR, PROCESSED_DIR, PREDICTIONS_DIR,
    REPORTS_DIR, FIGURES_DIR, SQL_DIR, QUERIES_DIR, SCHEMA_DIR,
    NOTEBOOKS_DIR, CALENDAR_CSV, SALES_TRAIN_VAL_CSV,
    SALES_TRAIN_EVAL_CSV, SELL_PRICES_CSV, SAMPLE_SUBMISSION_CSV
)


def verify_directories():
    """Verify and create all required project directories."""
    required_dirs = [
        DATA_DIR, PROCESSED_DIR, PREDICTIONS_DIR,
        REPORTS_DIR, FIGURES_DIR, SQL_DIR, QUERIES_DIR,
        SCHEMA_DIR, NOTEBOOKS_DIR
    ]
    dir_status = {}
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)
        dir_status[str(d.relative_to(PROJECT_ROOT))] = "EXISTS"
    return dir_status


def verify_packages():
    """Verify presence and versions of required core packages."""
    package_names = [
        "pandas", "numpy", "scipy", "matplotlib",
        "seaborn", "pyarrow", "streamlit", "lightgbm", "prophet"
    ]
    pkg_status = {}
    for pkg in package_names:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            pkg_status[pkg] = {"status": "INSTALLED", "version": version}
        except ImportError as e:
            pkg_status[pkg] = {"status": "MISSING", "error": str(e)}
    return pkg_status


def verify_datasets():
    """Verify presence, size, and existence of M5 Walmart raw files."""
    dataset_files = {
        "calendar": CALENDAR_CSV,
        "sales_train_validation": SALES_TRAIN_VAL_CSV,
        "sales_train_evaluation": SALES_TRAIN_EVAL_CSV,
        "sell_prices": SELL_PRICES_CSV,
        "sample_submission": SAMPLE_SUBMISSION_CSV,
    }
    data_status = {}
    for name, path in dataset_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            data_status[name] = {
                "exists": True,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_mb": round(size_mb, 2),
                "status": "VALID"
            }
        else:
            data_status[name] = {
                "exists": False,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_mb": 0.0,
                "status": "MISSING"
            }
    return data_status


def run_verification():
    """Run full environment and dataset setup verification."""
    print("=" * 60)
    print("M5 RETAIL FORECASTING - PROJECT SETUP VERIFICATION")
    print("=" * 60)

    dirs = verify_directories()
    print(f"\n[OK] Directories Verified: {len(dirs)} directories confirmed.")

    packages = verify_packages()
    all_packages_ok = all(v["status"] == "INSTALLED" for v in packages.values())
    print(f"\n[{'OK' if all_packages_ok else 'WARN'}] Packages Verified:")
    for pkg, info in packages.items():
        status = info["status"]
        ver = info.get("version", "N/A")
        print(f"  - {pkg:<15}: {status} ({ver})")

    datasets = verify_datasets()
    all_datasets_ok = all(v["exists"] for v in datasets.values())
    print(f"\n[{'OK' if all_datasets_ok else 'WARN'}] Datasets Verified:")
    total_size_mb = 0.0
    for name, info in datasets.items():
        status = info["status"]
        sz = info["size_mb"]
        total_size_mb += sz
        print(f"  - {name:<25}: {status} ({sz:.2f} MB)")
    print(f"  Total Raw Data Size: {total_size_mb:.2f} MB")

    report = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
            "machine": platform.machine()
        },
        "directories": dirs,
        "packages": packages,
        "datasets": datasets,
        "total_data_size_mb": round(total_size_mb, 2),
        "setup_complete": all_packages_ok and all_datasets_ok
    }

    report_path = REPORTS_DIR / "environment_setup.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Verification report written to: {report_path.relative_to(PROJECT_ROOT)}")
    print("=" * 60)
    return report


if __name__ == "__main__":
    run_verification()
