"""
Script to download and extract the M5 Walmart Forecasting dataset from Zenodo mirror.
"""
import os
import urllib.request
import zipfile

DATA_DIR = "data"
ZENODO_ZIP_URL = "https://zenodo.org/records/12636070/files/m5-forecasting-accuracy.zip"
ZIP_PATH = os.path.join(DATA_DIR, "m5-forecasting-accuracy.zip")

def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    csv_files = ["calendar.csv", "sales_train_validation.csv", "sell_prices.csv", "sample_submission.csv"]
    all_present = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in csv_files)
    
    if all_present:
        print("[INFO] All M5 dataset files are already present in data/ directory.")
        return

    if not os.path.exists(ZIP_PATH):
        print(f"[INFO] Downloading M5 dataset archive from {ZENODO_ZIP_URL}...")
        urllib.request.urlretrieve(ZENODO_ZIP_URL, ZIP_PATH)
        print("[INFO] Download complete.")
    else:
        print(f"[INFO] Found existing zip at {ZIP_PATH}.")

    print("[INFO] Extracting archive...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print(f"[INFO] Successfully extracted M5 dataset to {DATA_DIR}/")

if __name__ == "__main__":
    download_and_extract()
