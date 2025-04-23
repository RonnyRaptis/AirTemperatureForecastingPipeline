# config.py
import os

# Directory paths
DATA_DIR = "datasets"
OUTPUT_DIR = "outputs"

# File paths
DATASET_PATH = os.path.join(DATA_DIR, "era5_t2m.nc")
PROCESSED_PATH = os.path.join(DATA_DIR, "era5_t2m_monthly.nc")
MODEL_PATH = os.path.join(OUTPUT_DIR, "arima_model.pkl")

# ARIMA model parameters
ARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)
