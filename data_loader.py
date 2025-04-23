### data_loader.py

import os
import logging
import xarray as xr
import pandas as pd
import numpy as np
import config

def download_data() -> None:
    """
    Download ERA5 NetCDF data if it doesn't exist.
    This is just a placeholder; if you already have era5_t2m.nc in DATA_DIR,
    it will do nothing.
    """
    if not os.path.exists(config.DATASET_PATH):
        logging.info("Data file not found. Downloading data... (placeholder)")
        # TODO: insert real download code here using e.g. cdsapi
        logging.info("Data downloaded and saved to %s", config.DATASET_PATH)
    else:
        logging.info("Data file already exists: %s", config.DATASET_PATH)


def process_data() -> pd.DataFrame:
    # 1) Open and convert
    ds = xr.open_dataset(config.DATASET_PATH)
    t2m = ds['t2m'] - 273.15

    # 2) Figure out what the time coordinate is called
    #    commonly 'time' or 'valid_time'
    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    
    # 3) Build cosine(lat) weights
    weights = np.cos(np.deg2rad(ds.latitude))
    weights.name = "weights"

    # 4) Resample on the correct dim, then area‑weight
    t2m_monthly = (
        t2m
        .resample({time_coord: 'MS'})   # use the detected name here
        .mean()
        .weighted(weights)
        .mean(dim=('latitude','longitude'))
    )

    # 5) Rename the index to 'time' so downstream code can always use df['time']
    t2m_monthly = t2m_monthly.rename({time_coord: 'time'})

    # 6) Save and return
    os.makedirs(os.path.dirname(config.PROCESSED_PATH), exist_ok=True)
    t2m_monthly.to_netcdf(config.PROCESSED_PATH)
    logging.info("Processed area-weighted monthly data saved to %s", config.PROCESSED_PATH)

    df = t2m_monthly.to_series().reset_index()
    df.columns = ['time', 't2m']
    return df


def load_processed_data() -> pd.DataFrame:
    """
    Load the processed area‐weighted monthly series from disk.
    If it doesn’t exist, call process_data() first.
    """
    if not os.path.exists(config.PROCESSED_PATH):
        logging.info("Processed file not found. Running process_data()...")
        process_data()

    ds = xr.open_dataset(config.PROCESSED_PATH)
    df = ds.to_dataframe().reset_index()
    return df


if __name__ == '__main__':
    download_data()
    process_data()
