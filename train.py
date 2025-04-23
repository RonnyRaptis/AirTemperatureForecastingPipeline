# train.py
import os
import logging
import pandas as pd
from data_loader import load_processed_data
from model import ARIMAModel
import config

def run_training() -> None:
    """
    Load processed data, train a SARIMA model, and save it.
    """
    try:
        df = load_processed_data()
    except Exception as e:
        logging.error("Failed to load processed data: %s", e)
        return

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.index = pd.DatetimeIndex(df.index, freq='MS')

    # Use data up to 2024 for training
    train_series = df[df.index.year <= 2024]['t2m']

    model = ARIMAModel(order=config.ARIMA_ORDER, seasonal_order=config.SARIMA_SEASONAL_ORDER)
    model.train(train_series)

    try:
        model.save(config.MODEL_PATH)
    except Exception as e:
        logging.error("Failed to save the model: %s", e)
        return

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_training()
