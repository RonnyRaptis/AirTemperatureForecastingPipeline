# evaluate.py
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from data_loader import load_processed_data
from model import ARIMAModel
import config

def run_evaluation() -> None:
    """
    Evaluate the trained SARIMA model by forecasting on a test set
    and calculating RMSE. A plot comparing train, test, and forecast is saved.
    """
    try:
        df = load_processed_data()
    except Exception as e:
        logging.error("Failed to load processed data: %s", e)
        return

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Split data: train (up to 2020) and test (after 2020)
    train_series = df[df.index.year <= 2020]['t2m']
    test_series = df[df.index.year > 2020]['t2m']
    
    try:
        model = ARIMAModel.load(config.MODEL_PATH)
    except Exception as e:
        logging.error("Failed to load the model: %s", e)
        return
    
    steps = len(test_series)
    forecast = model.predict(steps)
    
    rmse = np.sqrt(mean_squared_error(test_series, forecast))
    logging.info("Test RMSE: %.2f", rmse)
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_series.index, train_series, label="Train")
    plt.plot(test_series.index, test_series, label="Test", color="blue")
    plt.plot(test_series.index, forecast, label="Forecast", color="red")
    plt.title("ARIMA Forecast Evaluation")
    plt.xlabel("Time")
    plt.ylabel("2m Air Temperature (deg C)")
    plt.legend()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "forecast_evaluation.png"))
    plt.show()
    logging.info("Evaluation complete. Plot saved to %s", os.path.join(config.OUTPUT_DIR, "forecast_evaluation.png"))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_evaluation()
