# predict.py
import os
import logging
import matplotlib.pyplot as plt
from model import ARIMAModel
import config

def run_prediction() -> None:
    """
    Load the trained model, forecast the next 12 months, and plot/save the predictions.
    """
    try:
        model = ARIMAModel.load(config.MODEL_PATH)
    except Exception as e:
        logging.error("Failed to load the model: %s", e)
        return

    steps = 12  # Predict 12 future months
    forecast = model.predict(steps)
    
    logging.info("Forecast for the next %d months:", steps)
    logging.info("\n%s", forecast)
    
    plt.figure(figsize=(10, 5))
    forecast.plot(title="Next 12 Months Air Temp Forecast")
    plt.xlabel("Month")
    plt.ylabel("2m Air Temperature (deg C)")
    plt.savefig(os.path.join(config.OUTPUT_DIR, "forecast_prediction.png"))
    plt.show()
    logging.info("Prediction complete. Plot saved to %s", os.path.join(config.OUTPUT_DIR, "forecast_prediction.png"))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_prediction()
