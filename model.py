### model.py (enabled trend term in SARIMAX)
import pickle
import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple

class ARIMAModel:
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def train(self, train_series: pd.Series) -> None:
        """
        Train a SARIMA model on the given Pandas Series, including a linear trend.
        """
        try:
            model = SARIMAX(
                train_series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend='ct',  # include constant and linear time trend
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_fit = model.fit(disp=False)
            logging.info(
                "Model trained with order=%s, seasonal_order=%s, trend='ct'",
                self.order, self.seasonal_order
            )
        except Exception as e:
            logging.error("Training failed: %s", e)
            raise

    def predict(self, steps: int) -> pd.Series:
        if self.model_fit is None:
            raise Exception("Model is not trained yet.")
        return self.model_fit.forecast(steps=steps)

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logging.info("Model saved to %s", filepath)

    @staticmethod
    def load(filepath: str) -> 'ARIMAModel':
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded from %s", filepath)
        return model
