"""Recursive multi-day forecasting beyond the last known close.

The model only predicts one day ahead, so to forecast N days we:
    1. predict tomorrow's return from today's features,
    2. append the predicted close to the price history,
    3. recompute every indicator and repeat.

Errors compound with each step, so the uncertainty band widens with the horizon
(roughly sigma * sqrt(h), where sigma is the model's one-day error on the test set).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, add_features


def forecast_future(model, history: pd.DataFrame, days: int, daily_error_std: float) -> pd.DataFrame:
    hist = history[["Close", "Volume"]].copy()
    assumed_volume = float(hist["Volume"].tail(20).mean())  # future volume is unknown
    future_dates = pd.bdate_range(hist.index[-1] + pd.Timedelta(days=1), periods=days)

    rows = []
    for step, day in enumerate(future_dates, start=1):
        features = add_features(hist.tail(400))  # enough history for SMA-200 and the EMA to settle
        x = features[FEATURE_COLUMNS].iloc[[-1]]
        if x.isna().any(axis=None):
            raise ValueError("Not enough history to compute features for forecasting.")
        predicted_return = float(model.predict(x)[0])
        next_close = float(hist["Close"].iloc[-1]) * (1 + predicted_return)
        hist.loc[day] = [next_close, assumed_volume]

        band = 1.96 * daily_error_std * np.sqrt(step)  # ~95% interval
        rows.append(
            {
                "Date": day,
                "Predicted_Close": next_close,
                "Predicted_Return": predicted_return,
                "Lower_95": next_close * (1 - band),
                "Upper_95": next_close * (1 + band),
            }
        )
    return pd.DataFrame(rows).set_index("Date")
