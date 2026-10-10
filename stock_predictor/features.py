"""Phase 2 - Feature engineering.

Turns raw prices into indicators the model can use. Every feature at day ``t``
uses only data available at the close of day ``t`` (no look-ahead bias).

Design choice: the model predicts the *next-day return* rather than the raw
price. Raw prices drift outside their training range (AAPL in 2024 is far above
its 2015-2023 range), which breaks scaling and tree models. Returns and
price-to-moving-average ratios stay in a stable range, and the predicted return
is converted back to a price: ``predicted_close = close_t * (1 + predicted_return)``.

Features are built from Close and Volume only, so the same function can be
reused for recursive multi-day forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Target_Return"

FEATURE_COLUMNS = [
    # Trend: where is price relative to its moving averages?
    "Close_SMA10_Ratio",
    "Close_SMA50_Ratio",
    "Close_SMA200_Ratio",
    "SMA50_SMA200_Ratio",   # > 0 means the 50-day is above the 200-day ("golden cross" regime)
    "Close_EMA20_Ratio",
    # Momentum
    "RSI_14",
    "Return_1d",
    "Return_5d",
    "Return_20d",
    "Return_Lag_1",
    "Return_Lag_2",
    "Return_Lag_3",
    # Risk & activity
    "Volatility_20",
    "Volume_Ratio_20",
]


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0)                     # only gains -> 100
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50)  # flat -> neutral
    return rsi.where(avg_gain.notna())                        # keep warm-up period as NaN


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators. Requires 'Close' and 'Volume' columns."""
    out = df.copy()
    close, volume = out["Close"], out["Volume"]

    out["SMA_10"] = close.rolling(10).mean()
    out["SMA_50"] = close.rolling(50).mean()
    out["SMA_200"] = close.rolling(200).mean()
    out["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    out["Close_SMA10_Ratio"] = close / out["SMA_10"] - 1
    out["Close_SMA50_Ratio"] = close / out["SMA_50"] - 1
    out["Close_SMA200_Ratio"] = close / out["SMA_200"] - 1
    out["SMA50_SMA200_Ratio"] = out["SMA_50"] / out["SMA_200"] - 1
    out["Close_EMA20_Ratio"] = close / out["EMA_20"] - 1

    out["RSI_14"] = compute_rsi(close, 14)

    out["Return_1d"] = close.pct_change(1)
    out["Return_5d"] = close.pct_change(5)
    out["Return_20d"] = close.pct_change(20)
    for lag in (1, 2, 3):
        out[f"Return_Lag_{lag}"] = out["Return_1d"].shift(lag)

    out["Volatility_20"] = out["Return_1d"].rolling(20).std()
    volume_ma = volume.rolling(20).mean().replace(0, np.nan)
    out["Volume_Ratio_20"] = (volume / volume_ma - 1).fillna(0)

    return out.replace([np.inf, -np.inf], np.nan)


def make_supervised(features: pd.DataFrame) -> pd.DataFrame:
    """Attach the next-day target to each row and drop rows that can't be used.

    Row ``t`` gets:
        Next_Close    - the close on the next trading day
        Target_Date   - the date of that next trading day
        Target_Return - Next_Close / Close - 1   (what the model learns)
    """
    data = features.copy()
    data["Next_Close"] = data["Close"].shift(-1)
    data["Target_Date"] = pd.Series(data.index, index=data.index).shift(-1)
    data[TARGET] = data["Next_Close"] / data["Close"] - 1
    return data.dropna(subset=FEATURE_COLUMNS + [TARGET])


def split_by_date(data: pd.DataFrame, test_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split on the *target* date so no test-period price leaks into training."""
    train = data[data["Target_Date"] < test_start]
    test = data[data["Target_Date"] >= test_start]
    return train, test
