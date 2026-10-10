"""Phase 3 - Models.

Every model is a scikit-learn Pipeline whose first step is MinMaxScaler, so the
scaler is fit on training data only (fitting it on all data would leak test
information into training).

    Level 1: Linear Regression  - the core model of this project
    Level 2: Random Forest      - captures non-linear indicator interactions
    Level 3: LSTM               - time-series neural network (TensorFlow/Keras)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .features import FEATURE_COLUMNS, TARGET
from sklearn.base import BaseEstimator, RegressorMixin

class TimeSeriesLSTM(BaseEstimator, RegressorMixin):
    """Scikit-Learn wrapper for a Keras LSTM that automatically windows the data."""
    def __init__(self, lookback: int = 30, epochs: int = 20, batch_size: int = 32):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None

    def _make_windows(self, X: np.ndarray, y: np.ndarray | None = None):
        # Pad the start so we return the same number of rows
        padded = np.vstack([np.tile(X[0], (self.lookback - 1, 1)), X])
        windows = np.array([padded[i : i + self.lookback] for i in range(len(X))])
        if y is not None:
            return windows, y
        return windows

    def fit(self, X, y):
        import tensorflow as tf
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence TF info logs
        tf.get_logger().setLevel('ERROR')

        X_arr, y_arr = np.asarray(X), np.asarray(y)
        seq_X, seq_y = self._make_windows(X_arr, y_arr)

        # Fix random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

        self.model_ = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(self.lookback, X_arr.shape[1])),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        self.model_.compile(optimizer="adam", loss="mse")
        
        # verbose=0 keeps the console clean
        self.model_.fit(seq_X, seq_y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        seq_X = self._make_windows(np.asarray(X))
        return self.model_.predict(seq_X, verbose=0).flatten()


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "Linear Regression": Pipeline(
            [("scaler", MinMaxScaler()), ("model", LinearRegression())]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=6,          # shallow trees + large leaves to limit overfitting to noise
                        min_samples_leaf=20,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "LSTM (Neural Net)": Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("model", TimeSeriesLSTM(lookback=30, epochs=15)),
            ]
        ),
    }


def fit(model: Pipeline, data: pd.DataFrame) -> Pipeline:
    model.fit(data[FEATURE_COLUMNS], data[TARGET])
    return model


def predict_returns(model: Pipeline, data: pd.DataFrame) -> np.ndarray:
    return model.predict(data[FEATURE_COLUMNS])


def walk_forward_cv(model: Pipeline, train: pd.DataFrame, n_splits: int = 5) -> float:
    """Time-series cross-validation on the training set: mean price RMSE across folds.

    Each fold trains on the past and validates on the block right after it, the
    same way the model would be used in reality.
    """
    from sklearn.base import clone

    rmses = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(train):
        fold_train, fold_val = train.iloc[train_idx], train.iloc[val_idx]
        fold_model = fit(clone(model), fold_train)
        pred_close = fold_val["Close"] * (1 + predict_returns(fold_model, fold_val))
        rmses.append(float(np.sqrt(np.mean((fold_val["Next_Close"] - pred_close) ** 2))))
    return float(np.mean(rmses))


def feature_importance(model: Pipeline) -> pd.Series:
    """Linear coefficients (on scaled features) or tree impurity importances."""
    est = model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = np.abs(est.coef_)
    else:
        # LSTM doesn't have a simple linear feature importance
        values = np.zeros(len(FEATURE_COLUMNS))
    return pd.Series(values, index=FEATURE_COLUMNS).sort_values(ascending=False)
