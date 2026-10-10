"""Phase 4 - Evaluation and backtesting.

Every model is compared against a naive baseline ("tomorrow's close = today's
close"). Because of the Efficient Market Hypothesis, beating that baseline on
price RMSE is hard, so the baseline keeps the results honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASELINE_NAME = "Naive (no change)"


def price_metrics(actual: np.ndarray, predicted: np.ndarray, prev_close: np.ndarray) -> dict:
    actual, predicted, prev_close = map(np.asarray, (actual, predicted, prev_close))
    actual_move = np.sign(actual - prev_close)
    predicted_move = np.sign(predicted - prev_close)
    has_view = predicted_move != 0  # the naive model never predicts a direction
    direction = float(np.mean(actual_move[has_view] == predicted_move[has_view])) if has_view.any() else np.nan

    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MAPE %": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
        "R2": float(r2_score(actual, predicted)),
        "Direction Acc %": direction * 100 if not np.isnan(direction) else np.nan,
    }


def build_backtest_frame(test: pd.DataFrame, predicted_returns: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per test day, indexed by the date being predicted."""
    frame = pd.DataFrame(
        {
            "Prev_Close": test["Close"].to_numpy(),
            "Actual_Close": test["Next_Close"].to_numpy(),
            "Actual_Return": test["Target_Return"].to_numpy(),
        },
        index=pd.DatetimeIndex(test["Target_Date"], name="Date"),
    )
    frame[f"Pred_{BASELINE_NAME}"] = frame["Prev_Close"]
    for name, returns in predicted_returns.items():
        frame[f"PredReturn_{name}"] = returns
        frame[f"Pred_{name}"] = frame["Prev_Close"] * (1 + returns)
    return frame


def metrics_table(backtest: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows = {}
    for name in [BASELINE_NAME, *model_names]:
        rows[name] = price_metrics(backtest["Actual_Close"], backtest[f"Pred_{name}"], backtest["Prev_Close"])
    return pd.DataFrame(rows).T


def strategy_equity(backtest: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """Simple long/flat strategy: hold the stock on days the model predicts a rise.

    Ignores transaction costs, slippage and taxes, so real results would be worse.
    """
    equity = pd.DataFrame(index=backtest.index)
    equity["Buy & Hold"] = (1 + backtest["Actual_Return"]).cumprod()
    for name in model_names:
        in_market = backtest[f"PredReturn_{name}"] > 0
        equity[name] = (1 + backtest["Actual_Return"].where(in_market, 0.0)).cumprod()
    return equity
