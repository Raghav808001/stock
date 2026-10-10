"""Matplotlib dashboard: history, backtest, forecast, RSI and strategy equity."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

from .evaluation import BASELINE_NAME

THEME = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "grid": "#30363d",
    "text": "#c9d1d9",
    "muted": "#8b949e",
    "actual": "#e6edf3",
    "sma50": "#f2cc60",
    "sma200": "#ff7b72",
    "forecast": "#d2a8ff",
    "test_zone": "#1f6feb",
}
MODEL_COLORS = {
    "Linear Regression": "#58a6ff",
    "Random Forest": "#3fb950",
    "LSTM (Neural Net)": "#ff7b72",
    BASELINE_NAME: "#8b949e",
    "Buy & Hold": "#e6edf3",
}


def _style_axis(ax, title: str) -> None:
    ax.set_facecolor(THEME["panel"])
    ax.set_title(title, color=THEME["text"], fontsize=11, fontweight="bold", loc="left", pad=10)
    ax.tick_params(colors=THEME["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(THEME["grid"])
    ax.grid(color=THEME["grid"], linewidth=0.6, alpha=0.7)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _legend(ax, **kwargs) -> None:
    leg = ax.legend(facecolor=THEME["bg"], edgecolor=THEME["grid"], fontsize=8, **kwargs)
    for text in leg.get_texts():
        text.set_color(THEME["text"])


def plot_dashboard(
    ticker: str,
    features: pd.DataFrame,
    backtest: pd.DataFrame,
    metrics: pd.DataFrame,
    equity: pd.DataFrame,
    forecast: pd.DataFrame,
    forecast_model: str,
    test_start: pd.Timestamp,
    output_path: Path,
    show: bool = True,
) -> Path:
    model_names = [m for m in metrics.index if m != BASELINE_NAME]

    fig = plt.figure(figsize=(16, 12), facecolor=THEME["bg"])
    grid = GridSpec(3, 2, figure=fig, height_ratios=[1.25, 1, 0.8], hspace=0.42, wspace=0.14)
    fig.suptitle(
        f"{ticker.upper()}  ·  Stock Price Predictor",
        color=THEME["actual"], fontsize=18, fontweight="bold", x=0.06, ha="left", y=0.975,
    )
    best = metrics["RMSE"].idxmin()
    fig.text(
        0.06, 0.943,
        f"Train < {test_start:%Y-%m-%d} ≤ Test   |   lowest test RMSE: {best} "
        f"({metrics.loc[best, 'RMSE']:.2f})   |   forecast model: {forecast_model}",
        color=THEME["muted"], fontsize=10,
    )

    # 1. Full history with moving averages and forecast ------------------------
    ax = fig.add_subplot(grid[0, :])
    _style_axis(ax, "Price history, moving averages & forecast")
    ax.plot(features.index, features["Close"], color=THEME["actual"], lw=1.1, label="Close")
    ax.plot(features.index, features["SMA_50"], color=THEME["sma50"], lw=1, alpha=0.9, label="SMA 50")
    ax.plot(features.index, features["SMA_200"], color=THEME["sma200"], lw=1, alpha=0.9, label="SMA 200")
    ax.axvspan(test_start, backtest.index[-1], color=THEME["test_zone"], alpha=0.12, label="Test period")
    ax.plot(forecast.index, forecast["Predicted_Close"], color=THEME["forecast"], lw=1.8,
            label=f"{len(forecast)}-day forecast")
    ax.fill_between(forecast.index, forecast["Lower_95"], forecast["Upper_95"],
                    color=THEME["forecast"], alpha=0.2)
    ax.set_ylabel("Price", color=THEME["muted"])
    _legend(ax, loc="upper left", ncol=5)

    # 2. Backtest: predicted vs actual on unseen data ---------------------------
    ax = fig.add_subplot(grid[1, 0])
    _style_axis(ax, "Backtest: next-day predictions vs actual (test set)")
    ax.plot(backtest.index, backtest["Actual_Close"], color=THEME["actual"], lw=1.6, label="Actual")
    for name in model_names:
        ax.plot(backtest.index, backtest[f"Pred_{name}"], color=MODEL_COLORS.get(name), lw=1,
                alpha=0.9, label=f"{name}  (RMSE {metrics.loc[name, 'RMSE']:.2f})")
    ax.plot([], [], " ", label=f"{BASELINE_NAME} RMSE {metrics.loc[BASELINE_NAME, 'RMSE']:.2f}")
    _legend(ax, loc="upper left")

    # 3. Forecast zoom ----------------------------------------------------------
    ax = fig.add_subplot(grid[1, 1])
    _style_axis(ax, f"Forecast: next {len(forecast)} business days ({forecast_model})")
    recent = features["Close"].tail(120)
    ax.plot(recent.index, recent, color=THEME["actual"], lw=1.4, label="Recent close")
    bridge = pd.concat([recent.tail(1), forecast["Predicted_Close"]])
    ax.plot(bridge.index, bridge, color=THEME["forecast"], lw=2, marker="o", markersize=2.5, label="Forecast")
    ax.fill_between(forecast.index, forecast["Lower_95"], forecast["Upper_95"],
                    color=THEME["forecast"], alpha=0.2, label="~95% band")
    _legend(ax, loc="upper left")

    # 4. RSI --------------------------------------------------------------------
    ax = fig.add_subplot(grid[2, 0])
    _style_axis(ax, "RSI (14) - momentum")
    rsi = features["RSI_14"].loc[features.index >= test_start - pd.Timedelta(days=180)]
    ax.plot(rsi.index, rsi, color=THEME["sma50"], lw=1)
    ax.axhline(70, color=THEME["sma200"], ls="--", lw=0.8)
    ax.axhline(30, color=MODEL_COLORS["Random Forest"], ls="--", lw=0.8)
    ax.fill_between(rsi.index, 70, rsi, where=rsi >= 70, color=THEME["sma200"], alpha=0.3, label="Overbought")
    ax.fill_between(rsi.index, 30, rsi, where=rsi <= 30, color=MODEL_COLORS["Random Forest"], alpha=0.3,
                    label="Oversold")
    ax.set_ylim(0, 100)
    _legend(ax, loc="lower left", ncol=2)

    # 5. Strategy equity curve --------------------------------------------------
    ax = fig.add_subplot(grid[2, 1])
    _style_axis(ax, "Strategy backtest: $1 invested (long when model predicts up)")
    for name in equity.columns:
        ax.plot(equity.index, equity[name], color=MODEL_COLORS.get(name), lw=1.3,
                label=f"{name}  ({(equity[name].iloc[-1] - 1) * 100:+.1f}%)")
    ax.axhline(1, color=THEME["grid"], lw=0.8)
    _legend(ax, loc="upper left")

    fig.text(0.06, 0.012, "Educational project - not financial advice. Strategy ignores costs and slippage.",
             color=THEME["muted"], fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path
