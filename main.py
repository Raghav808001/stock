"""Stock Price Predictor - command-line entry point.

Examples:
    python main.py                                 # AAPL, last 365 days used as the test set
    python main.py --ticker TSLA --forecast-days 20
    python main.py --ticker MSFT --start 2015-01-01 --test-start 2024-01-01 --end 2025-01-01
    python main.py --no-show                       # save the chart without opening a window
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from stock_predictor.config import OUTPUT_DIR, Config

LINE = "=" * 72


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="Train ML models to predict stock price trends.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL, TSLA, RELIANCE.NS")
    parser.add_argument("--start", default="2015-01-01", help="History start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="History end date (YYYY-MM-DD), default today")
    parser.add_argument("--test-start", default=None,
                        help="First date of the test set (default: last 365 days)")
    parser.add_argument("--forecast-days", type=int, default=30, help="Business days to forecast")
    parser.add_argument("--refresh", action="store_true", help="Re-download instead of using the cache")
    parser.add_argument("--no-show", action="store_true", help="Don't open the chart window")
    args = parser.parse_args(argv)

    if args.forecast_days < 1:
        parser.error("--forecast-days must be at least 1")
    return Config(
        ticker=args.ticker.upper(),
        start=args.start,
        end=args.end,
        test_start=args.test_start,
        forecast_days=args.forecast_days,
        refresh=args.refresh,
        show_plot=not args.no_show,
    )


def section(title: str) -> None:
    print(f"\n{LINE}\n {title}\n{LINE}")


def run(cfg: Config) -> None:
    if not cfg.show_plot:
        import matplotlib
        matplotlib.use("Agg")  # headless backend; must be set before pyplot is imported

    from stock_predictor import data, evaluation, features, forecast, models, visualize, nlp

    # ---- Phase 1: acquisition & cleaning ----------------------------------- #
    section(f"PHASE 1 - Data acquisition & cleaning  [{cfg.ticker}]")
    raw = data.fetch_data(cfg.ticker, cfg.start, cfg.end, cfg.refresh)
    prices, report = data.clean_data(raw)
    print(report)
    print(f"    date range             : {prices.index[0]:%Y-%m-%d} -> {prices.index[-1]:%Y-%m-%d}")

    # ---- Phase 2: feature engineering -------------------------------------- #
    section("PHASE 2 - Feature engineering")
    feats = features.add_features(prices)
    dataset = features.make_supervised(feats)
    if len(dataset) < 300:
        raise ValueError(
            f"Only {len(dataset)} usable rows after the 200-day warm-up. Use an earlier --start date."
        )

    test_start = (
        pd.Timestamp(cfg.test_start) if cfg.test_start else prices.index[-1] - pd.Timedelta(days=365)
    )
    train, test = features.split_by_date(dataset, test_start)
    if len(train) < 250 or len(test) < 20:
        raise ValueError(
            f"Split leaves {len(train)} train / {len(test)} test rows. Adjust --start / --test-start."
        )
    print(f"    features ({len(features.FEATURE_COLUMNS)}): {', '.join(features.FEATURE_COLUMNS)}")
    print(f"    target                 : next-day return -> converted back to price")
    print(f"    train                  : {len(train):,} rows  "
          f"({train['Target_Date'].iloc[0]:%Y-%m-%d} -> {train['Target_Date'].iloc[-1]:%Y-%m-%d})")
    print(f"    test                   : {len(test):,} rows  "
          f"({test['Target_Date'].iloc[0]:%Y-%m-%d} -> {test['Target_Date'].iloc[-1]:%Y-%m-%d})")
    print("    scaling                : MinMaxScaler(0, 1), fit on the training set only")

    # ---- Phase 3: model training ------------------------------------------- #
    section("PHASE 3 - Model training")
    trained, predicted_returns = {}, {}
    for name, model in models.build_models(cfg.random_state).items():
        cv_rmse = models.walk_forward_cv(model, train)
        trained[name] = models.fit(model, train)
        predicted_returns[name] = models.predict_returns(model, test)
        top = models.feature_importance(model).head(3)
        print(f"    {name:<18} walk-forward CV RMSE: {cv_rmse:8.3f} | top features: "
              + ", ".join(top.index))

    # ---- Phase 4: evaluation & backtest ------------------------------------ #
    section("PHASE 4 - Evaluation on unseen test data")
    names = list(trained)
    backtest = evaluation.build_backtest_frame(test, predicted_returns)
    metrics = evaluation.metrics_table(backtest, names)
    equity = evaluation.strategy_equity(backtest, names)
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 120):
        print(metrics.to_string())
    print("\n    Strategy backtest ($1 -> final value, long only when model predicts up):")
    for col in equity.columns:
        print(f"      {col:<18} {equity[col].iloc[-1]:.3f}")

    best_ml = metrics.loc[names, "RMSE"].idxmin()
    baseline_rmse = metrics.loc[evaluation.BASELINE_NAME, "RMSE"]
    edge = (baseline_rmse - metrics.loc[best_ml, "RMSE"]) / baseline_rmse * 100
    if abs(edge) < 1:
        verdict = f"is effectively TIED with the naive baseline ({edge:+.2f}% RMSE)"
    elif edge > 0:
        verdict = f"beats the naive baseline by {edge:.2f}% RMSE"
    else:
        verdict = f"is WORSE than the naive baseline by {-edge:.2f}% RMSE"
    print(f"\n    Reality check: best model ({best_ml}) {verdict}.")
    print("    (Efficient Market Hypothesis: past prices alone carry little predictive signal.)")

    # ---- Phase 5: NLP Sentiment Analysis ----------------------------------- #
    section(f"PHASE 5 - Real-time NLP Sentiment Analysis [{cfg.ticker}]")
    news_items = nlp.get_recent_news_sentiment(cfg.ticker)
    if news_items:
        avg_score, sentiment = nlp.aggregate_sentiment(news_items)
        print(f"    analyzed {len(news_items)} recent news headlines")
        print(f"    overall market mood    : {sentiment} ({avg_score:+.2f})")
        print("\n    Latest headlines:")
        for i, item in enumerate(news_items[:5]):
            print(f"      [{item.timestamp.strftime('%Y-%m-%d %H:%M')}] {item.headline[:80]}... "
                  f"({item.sentiment_label}: {item.sentiment_score:+.2f})")
    else:
        print("    No recent news found for sentiment analysis.")

    # ---- Forecast ---------------------------------------------------------- #
    section(f"FORECAST - next {cfg.forecast_days} business days ({best_ml}, retrained on all data)")
    final_model = models.fit(models.build_models(cfg.random_state)[best_ml], dataset)
    error_std = float((backtest["Actual_Return"] - backtest[f"PredReturn_{best_ml}"]).std())
    future = forecast.forecast_future(final_model, prices, cfg.forecast_days, error_std)
    last_close = prices["Close"].iloc[-1]
    end_close = future["Predicted_Close"].iloc[-1]
    print(f"    last close ({prices.index[-1]:%Y-%m-%d}) : {last_close:,.2f}")
    print(f"    forecast   ({future.index[-1]:%Y-%m-%d}) : {end_close:,.2f}  "
          f"({(end_close / last_close - 1) * 100:+.2f}%)  "
          f"95% band {future['Lower_95'].iloc[-1]:,.2f} - {future['Upper_95'].iloc[-1]:,.2f}")
    trend = "UPWARD" if end_close > last_close else "DOWNWARD"
    print(f"    predicted trend        : {trend}")

    # ---- Save outputs & visualise ------------------------------------------ #
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(OUTPUT_DIR / f"{cfg.ticker}_backtest.csv")
    future.to_csv(OUTPUT_DIR / f"{cfg.ticker}_forecast.csv")
    metrics.to_csv(OUTPUT_DIR / f"{cfg.ticker}_metrics.csv")
    chart = visualize.plot_dashboard(
        cfg.ticker, feats, backtest, metrics, equity, future, best_ml, test_start,
        OUTPUT_DIR / f"{cfg.ticker}_dashboard.png", show=cfg.show_plot,
    )
    section("Outputs")
    print(f"    chart   : {chart}")
    print(f"    tables  : {OUTPUT_DIR / (cfg.ticker + '_[backtest|forecast|metrics].csv')}")
    print("\n    Educational project - not financial advice.\n")


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    try:
        run(cfg)
    except (RuntimeError, ValueError) as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
