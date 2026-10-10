import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for streamlit

from stock_predictor import data, features, models, evaluation, forecast, visualize, nlp
from stock_predictor.config import Config, OUTPUT_DIR

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

st.title("📈 Advanced Stock Price Predictor")
st.markdown("Machine Learning Pipeline (Linear Regression, Random Forest, LSTM) & NLP Sentiment Analysis")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    TICKERS = {
        "AAPL": "AAPL (Apple)",
        "TSLA": "TSLA (Tesla)",
        "MSFT": "MSFT (Microsoft)",
        "GOOG": "GOOG (Google)",
        "AMZN": "AMZN (Amazon)",
        "NVDA": "NVDA (Nvidia)",
        "RELIANCE.NS": "RELIANCE.NS (Reliance Industries)",
        "TCS.NS": "TCS.NS (Tata Consultancy Services)",
        "TATAMOTORS.NS": "TATAMOTORS.NS (Tata Motors)",
        "HDFCBANK.NS": "HDFCBANK.NS (HDFC Bank)",
        "ZOMATO.NS": "ZOMATO.NS (Zomato)",
        "INFY.NS": "INFY.NS (Infosys)"
    }
    
    ticker = st.selectbox("Ticker Symbol", options=list(TICKERS.keys()), format_func=lambda x: TICKERS[x])
    forecast_days = st.slider("Forecast Days", min_value=5, max_value=60, value=30)
    st.markdown("---")
    run_btn = st.button("Run ML Pipeline", type="primary")

if run_btn:
    cfg = Config(ticker=ticker, forecast_days=forecast_days, show_plot=False)
    
    with st.spinner(f"Fetching Data for {ticker}..."):
        raw = data.fetch_data(ticker, cfg.start, cfg.end, cfg.refresh)
        prices, report = data.clean_data(raw)
        
    with st.spinner("Engineering Technical Features..."):
        feats = features.add_features(prices)
        dataset = features.make_supervised(feats)
        test_start = prices.index[-1] - pd.Timedelta(days=365)
        train, test = features.split_by_date(dataset, test_start)
        
    with st.spinner("Training Machine Learning Models (LR, RF, LSTM)..."):
        trained, predicted_returns = {}, {}
        for name, model in models.build_models().items():
            trained[name] = models.fit(model, train)
            predicted_returns[name] = models.predict_returns(model, test)
            
    with st.spinner("Evaluating & Forecasting..."):
        names = list(trained)
        backtest = evaluation.build_backtest_frame(test, predicted_returns)
        metrics = evaluation.metrics_table(backtest, names)
        equity = evaluation.strategy_equity(backtest, names)
        best_ml = metrics.loc[names, "RMSE"].idxmin()
        
        final_model = models.fit(models.build_models()[best_ml], dataset)
        error_std = float((backtest["Actual_Return"] - backtest[f"PredReturn_{best_ml}"]).std())
        future = forecast.forecast_future(final_model, prices, forecast_days, error_std)
        
        # Save output so we can show the chart
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        chart_path = OUTPUT_DIR / f"{ticker}_dashboard.png"
        visualize.plot_dashboard(
            ticker, feats, backtest, metrics, equity, future, best_ml, test_start,
            chart_path, show=False
        )

    # Layout Results
    st.success(f"Pipeline complete! Best performing model on test data: **{best_ml}**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Price Forecast & Backtest Dashboard")
        st.image(str(chart_path), use_column_width=True)
        
    with col2:
        st.subheader("📰 Real-time NLP Sentiment")
        news_items = nlp.get_recent_news_sentiment(ticker)
        if news_items:
            avg_score, sentiment = nlp.aggregate_sentiment(news_items)
            st.metric("Market Mood", sentiment, f"{avg_score:+.2f} VADER Score")
            for item in news_items[:7]:
                # Streamlit markdown colors
                color = "green" if item.sentiment_label == "POSITIVE" else "red" if item.sentiment_label == "NEGATIVE" else "gray"
                st.markdown(f"- **:{color}[{item.sentiment_label} ({item.sentiment_score:+.2f})]** [{item.timestamp.strftime('%m-%d %H:%M')}] {item.headline}")
        else:
            st.info("No recent news found.")
            
        st.subheader("📉 Test Year Performance")
        st.dataframe(metrics.style.format("{:.3f}"), use_container_width=True)
