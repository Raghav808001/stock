"""Phase 5 - Natural Language Processing (NLP) for Market Sentiment.

Historical price data lacks context (earnings reports, geopolitics, product launches).
This module fetches real-time news headlines using yfinance and scores their
sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).

Since we can only easily fetch *recent* news (not 10 years of history), we use
this to provide a real-time sentiment gauge to accompany our price forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

@dataclass
class NewsItem:
    headline: str
    publisher: str
    timestamp: datetime
    sentiment_score: float  # -1.0 (very negative) to 1.0 (very positive)
    sentiment_label: str

def get_recent_news_sentiment(ticker: str) -> list[NewsItem]:
    """Fetch recent news for a ticker and analyze sentiment."""
    import urllib.request
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return []

    analyzer = SentimentIntensityAnalyzer()
    
    # yfinance news is currently unreliable, fallback to Google News RSS
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        xml_data = urllib.request.urlopen(req, timeout=10).read()
        root = ET.fromstring(xml_data)
        news_data = root.findall('.//item')[:15]  # Get up to 15 recent headlines
    except Exception:
        news_data = []

    results = []
    for item in news_data:
        headline_elem = item.find("title")
        if headline_elem is None or not headline_elem.text:
            continue
        headline = headline_elem.text
            
        source_elem = item.find("source")
        publisher = source_elem.text if source_elem is not None else "News"
        
        pubDate_elem = item.find("pubDate")
        if pubDate_elem is not None and pubDate_elem.text:
            try:
                dt = parsedate_to_datetime(pubDate_elem.text)
            except Exception:
                dt = datetime.now()
        else:
            dt = datetime.now()

        # VADER compound score is normalized between -1 (extreme negative) and +1 (extreme positive)
        score = analyzer.polarity_scores(headline)["compound"]
        
        if score > 0.15:
            label = "POSITIVE"
        elif score < -0.15:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        results.append(
            NewsItem(
                headline=headline,
                publisher=publisher,
                timestamp=dt,
                sentiment_score=score,
                sentiment_label=label,
            )
        )
    
    # Sort by newest first
    results.sort(key=lambda x: x.timestamp, reverse=True)
    return results

def aggregate_sentiment(news: list[NewsItem]) -> tuple[float, str]:
    """Calculate the average market sentiment from a list of news items."""
    if not news:
        return 0.0, "UNKNOWN"
    
    avg_score = sum(item.sentiment_score for item in news) / len(news)
    if avg_score > 0.15:
        return avg_score, "BULLISH"
    elif avg_score < -0.15:
        return avg_score, "BEARISH"
    return avg_score, "NEUTRAL"
