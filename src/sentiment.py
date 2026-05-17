"""
News sentiment module (optional).

Graceful fallback when transformers or NewsAPI are unavailable:
all public functions return a neutral-sentiment dict.
"""

import os
import numpy as np

_FINBERT_AVAILABLE = False
_NEWSAPI_AVAILABLE = False

try:
    from transformers import pipeline as _hf_pipeline
    _FINBERT_AVAILABLE = True
except ImportError:
    pass

try:
    from newsapi import NewsApiClient as _NewsApiClient
    _NEWSAPI_AVAILABLE = True
except ImportError:
    pass

_NEUTRAL = {
    "score":      0.0,
    "positive":   0.333,
    "negative":   0.333,
    "neutral":    0.334,
    "available":  False,
    "n_articles": 0,
    "headlines":  [],
}


def get_news_sentiment(
    query: str = "gold price market commodity",
    max_articles: int = 15,
) -> dict:
    """
    Fetch recent gold-related headlines and score them with FinBERT.

    Requires:
      - NEWSAPI_KEY environment variable (or set at runtime)
      - `newsapi-python` package: pip install newsapi-python
      - `transformers` package: pip install transformers torch

    Returns a dict with keys: score, positive, negative, neutral,
    available, n_articles, headlines.
    """
    api_key = os.getenv("NEWSAPI_KEY", "")

    if not _NEWSAPI_AVAILABLE or not api_key:
        return {**_NEUTRAL, "reason": "NEWSAPI_KEY not set or newsapi-python not installed"}

    headlines: list[str] = []
    try:
        client = _NewsApiClient(api_key=api_key)
        resp = client.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=max_articles,
        )
        headlines = [
            a["title"] for a in resp.get("articles", [])
            if a.get("title") and "[Removed]" not in a["title"]
        ]
    except Exception as exc:
        return {**_NEUTRAL, "reason": f"NewsAPI error: {exc}"}

    if not headlines:
        return {**_NEUTRAL, "reason": "No headlines returned"}

    if not _FINBERT_AVAILABLE:
        return {
            **_NEUTRAL,
            "available": False,
            "n_articles": len(headlines),
            "headlines": headlines[:5],
            "reason": "transformers not installed — install with: pip install transformers torch",
        }

    try:
        clf = _hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,
            top_k=None,
        )
        bucket: dict[str, list] = {"positive": [], "negative": [], "neutral": []}
        for h in headlines[:max_articles]:
            try:
                for item in clf(h[:512])[0]:
                    lbl = item["label"].lower()
                    if lbl in bucket:
                        bucket[lbl].append(item["score"])
            except Exception:
                pass

        avg = {k: float(np.mean(v)) if v else 1 / 3 for k, v in bucket.items()}
        composite = avg["positive"] - avg["negative"]

        return {
            "score":      round(composite, 4),
            "positive":   round(avg["positive"], 4),
            "negative":   round(avg["negative"], 4),
            "neutral":    round(avg["neutral"], 4),
            "available":  True,
            "n_articles": len(headlines),
            "headlines":  headlines[:5],
        }
    except Exception as exc:
        return {**_NEUTRAL, "n_articles": len(headlines),
                "headlines": headlines[:5], "reason": f"FinBERT error: {exc}"}


def sentiment_label(score: float) -> str:
    if score > 0.15:
        return "Bullish"
    if score < -0.15:
        return "Bearish"
    return "Neutral"
