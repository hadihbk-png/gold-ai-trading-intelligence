"""
Market data ingestion (yfinance) + optional FRED macro merge.
Data is disk-cached to avoid re-downloading on every Streamlit rerun.
"""

import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

from src.config import (
    DATA_DIR, PRIMARY_TICKER, TICKERS,
    START_DATE, END_DATE, TEST_YEARS,
)


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns that newer yfinance versions return."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def download_data(
    start: str = START_DATE,
    end: str   = END_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download price data for all tickers.
    Returns a DataFrame with OHLCV columns for the primary ticker (GC=F)
    and Close columns for each additional ticker, aligned to business-day dates.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "raw_data.pkl")

    def _load_cached() -> pd.DataFrame | None:
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if not force_refresh:
        cached = _load_cached()
        if cached is not None:
            return cached

    raw: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            df = _flatten(df)
            if not df.empty and "Close" in df.columns:
                raw[ticker] = df
        except Exception as e:
            print(f"Warning – {ticker}: {e}")

    if PRIMARY_TICKER not in raw:
        cached = _load_cached()
        if cached is not None:
            return cached
        raise RuntimeError(
            f"Primary ticker {PRIMARY_TICKER} failed to download and no cached data is available."
        )

    gold = raw[PRIMARY_TICKER][["Open", "High", "Low", "Close", "Volume"]].copy()

    for ticker in TICKERS:
        if ticker == PRIMARY_TICKER or ticker not in raw:
            continue
        col = ticker.replace("=", "_").replace("^", "").replace("-", "_")
        try:
            s = raw[ticker]["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            gold[f"{col}_Close"] = s
        except Exception as e:
            print(f"Warning – merging {ticker}: {e}")

    gold.dropna(subset=["Close", "High", "Low"], inplace=True)
    gold.sort_index(inplace=True)

    # Forward-fill sparse external series (bond yields, VIX has no weekend bars)
    ext = [c for c in gold.columns if c.endswith("_Close")]
    gold[ext] = gold[ext].ffill()

    with open(cache_path, "wb") as f:
        pickle.dump(gold, f)

    return gold


def get_train_test_split(df: pd.DataFrame):
    """Strictly time-ordered split.  Last TEST_YEARS year(s) → test set."""
    split = df.index[-1] - pd.DateOffset(years=TEST_YEARS)
    return df[df.index <= split].copy(), df[df.index > split].copy()
