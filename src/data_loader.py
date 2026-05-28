"""
Market data ingestion (yfinance) + optional FRED macro merge.
Data is disk-cached to avoid re-downloading on every Streamlit rerun.
"""

import os
import pickle
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import yfinance as yf

from src.config import (
    DATA_DIR, PRIMARY_TICKER, TICKERS,
    TEST_YEARS,
)

# ── Fixed FX pegs (central bank rates, no API needed) ──────────────────────────
_FIXED_PEGS = {
    "USD": 1.0,
    "AED": 3.6725,   # UAE Central Bank
    "JOD": 0.709,    # Jordan Central Bank
    "SAR": 3.75,     # Saudi Arabia
}

# ── Approximate fallback live rates (used if yfinance fetch fails) ─────────────
_FX_FALLBACKS = {"GBP": 0.787, "EUR": 0.926, "INR": 83.5, "JPY": 150.0, "CNY": 7.25}

_CACHE_STALE_DAYS = 3  # treat disk cache as stale if last bar is older than this many calendar days


def _is_cache_stale(df: pd.DataFrame) -> bool:
    """Return True if df's last bar is more than _CACHE_STALE_DAYS calendar days old."""
    if df is None or df.empty:
        return True
    last_bar = pd.Timestamp(df.index[-1]).normalize().date()
    return (datetime.today().date() - last_bar).days > _CACHE_STALE_DAYS


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns that newer yfinance versions return."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def download_data(
    start: str = None,
    end: str   = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download price data for all tickers.
    Returns a DataFrame with OHLCV columns for the primary ticker (GC=F)
    and Close columns for each additional ticker, aligned to business-day dates.

    start/end default to None so the date range is computed at call time (not
    at import time), preventing stale date ranges in long-running servers.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.today() - timedelta(days=365 * 5 + 90)).strftime("%Y-%m-%d")

    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "raw_data.pkl")

    def _load_cached() -> pd.DataFrame | None:
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if not force_refresh:
        cached = _load_cached()
        if cached is not None and not _is_cache_stale(cached):
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


def get_live_spot_price(td_api_key: str = "", av_api_key: str = "") -> tuple:
    """Fetch XAU/USD spot price. Tries metals.live → Alpha Vantage → Twelve Data → yfinance.
    Returns (price, source_label)."""
    import requests

    # Method A: metals.live (no key needed — public free tier)
    try:
        resp = requests.get("https://api.metals.live/v1/spot/gold", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                data = data[0]
            price = float(data.get("gold") or data.get("price") or 0)
            if price > 100:
                return price, "Kitco Spot Price (live)"
    except Exception:
        pass

    # Method C: Alpha Vantage (free key from alphavantage.co)
    if av_api_key:
        try:
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "CURRENCY_EXCHANGE_RATE",
                    "from_currency": "XAU",
                    "to_currency": "USD",
                    "apikey": av_api_key,
                },
                timeout=5,
            )
            rate_info = resp.json().get("Realtime Currency Exchange Rate", {})
            price = float(rate_info.get("5. Exchange Rate", 0))
            if price > 100:
                return price, "Alpha Vantage (live)"
        except Exception:
            pass

    # Method D: Twelve Data (existing fallback)
    if td_api_key:
        try:
            resp = requests.get(
                "https://api.twelvedata.com/price",
                params={"symbol": "XAU/USD", "apikey": td_api_key},
                timeout=5,
            )
            price = float(resp.json()["price"])
            if price > 100:
                return price, "Twelve Data (live)"
        except Exception:
            pass

    # Method E: yfinance last resort
    return None, "Market Data (delayed)"


def get_lbma_fix() -> dict:
    """Get LBMA gold fix proxy via yfinance GLD ETF (GLD × 10 ≈ gold oz price).
    Returns {"am": float, "pm": float, "date": str} or {} on failure."""
    try:
        gld = yf.download("GLD", period="5d", progress=False, auto_adjust=True)
        gld = _flatten(gld)
        if gld.empty or "Open" not in gld.columns or "Close" not in gld.columns:
            return {}
        last = gld.iloc[-1]
        return {
            "am":   round(float(last["Open"])  * 10, 2),
            "pm":   round(float(last["Close"]) * 10, 2),
            "date": gld.index[-1].strftime("%Y-%m-%d"),
        }
    except Exception:
        return {}


def get_fx_rates() -> dict:
    """Fetch FX rates. Fixed pegs are hardcoded; floating currencies use yfinance.
    Returns {"rates": {code: units_per_usd}, "fetched_utc": iso_string}."""
    rates = dict(_FIXED_PEGS)

    # Batch download live pairs — all at once for speed
    live_pairs = {
        "GBP": ("GBPUSD=X", True),   # GBPUSD gives USD/GBP → invert for GBP/USD
        "EUR": ("EURUSD=X", True),   # EURUSD gives USD/EUR → invert
        "INR": ("INR=X",    False),  # USD/INR directly
        "JPY": ("JPY=X",    False),  # USD/JPY directly
        "CNY": ("CNY=X",    False),  # USD/CNY directly
    }
    tickers = [t for _, (t, _) in live_pairs.items()]
    try:
        raw = yf.download(tickers, period="2d", progress=False, auto_adjust=True)
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        for code, (ticker, invert) in live_pairs.items():
            try:
                price = float(closes[ticker].dropna().iloc[-1])
                if price > 0:
                    rates[code] = round(1.0 / price if invert else price, 6)
            except Exception:
                rates.setdefault(code, _FX_FALLBACKS.get(code, 1.0))
    except Exception:
        for code, fallback in _FX_FALLBACKS.items():
            rates.setdefault(code, fallback)

    return {
        "rates":       rates,
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
    }
