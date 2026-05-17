"""
Multi-timeframe architecture:
  - Weekly trend features derived from daily data (no look-ahead)
  - 4H confirmation signal using 1h yfinance data resampled to 4H
  - Weekly filter to block counter-trend daily signals
"""

import numpy as np
import pandas as pd


# ── Weekly features ────────────────────────────────────────────────────────────

def add_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly-resolution features from daily data.
    Uses .shift(1) on weekly series before reindexing to daily
    so no same-week future data leaks into daily bars.
    """
    df = df.copy()
    close = df["Close"]

    # Weekly close (last trading day of each ISO week), shifted to prevent leakage
    weekly_close = close.resample("W-FRI").last().shift(1)
    w_daily = weekly_close.reindex(df.index, method="ffill")

    # Weekly SMA20 (20 weeks ≈ 5 months) and SMA50 (50 weeks ≈ 1 year)
    w_sma20 = weekly_close.rolling(20).mean().reindex(df.index, method="ffill")
    w_sma50 = weekly_close.rolling(50).mean().reindex(df.index, method="ffill")

    df["Weekly_Close"]   = w_daily
    df["Weekly_SMA20"]   = w_sma20
    df["Weekly_SMA50"]   = w_sma50
    df["Weekly_Trend"]   = np.where(
        w_sma20 > w_sma50, 1,
        np.where(w_sma20 < w_sma50, -1, 0),
    ).astype(float)

    # Weekly momentum at 1, 4, 8 weeks
    df["Weekly_Mom1"]  = (weekly_close.pct_change(1)).reindex(df.index, method="ffill")
    df["Weekly_Mom4"]  = (weekly_close.pct_change(4)).reindex(df.index, method="ffill")
    df["Weekly_Mom8"]  = (weekly_close.pct_change(8)).reindex(df.index, method="ffill")

    # Weekly realized volatility (12-week rolling std of weekly returns)
    weekly_ret = weekly_close.pct_change()
    df["Weekly_Vol"] = weekly_ret.rolling(12).std().reindex(df.index, method="ffill")

    # Weekly RSI (14-period on weekly close)
    wr = weekly_close.diff()
    wg = wr.clip(lower=0).ewm(com=13, min_periods=14).mean()
    wl = (-wr.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    w_rsi = 100 - (100 / (1 + wg / wl.replace(0, np.nan)))
    df["Weekly_RSI"] = w_rsi.reindex(df.index, method="ffill")

    return df


# ── 4H signal from intraday data ───────────────────────────────────────────────

def get_4h_signal(ticker: str = "GC=F") -> dict:
    """
    Download 1h data from yfinance and resample to 4H.
    Produces a directional signal from EMA9/EMA21 crossover + RSI.

    Returns dict with keys: signal (0/1/2), confidence (float), available (bool).
    Falls back to neutral on any error.
    """
    _neutral = {"signal": 1, "confidence": 0.45, "available": False,
                "ema9": None, "ema21": None, "rsi14": None, "latest_date": None}
    try:
        import yfinance as yf
        h1 = yf.download(ticker, period="59d", interval="1h",
                         progress=False, auto_adjust=True)
        if h1.empty:
            return _neutral

        if isinstance(h1.columns, pd.MultiIndex):
            h1.columns = h1.columns.droplevel(1)

        h4 = h1.resample("4h").agg({
            "Open": "first", "High": "max",
            "Low": "min", "Close": "last", "Volume": "sum",
        }).dropna()

        close = h4["Close"]
        if len(close) < 30:
            return {**_neutral, "available": True}

        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()

        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
        rsi14 = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        e9  = float(ema9.iloc[-1])
        e21 = float(ema21.iloc[-1])
        rsi = float(rsi14.iloc[-1]) if not np.isnan(rsi14.iloc[-1]) else 50.0

        spread_pct = abs(e9 - e21) / e21 if e21 != 0 else 0
        base_conf  = min(0.85, 0.55 + spread_pct * 8)

        if e9 > e21 and rsi > 50:
            sig, conf = 2, base_conf
        elif e9 < e21 and rsi < 50:
            sig, conf = 0, base_conf
        else:
            sig, conf = 1, 0.45

        return {
            "signal": sig,
            "confidence": round(conf, 3),
            "available": True,
            "ema9": round(e9, 2),
            "ema21": round(e21, 2),
            "rsi14": round(rsi, 1),
            "latest_date": str(h4.index[-1]),
        }
    except Exception as exc:
        return {**_neutral, "error": str(exc)}


# ── Weekly trend filter ────────────────────────────────────────────────────────

def apply_weekly_filter(daily_signal: int, weekly_trend: int) -> int:
    """
    Block counter-trend daily signals:
      - Weekly uptrend  (1)  : suppress SELL/DOWN signals
      - Weekly downtrend (-1): suppress BUY/UP signals
      - Weekly neutral  (0)  : pass all signals through
    """
    if weekly_trend == 1 and daily_signal == 0:
        return 1
    if weekly_trend == -1 and daily_signal == 2:
        return 1
    return daily_signal
