import numpy as np
import pandas as pd
from src.config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, ATR_PERIOD,
    SMA_PERIODS, EMA_PERIODS, LAG_PERIODS,
    SIGNAL_THRESHOLD, DIRECTION_THRESHOLD,
)


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series, fast=MACD_FAST, slow=MACD_SLOW, sig=MACD_SIGNAL):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    signal = line.ewm(span=sig, adjust=False).mean()
    return line, signal, line - signal


def _bollinger(series, period=BB_PERIOD, n_std=BB_STD):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    bw = (upper - lower) / sma.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, sma, lower, bw, pct_b


def _atr(high, low, close, period=ATR_PERIOD):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(high, low, close, period=14):
    """Average Directional Index, +DI, -DI using Wilder's smoothing."""
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    up_move   = high - prev_high
    down_move = prev_low - low
    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )
    alpha    = 1.0 / period
    atr_w    = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di  = (100 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
                / atr_w.replace(0, np.nan))
    minus_di = (100 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
                / atr_w.replace(0, np.nan))
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


# ── Main feature builder ──────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators and lag features. No look-ahead."""
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["RSI"] = _rsi(close)

    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = _macd(close)

    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"], df["BB_Width"], df["BB_PctB"] = _bollinger(close)

    df["ATR"] = _atr(high, low, close)
    df["ATR_Pct"] = df["ATR"] / close.replace(0, np.nan)

    for p in SMA_PERIODS:
        df[f"SMA_{p}"] = close.rolling(p).mean()
        df[f"Price_vs_SMA{p}"] = close / df[f"SMA_{p}"].replace(0, np.nan) - 1

    for p in EMA_PERIODS:
        df[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()
        df[f"Price_vs_EMA{p}"] = close / df[f"EMA_{p}"].replace(0, np.nan) - 1

    df["SMA20_50_Diff"] = df["SMA_20"] - df["SMA_50"]
    df["SMA50_200_Diff"] = df["SMA_50"] - df["SMA_200"]
    df["EMA20_50_Diff"] = df["EMA_20"] - df["EMA_50"]

    df["Daily_Return"] = close.pct_change()
    df["Log_Return"] = np.log(close / close.shift(1))
    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std()

    for p in [5, 10, 20]:
        df[f"Mom_{p}"] = close / close.shift(p).replace(0, np.nan) - 1

    df["HL_Range"] = (high - low) / close.replace(0, np.nan)
    df["OC_Range"] = (close - df["Open"]) / close.replace(0, np.nan)

    if "Volume" in df.columns:
        vol_ma = df["Volume"].rolling(20).mean()
        df["Volume_Ratio"] = df["Volume"] / vol_ma.replace(0, np.nan)

    for lag in LAG_PERIODS:
        df[f"Return_Lag{lag}"] = df["Daily_Return"].shift(lag)
        df[f"Close_Lag{lag}_Pct"] = close.pct_change(lag).shift(1)

    ext_cols = [c for c in df.columns if c.endswith("_Close")]
    for col in ext_cols:
        prefix = col[:-6]
        df[f"{prefix}_Ret"] = df[col].pct_change()
        df[f"{prefix}_Ret_Lag1"] = df[f"{prefix}_Ret"].shift(1)

    # ── Macro regime awareness features ──────────────────────────────────────
    # SMA trend slopes — rate of change over 20 bars (look-ahead safe)
    df["SMA50_Slope"]  = df["SMA_50"].pct_change(20) * 100
    df["SMA200_Slope"] = df["SMA_200"].pct_change(20) * 100
    df["SMA50_vs_SMA200_Ratio"] = df["SMA_50"] / df["SMA_200"].replace(0, np.nan) - 1

    # DXY (dollar index) — key inverse driver for gold
    _dxy = df.get("DX_Y_NYB_Close")
    if _dxy is not None:
        _dxy_ma20 = _dxy.rolling(20).mean()
        df["DXY_vs_SMA20"] = _dxy / _dxy_ma20.replace(0, np.nan) - 1
        df["DXY_Slope10"]  = _dxy.pct_change(10) * 100
        df["DXY_Slope20"]  = _dxy.pct_change(20) * 100

    # 10Y Treasury yield — rising yields = gold headwind
    _tnx = df.get("TNX_Close")
    if _tnx is not None:
        _tnx_ma20 = _tnx.rolling(20).mean()
        df["TNX_vs_SMA20"] = _tnx / _tnx_ma20.replace(0, np.nan) - 1
        df["TNX_Slope20"]  = _tnx.diff(20)    # yield change in pp (not %)
        df["TNX_Chg5"]     = _tnx.diff(5)     # short-term yield momentum

    # ATR volatility regime — ratio to 63-day rolling mean
    if "ATR_Pct" in df.columns:
        _atr_ma63 = df["ATR_Pct"].rolling(63).mean()
        df["ATR_Regime"] = df["ATR_Pct"] / _atr_ma63.replace(0, np.nan)

    # VIX regime features
    _vix = df.get("VIX_Close")
    if _vix is not None:
        _vix_ma20 = _vix.rolling(20).mean()
        df["VIX_vs_SMA20"] = _vix / _vix_ma20.replace(0, np.nan) - 1
        df["VIX_Slope10"]  = _vix.pct_change(10) * 100

    # SPY (equity risk sentiment) and composite risk-off score
    _spy = df.get("SPY_Close")
    if _spy is not None:
        _spy_ma20 = _spy.rolling(20).mean()
        df["SPY_vs_SMA20"] = _spy / _spy_ma20.replace(0, np.nan) - 1
        df["SPY_Slope10"]  = _spy.pct_change(10) * 100
        if "VIX_vs_SMA20" in df.columns:
            # Positive = risk-off (VIX elevated above its MA + SPY below its MA)
            df["Risk_Off_Score"] = df["VIX_vs_SMA20"] - df["SPY_vs_SMA20"]

    # ── Momentum / breakout additions ────────────────────────────────────────
    df["MACD_Hist_Slope"] = df["MACD_Hist"].diff(3)
    df["RSI_Slope"]       = df["RSI"].diff(5)
    df["ADX"], df["Plus_DI"], df["Minus_DI"] = _adx(high, low, close)
    df["DI_Spread"] = df["Plus_DI"] - df["Minus_DI"]
    for n in [20, 50]:
        rolling_high = high.rolling(n).max().replace(0, np.nan)
        rolling_low  = low.rolling(n).min().replace(0, np.nan)
        df[f"Breakout_High{n}"] = close / rolling_high - 1
        df[f"Breakout_Low{n}"]  = close / rolling_low  - 1

    # ── Phase 3A: enriched indicators ────────────────────────────────────────
    df["EMA_9"]  = close.ewm(span=9,  adjust=False).mean()
    df["EMA_21"] = close.ewm(span=21, adjust=False).mean()
    df["EMA9_21_Cross"] = (df["EMA_9"] > df["EMA_21"]).astype(int)

    df["ROC_5"]  = close.pct_change(5)  * 100
    df["ROC_10"] = close.pct_change(10) * 100

    _wr_high  = high.rolling(14).max()
    _wr_low   = low.rolling(14).min()
    _wr_range = (_wr_high - _wr_low).replace(0, np.nan)
    df["Williams_R"] = -100 * (_wr_high - close) / _wr_range

    df["DayOfWeek"] = df.index.dayofweek   # 0=Mon … 4=Fri
    df["Month"]     = df.index.month       # 1–12

    # Targets (shift -1 → next-day; last row gets NaN → dropped at training)
    df["Target_Close"] = close.shift(-1)
    df["Target_Return"] = close.pct_change().shift(-1)

    ret_next = df["Target_Return"]

    # Legacy ±0.3% signal (kept for backward compatibility)
    df["Target_Signal"] = np.select(
        [ret_next < -SIGNAL_THRESHOLD, ret_next > SIGNAL_THRESHOLD],
        [0, 2], default=1,
    ).astype(float)
    df.loc[ret_next.isna(), "Target_Signal"] = np.nan

    # Primary ±1.0% directional target: 0=DOWN, 1=SIDEWAYS, 2=UP
    df["Target_Direction"] = np.select(
        [ret_next < -DIRECTION_THRESHOLD, ret_next > DIRECTION_THRESHOLD],
        [0, 2], default=1,
    ).astype(float)
    df.loc[ret_next.isna(), "Target_Direction"] = np.nan

    # ── Economic event features ───────────────────────────────────────────────
    try:
        from src.events import add_event_features
        df = add_event_features(df)
    except Exception:
        pass

    # ── Weekly trend features ─────────────────────────────────────────────────
    try:
        from src.timeframes import add_weekly_features
        df = add_weekly_features(df)
    except Exception:
        pass

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return model input columns — no raw OHLCV, no targets, no ext _Close."""
    exclude = {
        "Open", "High", "Low", "Close", "Volume",
        "Target_Close", "Target_Return", "Target_Signal", "Target_Direction",
    }
    exclude.update(c for c in df.columns if c.endswith("_Close"))
    return [c for c in df.columns if c not in exclude]
