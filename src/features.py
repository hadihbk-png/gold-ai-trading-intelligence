import numpy as np
import pandas as pd
from src.config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, ATR_PERIOD,
    SMA_PERIODS, EMA_PERIODS, LAG_PERIODS,
    SIGNAL_THRESHOLD, DIRECTION_THRESHOLD,
)


# ── Safe macro data fetcher ────────────────────────────────────────────────────

def fetch_macro_data(ticker: str, start: str, end: str, name: str):
    """Fetch a single macro series via yfinance; returns None on any failure."""
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if hasattr(data.columns, "levels"):
            data.columns = [c[0] for c in data.columns]
        if data.empty:
            return None
        return data["Close"].rename(name)
    except Exception:
        return None


# ── Regime classification ──────────────────────────────────────────────────────

def classify_regime(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Classify each bar into 'high_vol', 'trending', or 'neutral' regime.
    Used to select which regime-conditional model applies at inference.
    """
    df = df.copy()
    if "ATR" not in df.columns or "Close" not in df.columns:
        df["regime"] = "neutral"
        return df

    atr_pct     = df["ATR"] / df["Close"].replace(0, np.nan)
    atr_median  = atr_pct.rolling(60).median()
    price_chg   = df["Close"].pct_change(window).abs()

    df["regime"] = "neutral"
    df.loc[atr_pct > atr_median * 1.3, "regime"] = "high_vol"
    df.loc[
        (price_chg > 0.03) & (atr_pct <= atr_median * 1.3),
        "regime",
    ] = "trending"
    return df


# ── Feature drift detection ────────────────────────────────────────────────────

def detect_feature_drift(live_features_dict: dict, stats_dict: dict) -> tuple:
    """
    Compare live feature values against training distribution statistics.

    Returns (drift_score, drifted_list) where drift_score ∈ [0, 1]
    and drifted_list contains per-feature detail dicts.
    """
    drifted = []
    try:
        for feat, val in live_features_dict.items():
            if feat not in stats_dict:
                continue
            std = stats_dict[feat].get("std", 0)
            if std == 0 or std is None:
                continue
            try:
                z = abs((float(val) - stats_dict[feat]["mean"]) / std)
            except (TypeError, ValueError):
                continue
            if z > 3.0:
                drifted.append({
                    "feature":    feat,
                    "z_score":    round(z, 1),
                    "live_value": round(float(val), 4),
                    "train_mean": round(stats_dict[feat]["mean"], 4),
                })
        drift_score = min(len(drifted) / 10, 1.0)
        return drift_score, drifted
    except Exception:
        return 0.0, []


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

    # ── DOWN-class sensitivity features ──────────────────────────────────────
    df["momentum_3d"]  = close.pct_change(3)
    df["momentum_5d"]  = close.pct_change(5)
    df["momentum_10d"] = close.pct_change(10)

    df["rsi_overbought"] = (df["RSI"] > 70).astype(int)
    df["rsi_oversold"]   = (df["RSI"] < 30).astype(int)
    df["rsi_extreme"]    = ((df["RSI"] > 75) | (df["RSI"] < 25)).astype(int)

    df["dist_from_20d_high"] = close / high.rolling(20).max().replace(0, np.nan) - 1
    df["dist_from_20d_low"]  = close / low.rolling(20).min().replace(0, np.nan) - 1

    up_days = close > close.shift(1)
    df["consec_up_count"] = (
        up_days.groupby((up_days != up_days.shift()).cumsum()).cumcount() + 1
    )
    df.loc[~up_days, "consec_up_count"] = 0

    down_days = close < close.shift(1)
    df["consec_down_count"] = (
        down_days.groupby((down_days != down_days.shift()).cumsum()).cumcount() + 1
    )
    df.loc[~down_days, "consec_down_count"] = 0

    df["DayOfWeek"] = df.index.dayofweek   # 0=Mon … 4=Fri
    df["Month"]     = df.index.month       # 1–12

    # ── 4B: Seasonality features ──────────────────────────────────────────────
    df["quarter"]       = df.index.quarter
    df["day_of_week"]   = df.index.dayofweek
    try:
        df["week_of_year"] = df.index.isocalendar().week.astype(int)
    except Exception:
        df["week_of_year"] = df.index.to_series().apply(lambda x: x.isocalendar()[1])
    df["q4_demand"]      = df["Month"].isin([10, 11, 12]).astype(int)
    df["january_effect"] = (df["Month"] == 1).astype(int)
    df["summer_low_vol"] = df["Month"].isin([6, 7, 8]).astype(int)
    df["is_monday"]      = (df["DayOfWeek"] == 0).astype(int)
    df["is_friday"]      = (df["DayOfWeek"] == 4).astype(int)

    # ── 4A: Extended macro features & gold-macro relationships ────────────────
    # Reuse existing _Close columns (already downloaded by data_loader)
    _dxy_s   = df.get("DX_Y_NYB_Close")
    _tnx_s   = df.get("TNX_Close")
    _oil_s   = df.get("CL_F_Close")
    _vix_s   = df.get("VIX_Close")
    _spy_s   = df.get("SPY_Close")

    def _safe_series(src):
        return src if src is not None else pd.Series(np.nan, index=df.index)

    for _col_name, _src in [
        ("dxy",   _dxy_s),
        ("us10y", _tnx_s),
        ("oil",   _oil_s),
        ("vix",   _vix_s),
        ("sp500", _spy_s),
    ]:
        if _src is None:
            continue
        _s = _src.copy()
        df[f"{_col_name}_close"]    = _s
        df[f"{_col_name}_chg1d"]    = _s.pct_change(1)
        df[f"{_col_name}_chg5d"]    = _s.pct_change(5)
        _sma20 = _s.rolling(20).mean()
        df[f"{_col_name}_vs_sma20"] = _s / _sma20.replace(0, np.nan) - 1

    # OIL derived features (not covered in existing macro block)
    if _oil_s is not None:
        _oil_ma20 = _oil_s.rolling(20).mean()
        df["oil_vs_sma20_slope"] = _oil_s.pct_change(10) * 100

    # Gold-macro cross-asset relationships
    if _dxy_s is not None:
        df["gold_dxy_corr20"] = close.rolling(20).corr(_dxy_s)
    if _oil_s is not None:
        df["gold_oil_corr20"] = close.rolling(20).corr(_oil_s)
    if _tnx_s is not None and _vix_s is not None:
        df["real_rate_proxy"] = _tnx_s - _vix_s * 0.1
    if _vix_s is not None and _spy_s is not None:
        _vix_chg1d  = _vix_s.pct_change(1)
        _spy_chg1d  = _spy_s.pct_change(1)
        df["risk_off_index"] = (_vix_chg1d * -1 + _spy_chg1d * -1) / 2

    # Forward-fill then back-fill any NaN introduced by macro series gaps
    _new_macro_cols = [c for c in df.columns if c.endswith(("_chg1d", "_chg5d",
                       "_vs_sma20", "_close", "_corr20"))
                       if c in ("dxy_close", "us10y_close", "oil_close",
                                "vix_close", "sp500_close",
                                "dxy_chg1d", "us10y_chg1d", "oil_chg1d",
                                "vix_chg1d", "sp500_chg1d",
                                "dxy_chg5d", "us10y_chg5d", "oil_chg5d",
                                "vix_chg5d", "sp500_chg5d",
                                "dxy_vs_sma20", "us10y_vs_sma20", "oil_vs_sma20",
                                "vix_vs_sma20", "sp500_vs_sma20",
                                "gold_dxy_corr20", "gold_oil_corr20",
                                "real_rate_proxy", "risk_off_index")]
    for _nc in _new_macro_cols:
        if _nc in df.columns:
            df[_nc] = df[_nc].ffill().bfill()

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
    """Return model input columns — no raw OHLCV, no targets, no ext _Close, no string cols."""
    exclude = {
        "Open", "High", "Low", "Close", "Volume",
        "Target_Close", "Target_Return", "Target_Signal", "Target_Direction",
        "regime",  # string column — used for regime model selection, not as feature
    }
    exclude.update(c for c in df.columns if c.endswith("_Close"))
    # Also exclude any remaining object/string dtype columns
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    return [c for c in df.columns if c not in exclude and c in numeric_cols]
