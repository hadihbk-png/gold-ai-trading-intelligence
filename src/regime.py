"""
Market regime detection.

Regimes (rule-based composite):
  0 – High Volatility   : VIX > 25 or realized vol 1.5× median
  1 – Low Volatility    : VIX < 15 and realized vol < 0.5× median
  2 – Risk-On           : Yield curve steepening, equities bid
  3 – Risk-Off          : Inverted/flat yield curve, VIX elevated
  4 – Inflationary      : CPI YoY > 3%  and real rates negative
  5 – Neutral           : None of the above

Each bar receives exactly one regime label (priority order: 0 > 4 > 3 > 2 > 1 > 5).
"""

import numpy as np
import pandas as pd
from src.config import (
    VIX_HIGH_THRESHOLD, VIX_LOW_THRESHOLD,
    REGIME_LOOKBACK, CPI_INFLATION_LEVEL, YIELD_CURVE_INVERT,
)

REGIME_LABELS = {
    0: "High Volatility",
    1: "Low Volatility",
    2: "Risk-On",
    3: "Risk-Off",
    4: "Inflationary",
    5: "Neutral",
}

REGIME_COLORS = {
    0: "#FF4B4B",   # red
    1: "#00CC88",   # green
    2: "#00BFFF",   # blue
    3: "#FFA500",   # orange
    4: "#FF69B4",   # pink
    5: "#888888",   # grey
}

REGIME_EMOJI = {
    0: "🔴", 1: "🟢", 2: "🔵", 3: "🟠", 4: "🩷", 5: "⚪",
}


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """
    Assign a regime integer to every row.  Returns a Series indexed like df.
    All inputs used are ALREADY lagged or computed purely from past data.
    """
    n = len(df)
    regime = np.full(n, 5, dtype=int)   # default: Neutral

    # ── VIX / realized volatility ────────────────────────────────────────────
    vix = df.get("VIX_Close")
    vol20 = df.get("Volatility_20")

    if vix is not None:
        vix_arr = vix.values
        regime[vix_arr > VIX_HIGH_THRESHOLD] = 0
        # Only set low-vol if not already marked high-vol
        low_mask = (vix_arr < VIX_LOW_THRESHOLD) & (regime != 0)
        regime[low_mask] = 1

    if vol20 is not None:
        vol_arr = vol20.values
        vol_median = pd.Series(vol_arr).rolling(REGIME_LOOKBACK).median().values
        high_vol_mask = (vol_arr > vol_median * 1.5) & ~np.isnan(vol_median)
        regime[high_vol_mask] = 0
        low_vol_mask = (vol_arr < vol_median * 0.5) & ~np.isnan(vol_median) & (regime != 0)
        regime[low_vol_mask] = 1

    # ── Inflationary ─────────────────────────────────────────────────────────
    cpi_yoy  = df.get("MACRO_CPI_YoY")
    real_rate = df.get("MACRO_RealRate")

    if cpi_yoy is not None:
        cpi_arr = cpi_yoy.values
        infl_mask = (cpi_arr > CPI_INFLATION_LEVEL) & (regime == 5)
        if real_rate is not None:
            infl_mask = infl_mask & (real_rate.values < 0.5)
        regime[infl_mask] = 4

    # ── Risk-Off / Risk-On ────────────────────────────────────────────────────
    yc = df.get("MACRO_YieldCurve")

    if yc is not None:
        yc_arr = yc.values
        yc_ma  = pd.Series(yc_arr).rolling(20).mean().values

        riskoff_mask = (yc_arr < YIELD_CURVE_INVERT) & (regime == 5)
        regime[riskoff_mask] = 3

        riskon_mask = (yc_arr > yc_ma) & ~np.isnan(yc_ma) & (regime == 5)
        regime[riskon_mask] = 2

    return pd.Series(regime, index=df.index, name="Regime")


def get_current_regime(df: pd.DataFrame) -> dict:
    """Detect regimes and return summary for the latest bar."""
    series = detect_regime(df)
    current = int(series.iloc[-1])
    return {
        "regime_int":     current,
        "regime_label":   REGIME_LABELS[current],
        "regime_color":   REGIME_COLORS[current],
        "regime_emoji":   REGIME_EMOJI[current],
        "regime_history": series,
        "regime_counts":  series.value_counts().sort_index(),
    }


def regime_adjusted_size_factor(regime_int: int) -> float:
    """Scale position size by regime (high vol → smaller, risk-on → normal)."""
    factors = {0: 0.5, 1: 1.0, 2: 1.0, 3: 0.6, 4: 0.7, 5: 0.8}
    return factors.get(regime_int, 0.8)
