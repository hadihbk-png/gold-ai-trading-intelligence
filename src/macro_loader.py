"""
FRED macro data ingestion.

Series downloaded:
  FEDFUNDS  – Fed Funds effective rate
  DGS10     – 10-Year Treasury constant-maturity yield
  DGS2      – 2-Year Treasury constant-maturity yield
  T10YIE    – 10-Year breakeven inflation rate
  CPIAUCSL  – CPI for All Urban Consumers (not seasonally adjusted)
  UNRATE    – Civilian unemployment rate

Derived features added:
  YieldCurve      = DGS10 - DGS2         (steepness indicator)
  YieldCurve_Chg  = 1-day change in YieldCurve
  RealRate        = DGS10 - T10YIE       (real 10Y yield)
  CPI_YoY         = 12-month % change in CPI
  FedFunds_Chg    = 1-step change in FEDFUNDS
  FedFunds_Gap    = FEDFUNDS - DGS2      (policy-market spread)
"""

import os
import pickle
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.config import DATA_DIR, FRED_API_KEY, FRED_SERIES


def download_fred(start: str = None,
                  end: str = None,
                  force_refresh: bool = False) -> pd.DataFrame:
    """
    Download FRED series and cache to disk.
    Returns a daily (business-day) DataFrame.
    Returns empty DataFrame if FRED_API_KEY is unset.

    start/end default to None so dates are computed at call time, not import time.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.today() - timedelta(days=365 * 5 + 90)).strftime("%Y-%m-%d")

    cache_path = os.path.join(DATA_DIR, "macro_data.pkl")
    os.makedirs(DATA_DIR, exist_ok=True)

    def _load_cached() -> pd.DataFrame | None:
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if not force_refresh and os.path.exists(cache_path):
        age_h = (datetime.now().timestamp() - os.path.getmtime(cache_path)) / 3600
        if age_h < 24:
            cached = _load_cached()
            if cached is not None:
                return cached

    api_key = os.getenv("FRED_API_KEY", FRED_API_KEY)
    if not api_key:
        return pd.DataFrame()

    try:
        from fredapi import Fred  # type: ignore
    except ImportError:
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    frames = {}
    for sid in FRED_SERIES:
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            s.name = sid
            frames[sid] = s
        except Exception as exc:
            print(f"FRED {sid}: {exc}")

    if not frames:
        cached = _load_cached()
        return cached if cached is not None else pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.resample("B").last()   # business-day grid
    df = df.ffill()                # sparse monthly series → forward-fill

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)

    return df


def add_macro_features(price_df: pd.DataFrame,
                       macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge FRED data into the price DataFrame and engineer derived features.
    All macro values are lagged by 1 day to prevent look-ahead.
    """
    if macro_df is None or macro_df.empty:
        return price_df

    df = price_df.copy()

    # Align to price index (forward-fill gaps, then lag 1 to avoid leakage)
    m = macro_df.reindex(df.index).ffill().shift(1)

    for col in m.columns:
        df[f"MACRO_{col}"] = m[col]

    # ── Derived features ──────────────────────────────────────────────────────
    dgs10 = m.get("DGS10")
    dgs2  = m.get("DGS2")
    t10ie = m.get("T10YIE")
    cpi   = m.get("CPIAUCSL")
    ff    = m.get("FEDFUNDS")

    if dgs10 is not None and dgs2 is not None:
        yc = dgs10 - dgs2
        df["MACRO_YieldCurve"]     = yc
        df["MACRO_YieldCurve_Chg"] = yc.diff()
        df["MACRO_YieldCurve_MA20"] = yc.rolling(20).mean()

    if dgs10 is not None and t10ie is not None:
        df["MACRO_RealRate"] = dgs10 - t10ie

    if cpi is not None:
        # Annualised YoY change (CPI is monthly; approximate with 252-day rolling)
        df["MACRO_CPI_YoY"] = cpi.pct_change(12) * 100
        df["MACRO_CPI_MoM"] = cpi.pct_change(1) * 100

    if ff is not None:
        df["MACRO_FedFunds_Chg"]  = ff.diff()
        if dgs2 is not None:
            df["MACRO_FedFunds_Gap"] = ff - dgs2  # policy vs market

    return df


def get_macro_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("MACRO_")]
