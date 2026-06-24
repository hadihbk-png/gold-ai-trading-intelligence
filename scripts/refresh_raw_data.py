#!/usr/bin/env python3
"""
Refresh data/raw_data.pkl from a clean (non-Yahoo-blocked) CI runner IP.

WHY
    Streamlit Cloud's datacenter IPs are blocked by Yahoo, so the app's live
    yfinance fetch fails on Cloud and falls back to whatever raw_data.pkl is
    committed. A GitHub-hosted runner IP is NOT blocked (verified 2026-06-24,
    8/8 tickers fresh), so this job rebuilds the cache here and commits it,
    keeping Cloud's model inputs current -- WITHOUT changing the data source
    the locked model was trained on (yfinance stays the source; zero skew).

HOW
    It calls the app's OWN builder, download_data(force_refresh=True), so the
    pickle is byte-faithful to what the app expects -- no schema drift, no
    reimplementation of the fetch.

CLOUD PANDAS COMPATIBILITY (added 2026-06-24)
    Newer pandas (3.x) stores string COLUMN NAMES as the new StringDtype.
    An older pandas on Streamlit Cloud cannot unpickle that
    ("Failed to load data: StringDtype(...)"). After validation, this script
    coerces the columns Index back to plain object dtype (data columns are left
    untouched) and re-writes the cache, so any pandas 2.x can read it. The
    workflow also pins pandas <2.3 as a first line of defense.

DEFENSIVE
    Validates the rebuilt frame (primary OHLCV present, every expected aux
    *_Close column present, sane row count, RECENT last bar). If anything is
    off (e.g. the fetch failed and download_data fell back to the stale cache),
    it exits non-zero so the workflow goes RED and the commit step is skipped --
    a bad/stale pickle is never committed over a good one.

Usage (from repo root):
    python scripts/refresh_raw_data.py
"""

from __future__ import annotations

import datetime as dt
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, PRIMARY_TICKER, TICKERS
from src.data_loader import download_data

MAX_AGE_DAYS = 4      # tolerate a weekend / single-holiday gap
MIN_ROWS = 1000       # ~4y of business days; guards against a truncated fetch
PRIMARY_COLS = ["Open", "High", "Low", "Close", "Volume"]


def expected_aux_columns() -> list[str]:
    """Mirror download_data's column naming for the non-primary tickers."""
    cols = []
    for t in TICKERS:
        if t == PRIMARY_TICKER:
            continue
        col = t.replace("=", "_").replace("^", "").replace("-", "_")
        cols.append(f"{col}_Close")
    return cols


def main() -> None:
    print("Rebuilding raw_data.pkl via download_data(force_refresh=True) ...")
    df = download_data(force_refresh=True)

    if df is None or getattr(df, "empty", True):
        print("FAIL: download_data returned an empty frame.")
        sys.exit(1)

    problems: list[str] = []

    for c in PRIMARY_COLS:
        if c not in df.columns:
            problems.append(f"missing primary column '{c}'")

    for c in expected_aux_columns():
        if c not in df.columns:
            problems.append(f"missing aux column '{c}'")

    if len(df) < MIN_ROWS:
        problems.append(f"only {len(df)} rows (< {MIN_ROWS})")

    last = pd.Timestamp(df.index[-1]).date()
    age = (dt.datetime.now(dt.timezone.utc).date() - last).days
    if age > MAX_AGE_DAYS:
        problems.append(
            f"last bar {last} is {age}d old (> {MAX_AGE_DAYS}) -- fetch likely "
            f"failed and fell back to the stale cache"
        )

    print(f"rows={len(df)}  last_bar={last}  age_days={age}  cols={len(df.columns)}")

    if problems:
        print("FAIL -- refusing to commit:")
        for p in problems:
            print(f"   - {p}")
        sys.exit(1)

    # -- Cloud pandas-compatibility guard --------------------------------------
    # Coerce the column Index to plain object dtype (leaves float data columns
    # as-is) and re-write the cache so an older pandas on Cloud can unpickle it.
    cache_path = os.path.join(DATA_DIR, "raw_data.pkl")
    before = str(df.columns.dtype)
    df.columns = df.columns.astype(object)
    with open(cache_path, "wb") as f:
        pickle.dump(df, f, protocol=4)
    print(
        f"OK: passed all checks. Column index dtype '{before}' -> 'object'; "
        f"re-wrote {cache_path} (protocol=4) for Cloud pandas compatibility."
    )


if __name__ == "__main__":
    main()
