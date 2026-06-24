#!/usr/bin/env python3
"""
yfinance-from-runner probe  (READ-ONLY, throwaway diagnostic).

WHY: Determine whether a GitHub-hosted runner IP can fetch fresh OHLC from
Yahoo via yfinance. If it can, the safe fix is viable: a scheduled Action
refreshes data/raw_data.pkl from a clean IP and commits it, with NO vendor
swap and NO change to the locked model's data source.

SAFETY: only downloads market data and prints a table. Writes nothing,
commits nothing, touches no model/threshold/locked file.

This script is the prototype of the eventual refresh job, minus the commit.
"""

import datetime as dt
import pandas as pd
import yfinance as yf

TICKERS = ["GC=F", "GLD", "DX-Y.NYB", "^TNX", "^VIX", "SI=F", "CL=F", "SPY"]
FRESH_MAX_AGE_DAYS = 4  # allows a weekend gap


def main():
    today = dt.datetime.now(dt.timezone.utc).date()
    print(f"runner date (UTC): {today}\n")
    print(f"{'ticker':<12}{'rows':>5}  {'last_bar':<11} {'age_d':>5}  status")
    print("-" * 56)

    fresh = 0
    for t in TICKERS:
        try:
            df = yf.download(t, period="10d", progress=False, auto_adjust=True)
            if df is None or df.empty:
                print(f"{t:<12}{0:>5}  {'--':<11} {'--':>5}  EMPTY (blocked / no data?)")
                continue
            last = pd.Timestamp(df.index[-1]).date()
            age = (today - last).days
            is_fresh = age <= FRESH_MAX_AGE_DAYS
            fresh += 1 if is_fresh else 0
            print(f"{t:<12}{len(df):>5}  {str(last):<11} {age:>5}  "
                  f"{'FRESH' if is_fresh else 'STALE'}")
        except Exception as e:
            print(f"{t:<12}{0:>5}  {'--':<11} {'--':>5}  ERROR: {str(e)[:60]}")

    print("-" * 56)
    print(f"\nFRESH tickers: {fresh}/{len(TICKERS)}")
    if fresh >= 6:
        print("VERDICT: runner IP is NOT Yahoo-blocked -> the safe refresh-and-commit "
              "path (yfinance preserved) is viable.")
    elif fresh == 0:
        print("VERDICT: runner IP appears blocked too -> safe path is out; fall back to "
              "the Twelve Data probe / vendor-swap path with the equivalence guard.")
    else:
        print("VERDICT: partial -> some tickers blocked/stale. Inspect which; the auxiliary "
              "series (^VIX/SPY/^TNX) are load-bearing, so partial coverage is not enough.")


if __name__ == "__main__":
    main()
