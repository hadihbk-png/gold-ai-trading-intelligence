"""
Economic event calendar: FOMC, CPI, and NFP proximity features.
All features use shift-based logic so no look-ahead leakage.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta


# ── Known FOMC decision dates (announcement day) ──────────────────────────────
_FOMC_DATES = [
    # 2020
    "2020-01-29","2020-03-15","2020-03-23","2020-04-29","2020-06-10",
    "2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    # 2021
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28",
    "2021-09-22","2021-11-03","2021-12-15",
    # 2022
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27",
    "2022-09-21","2022-11-02","2022-12-14",
    # 2023
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26",
    "2023-09-20","2023-11-01","2023-12-13",
    # 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31",
    "2024-09-18","2024-11-07","2024-12-18",
    # 2025
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30",
    "2025-09-17","2025-10-29","2025-12-10",
    # 2026
    "2026-01-28","2026-03-18","2026-04-29","2026-06-17","2026-07-29",
    "2026-09-16","2026-10-28","2026-12-09",
]


def _first_friday_of_month(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def _second_wednesday_of_month(year: int, month: int) -> date:
    d = date(year, month, 1)
    count = 0
    while True:
        if d.weekday() == 2:
            count += 1
            if count == 2:
                break
        d += timedelta(days=1)
    return d


def _generate_nfp_dates(start_year: int = 2019, end_year: int = 2027) -> list:
    return [str(_first_friday_of_month(y, m))
            for y in range(start_year, end_year + 1)
            for m in range(1, 13)]


def _generate_cpi_dates(start_year: int = 2019, end_year: int = 2027) -> list:
    return [str(_second_wednesday_of_month(y, m))
            for y in range(start_year, end_year + 1)
            for m in range(1, 13)]


_NFP_DATES = _generate_nfp_dates()
_CPI_DATES = _generate_cpi_dates()


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add economic event proximity features. No look-ahead — all distances
    are to the *next* future event from each bar's perspective.
    """
    df = df.copy()
    idx = pd.to_datetime(df.index).normalize()

    fomc_ts = pd.to_datetime(_FOMC_DATES)
    nfp_ts  = pd.to_datetime(_NFP_DATES)
    cpi_ts  = pd.to_datetime(_CPI_DATES)

    def _days_to_next(event_dates: pd.DatetimeIndex, cap: int = 30) -> np.ndarray:
        result = np.full(len(idx), float(cap))
        for i, d in enumerate(idx):
            future = event_dates[event_dates > d]
            if len(future) > 0:
                result[i] = min((future[0] - d).days, cap)
        return result

    def _days_since_last(event_dates: pd.DatetimeIndex, cap: int = 30) -> np.ndarray:
        result = np.full(len(idx), float(cap))
        for i, d in enumerate(idx):
            past = event_dates[event_dates <= d]
            if len(past) > 0:
                result[i] = min((d - past[-1]).days, cap)
        return result

    df["Days_to_FOMC"]  = _days_to_next(fomc_ts)
    df["Days_to_NFP"]   = _days_to_next(nfp_ts)
    df["Days_to_CPI"]   = _days_to_next(cpi_ts)
    df["Days_post_FOMC"] = _days_since_last(fomc_ts)

    df["Is_FOMC_Week"]  = (df["Days_to_FOMC"] <= 5).astype(int)
    df["Is_Pre_FOMC"]   = ((df["Days_to_FOMC"] > 0) & (df["Days_to_FOMC"] <= 3)).astype(int)
    df["Is_FOMC_Day"]   = (df["Days_to_FOMC"] == 0).astype(int)
    df["Is_Post_FOMC"]  = (df["Days_post_FOMC"] <= 2).astype(int)
    df["Is_NFP_Day"]    = (df["Days_to_NFP"] <= 1).astype(int)
    df["Is_CPI_Week"]   = (df["Days_to_CPI"] <= 5).astype(int)

    return df


def get_upcoming_events(n: int = 5) -> list[dict]:
    """Return next n economic events sorted by date."""
    today = pd.Timestamp.now().normalize()
    upcoming = []
    for label, dates in [
        ("FOMC", pd.to_datetime(_FOMC_DATES)),
        ("NFP",  pd.to_datetime(_NFP_DATES)),
        ("CPI",  pd.to_datetime(_CPI_DATES)),
    ]:
        future = dates[dates > today]
        if len(future) > 0:
            d = future[0]
            upcoming.append({
                "Event": label,
                "Date": str(d.date()),
                "Days Away": (d - today).days,
            })
    upcoming.sort(key=lambda x: x["Days Away"])
    return upcoming[:n]
