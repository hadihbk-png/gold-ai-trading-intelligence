import pandas as pd
from datetime import datetime, timezone, timedelta
from src.data_loader import _drop_unsettled_tail


def _frame(dates):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="Date")
    return pd.DataFrame({"Close": range(len(idx))}, index=idx)


def test_drops_today_utc_bar():
    today = datetime.now(timezone.utc).date()
    df = _frame([today - timedelta(days=2), today - timedelta(days=1), today])
    out = _drop_unsettled_tail(df)
    assert out.index[-1] == pd.Timestamp(today - timedelta(days=1))
    assert pd.Timestamp(today) not in out.index
    assert len(out) == 2


def test_keeps_all_settled_bars():
    today = datetime.now(timezone.utc).date()
    df = _frame([today - timedelta(days=3), today - timedelta(days=2), today - timedelta(days=1)])
    out = _drop_unsettled_tail(df)
    assert len(out) == 3


def test_empty_frame_passthrough():
    assert _drop_unsettled_tail(pd.DataFrame()).empty


def test_non_datetime_index_passthrough():
    df = pd.DataFrame({"Close": [1, 2, 3]})
    assert len(_drop_unsettled_tail(df)) == 3


def test_trim_to_empty_returns_original():
    today = datetime.now(timezone.utc).date()
    df = _frame([today, today + timedelta(days=1)])
    assert len(_drop_unsettled_tail(df)) == 2
