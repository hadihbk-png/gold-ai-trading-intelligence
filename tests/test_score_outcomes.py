"""
Tests for src/score_outcomes.py.

1. Correctness — UP hit, DOWN miss, SIDEWAYS hit with flat return.
2. Idempotency — second run scores 0, already_scored == 1.
3. Chain safety — verify_chain True after update; record_hash byte-identical.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.track_logger import (
    LocalJsonStore,
    log_prediction,
    verify_chain,
    DIRECTION_THRESHOLD,
)
from src.score_outcomes import (
    ROUND_TRIP_COST_BPS,
    score_all_pending,
)

# ── Fixtures / helpers ─────────────────────────────────────────────────────────

AS_OF   = date(2024, 1, 15)   # Monday; _derive_horizon_resolves_at → 2024-01-16
RESOLVE = date(2024, 1, 16)
PRICE   = 2050.00


def _log(store, *, raw_signal: int, metal: str = "gold",
         as_of: date = AS_OF, price: float = PRICE) -> dict:
    return log_prediction(
        store=store,
        metal=metal,
        timestamp_utc=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        as_of_date=as_of,
        raw_signal=raw_signal,
        displayed_signal=["DOWN", "SIDEWAYS", "UP"][raw_signal],
        verdict="TEST",
        proba_vector=[0.7, 0.2, 0.1] if raw_signal == 0 else [0.1, 0.2, 0.7],
        price_at_decision=price,
        atr_pct=0.01,
        filter_state="TEST",
        model_version="test_v1",
        data_provenance={"source": "yfinance", "as_of_bar": str(as_of)},
        regime="trending",
    )


def _ohlcv(close: float, resolve_date: date = RESOLVE) -> pd.DataFrame:
    """Minimal single-bar OHLCV DataFrame at resolve_date."""
    idx = pd.DatetimeIndex([pd.Timestamp(resolve_date)])
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000},
        index=idx,
    )


@pytest.fixture
def store(tmp_path):
    return LocalJsonStore(str(tmp_path / "predictions.jsonl"))


# ── Test 1a: UP signal that hits ───────────────────────────────────────────────

def test_up_hit(store, monkeypatch):
    """UP signal, actual > threshold → realized=UP, hit=True, correct net return."""
    _log(store, raw_signal=2)
    actual = PRICE * (1 + DIRECTION_THRESHOLD + 0.002)  # +0.7% > +0.5%

    monkeypatch.setattr("src.score_outcomes._fetch_ohlcv", lambda _m: _ohlcv(actual))
    s = score_all_pending(store)

    assert s["scored"] == 1
    row = store.read_all()[0]
    assert row["realized_direction"] == 2
    assert row["hit"] is True
    expected = round(actual / PRICE - 1 - ROUND_TRIP_COST_BPS / 10_000, 8)
    assert abs(row["realized_return_net_of_cost"] - expected) < 1e-9
    assert row["scored_at"] is not None


# ── Test 1b: DOWN signal that misses (market went UP) ─────────────────────────

def test_down_miss(store, monkeypatch):
    """DOWN signal, actual +0.7% → realized=UP, hit=False."""
    _log(store, raw_signal=0)
    actual = PRICE * (1 + DIRECTION_THRESHOLD + 0.002)

    monkeypatch.setattr("src.score_outcomes._fetch_ohlcv", lambda _m: _ohlcv(actual))
    score_all_pending(store)

    row = store.read_all()[0]
    assert row["realized_direction"] == 2
    assert row["hit"] is False


# ── Test 1c: SIDEWAYS signal inside band → flat return ────────────────────────

def test_sideways_flat_return(store, monkeypatch):
    """SIDEWAYS signal, move within ±0.5% → realized=SIDEWAYS, return=0.0."""
    _log(store, raw_signal=1)
    actual = PRICE * 1.002  # +0.2%, inside band

    monkeypatch.setattr("src.score_outcomes._fetch_ohlcv", lambda _m: _ohlcv(actual))
    score_all_pending(store)

    row = store.read_all()[0]
    assert row["realized_direction"] == 1
    assert row["hit"] is True
    assert row["realized_return_net_of_cost"] == 0.0


# ── Test 1d: Holiday / gap — resolves to first bar after as_of, not calendar date ──

def test_holiday_gap_resolves_to_next_trading_bar(store, monkeypatch):
    """as_of = Friday, index skips Mon (holiday) and has Tue — must resolve to Tue.

    Validates shift(-1) semantics: the scorer looks for the first index bar
    strictly after as_of_date, so a missing Monday never causes skipped_no_close.
    """
    friday   = date(2024, 1, 12)   # Friday
    tuesday  = date(2024, 1, 16)   # Monday 2024-01-15 = MLK Day (market closed) → Tue
    tue_close = PRICE * 1.008      # +0.8% > +0.5% band → UP

    _log(store, raw_signal=2, as_of=friday)

    # Index has Friday + Tuesday only (Monday gap simulates the holiday)
    idx = pd.DatetimeIndex([pd.Timestamp(friday), pd.Timestamp(tuesday)])
    gap_df = pd.DataFrame(
        {"Open": [PRICE, tue_close], "High": [PRICE, tue_close],
         "Low":  [PRICE, tue_close], "Close": [PRICE, tue_close], "Volume": [1000, 1000]},
        index=idx,
    )
    monkeypatch.setattr("src.score_outcomes._fetch_ohlcv", lambda _m: gap_df)

    s = score_all_pending(store)

    assert s["scored"] == 1, f"expected scored=1, got {s}"
    assert s["skipped_no_close"] == 0
    row = store.read_all()[0]
    assert row["realized_direction"] == 2          # UP
    assert row["hit"] is True
    expected_return = round(tue_close / PRICE - 1 - ROUND_TRIP_COST_BPS / 10_000, 8)
    assert abs(row["realized_return_net_of_cost"] - expected_return) < 1e-9


# ── Test 2: Idempotency ────────────────────────────────────────────────────────

def test_idempotency(store, monkeypatch):
    """Second run scores 0 new rows; first-run outcome is preserved unchanged."""
    _log(store, raw_signal=2)
    actual = PRICE * 1.01

    monkeypatch.setattr("src.score_outcomes._fetch_ohlcv", lambda _m: _ohlcv(actual))

    s1 = score_all_pending(store)
    outcome_after_run1 = store.read_all()[0]["realized_return_net_of_cost"]

    s2 = score_all_pending(store)
    outcome_after_run2 = store.read_all()[0]["realized_return_net_of_cost"]

    assert s1["scored"] == 1
    assert s2["scored"] == 0
    assert s2["already_scored"] == 1
    assert outcome_after_run1 == outcome_after_run2


# ── Test 3: Chain safety ───────────────────────────────────────────────────────

def test_chain_safety(store, monkeypatch):
    """verify_chain True after scoring; record_hash byte-identical before and after."""
    _log(store, raw_signal=2)
    hash_before = store.read_all()[0]["record_hash"]

    actual = PRICE * 1.01
    monkeypatch.setattr("src.score_outcomes._fetch_ohlcv", lambda _m: _ohlcv(actual))
    s = score_all_pending(store)

    rows_after = store.read_all()
    assert s["chain_verified"] is True
    assert verify_chain(rows_after) is True
    assert rows_after[0]["record_hash"] == hash_before


# ── Test 4: Skipped when horizon not yet resolved ─────────────────────────────

def test_skipped_unresolved(store, monkeypatch):
    """Row whose resolve date is far in the future is counted as skipped_unresolved."""
    future_date = date(2099, 12, 31)
    _log(store, raw_signal=2, as_of=future_date)

    monkeypatch.setattr("src.score_outcomes._fetch_ohlcv", lambda _m: _ohlcv(PRICE * 1.01))
    s = score_all_pending(store)

    assert s["scored"] == 0
    assert s["skipped_unresolved"] == 1


# ── Test 5: Skipped when no close available for resolve date ──────────────────

def test_skipped_no_close(store, monkeypatch):
    """Row skipped when OHLCV has no bar for the resolve date."""
    _log(store, raw_signal=2)

    # Return an empty DataFrame so the resolve_ts lookup fails
    monkeypatch.setattr("src.score_outcomes._fetch_ohlcv", lambda _m: pd.DataFrame())
    s = score_all_pending(store)

    assert s["scored"] == 0
    assert s["skipped_no_close"] == 1


# ── Test 6: update_outcome rejects non-outcome keys ──────────────────────────

def test_update_outcome_rejects_hash_field(store):
    """update_outcome raises if caller tries to overwrite a _HASH_FIELDS key."""
    _log(store, raw_signal=2)
    pid = store.read_all()[0]["prediction_id"]

    with pytest.raises(ValueError, match="illegal keys"):
        store.update_outcome(pid, {"record_hash": "tampered"})


# ── Test 7: update_outcome raises on unknown prediction_id ────────────────────

def test_update_outcome_unknown_id(store):
    _log(store, raw_signal=2)
    with pytest.raises(KeyError):
        store.update_outcome("nonexistent_id", {"actual_price": 2100.0})
