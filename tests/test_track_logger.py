"""
Tests for src/track_logger.py — capture-only prediction logger core.

Covers:
  1. Three metals → 3 distinct "appended" rows
  2. Duplicate re-log → "duplicate" status, no new rows
  3. verify_chain returns True after writes
  4. Field mutation → verify_chain returns False
  5. Failing store.append() → "error" status, no exception raised
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from src.track_logger import LocalJsonStore, log_prediction, verify_chain

# ── Shared fixtures / helpers ──────────────────────────────────────────────────

COMMON_DATE = date(2024, 1, 15)
COMMON_MODEL = "v6c"
METALS = ["gold", "silver", "platinum"]


def _kwargs(metal: str) -> dict:
    return dict(
        metal=metal,
        timestamp_utc=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        as_of_date=COMMON_DATE,
        raw_signal=2,
        displayed_signal="UP",
        verdict="STRONG_UP",
        proba_vector=[0.10, 0.20, 0.70],
        price_at_decision=2050.50,
        atr_pct=0.012345,
        filter_state="BULLISH",
        model_version=COMMON_MODEL,
        data_provenance="yfinance/GLD",
        regime="trending",
    )


def _log_all(store) -> list[dict]:
    return [log_prediction(store=store, **_kwargs(m)) for m in METALS]


@pytest.fixture
def tmp_store(tmp_path):
    path = str(tmp_path / "predictions.jsonl")
    return LocalJsonStore(path)


# ── Test 1: three distinct metals → 3 appended rows ───────────────────────────

def test_three_metals_appended(tmp_store):
    results = _log_all(tmp_store)

    assert all(r["status"] == "appended" for r in results), \
        f"Expected all 'appended', got: {[r['status'] for r in results]}"
    assert len(tmp_store.read_all()) == 3


# ── Test 2: re-running same logs → duplicate, still 3 rows ────────────────────

def test_duplicate_no_new_rows(tmp_store):
    _log_all(tmp_store)

    dups = _log_all(tmp_store)

    assert all(r["status"] == "duplicate" for r in dups), \
        f"Expected all 'duplicate', got: {[r['status'] for r in dups]}"
    assert len(tmp_store.read_all()) == 3


# ── Test 3: chain validates after all writes ───────────────────────────────────

def test_verify_chain_valid(tmp_store):
    _log_all(tmp_store)

    assert verify_chain(tmp_store.read_all()) is True


# ── Test 4: field mutation breaks chain ───────────────────────────────────────

def test_verify_chain_detects_mutation(tmp_store):
    _log_all(tmp_store)

    rows = tmp_store.read_all()
    # Mutate a frozen field in the second row (silver); chain from row 1 onward breaks
    rows[1]["verdict"] = "TAMPERED"

    assert verify_chain(rows) is False


# ── Test 5: failing append → error result, no exception propagated ─────────────

def test_failing_store_returns_error_no_raise():
    class BrokenStore:
        def read_all(self) -> list:
            return []

        def append(self, row: dict) -> None:
            raise RuntimeError("disk full")

    result = log_prediction(store=BrokenStore(), **_kwargs("gold"))

    assert result["status"] == "error"
    assert result["record_hash"] is None
    # Reaching here means no exception propagated to the caller


# ── Test 6: bad proba_vector length → error, no write ─────────────────────────

def test_bad_proba_vector_length(tmp_store):
    bad = {**_kwargs("gold"), "proba_vector": [0.5, 0.5]}  # length 2
    result = log_prediction(store=tmp_store, **bad)

    assert result["status"] == "error"
    assert result["record_hash"] is None
    assert len(tmp_store.read_all()) == 0


# ── Test 7: raw_signal outside {0,1,2} → error, no write ──────────────────────

def test_bad_raw_signal(tmp_store):
    bad = {**_kwargs("gold"), "raw_signal": 5}
    result = log_prediction(store=tmp_store, **bad)

    assert result["status"] == "error"
    assert result["record_hash"] is None
    assert len(tmp_store.read_all()) == 0


# ── Test 8: mixed-case metal normalized; dedup works on normalized key ─────────

def test_mixed_case_metal_normalized(tmp_store):
    # Log "Gold" (capitalized) — should succeed and normalize to "gold"
    r1 = log_prediction(store=tmp_store, **{**_kwargs("gold"), "metal": "Gold"})
    assert r1["status"] == "appended"

    # Same as_of_date + model_version, now lowercase — must be a duplicate
    r2 = log_prediction(store=tmp_store, **_kwargs("gold"))
    assert r2["status"] == "duplicate"

    rows = tmp_store.read_all()
    assert len(rows) == 1
    assert rows[0]["metal"] == "gold"


# ── Test 9: unknown metal → error, no write ────────────────────────────────────

def test_unknown_metal_rejected(tmp_store):
    bad = {**_kwargs("gold"), "metal": "copper"}
    result = log_prediction(store=tmp_store, **bad)

    assert result["status"] == "error"
    assert result["record_hash"] is None
    assert len(tmp_store.read_all()) == 0
