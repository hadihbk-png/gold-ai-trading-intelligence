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

from src.track_logger import (
    LocalJsonStore,
    SheetsStore,
    StoreIntegrityError,
    SHEET_HEADER,
    log_prediction,
    verify_chain,
)

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


# ── FakeWorksheet — pure-Python Sheets simulation ─────────────────────────────

class FakeWorksheet:
    """
    Simulates a gspread Worksheet with Sheets-style string coercion.
    Every value passed to append_row is stored as str(), matching real Sheets
    behaviour where every cell value is a string regardless of type.
    """

    def __init__(self):
        self._rows = []

    def append_row(self, values, value_input_option=None):
        self._rows.append([str(v) for v in values])

    def get_all_values(self):
        return [list(r) for r in self._rows]


# ── SheetsStore tests (10-14) ─────────────────────────────────────────────────

def test_sheets_three_metals_chain():
    """3 metals → 3 dicts from read_all(); verify_chain validates through Sheets round-trip."""
    store = SheetsStore(FakeWorksheet())
    results = [log_prediction(store=store, **_kwargs(m)) for m in METALS]

    assert all(r["status"] == "appended" for r in results)
    rows = store.read_all()
    assert len(rows) == 3
    assert verify_chain(rows) is True


def test_sheets_nested_fields_round_trip():
    """proba_vector (list) and data_provenance (str) survive the Sheets string round-trip."""
    store = SheetsStore(FakeWorksheet())
    log_prediction(store=store, **_kwargs("gold"))

    row = store.read_all()[0]
    assert isinstance(row["proba_vector"], list)
    assert row["proba_vector"] == [0.10, 0.20, 0.70]
    assert row["data_provenance"] == "yfinance/GLD"


def test_sheets_messy_float_round_trip():
    """price_at_decision with excess decimals is stored rounded to FLOAT_PRECISION; chain holds."""
    store = SheetsStore(FakeWorksheet())
    log_prediction(store=store, **{**_kwargs("gold"), "price_at_decision": 2050.123456789})

    rows = store.read_all()
    assert rows[0]["price_at_decision"] == round(2050.123456789, 6)
    assert verify_chain(rows) is True


def test_sheets_dedup():
    """Re-logging gold after all 3 metals → duplicate, read_all stays length 3."""
    store = SheetsStore(FakeWorksheet())
    for m in METALS:
        log_prediction(store=store, **_kwargs(m))

    r = log_prediction(store=store, **_kwargs("gold"))
    assert r["status"] == "duplicate"
    assert len(store.read_all()) == 3


def test_sheets_header():
    """After the first append the first worksheet row equals SHEET_HEADER."""
    ws = FakeWorksheet()
    store = SheetsStore(ws)
    log_prediction(store=store, **_kwargs("gold"))

    assert ws.get_all_values()[0] == SHEET_HEADER


# ── SheetsStore edge-case tests (15-17) ───────────────────────────────────────

def test_ensure_header_on_empty_string_grid():
    """_ensure_header writes SHEET_HEADER when get_all_values() returns a grid of empty strings
    (the state a freshly created gspread worksheet is in before any data is written)."""
    ws = FakeWorksheet()
    # Simulate a real fresh worksheet: multiple rows, all empty strings
    ws._rows = [[""] * len(SHEET_HEADER), [""] * len(SHEET_HEADER)]
    store = SheetsStore(ws)

    store._ensure_header()

    # The header must have been appended as the next row
    assert ws.get_all_values()[-1] == SHEET_HEADER


def test_read_all_empty_on_empty_string_grid():
    """read_all returns [] without raising when the sheet contains only empty strings."""
    ws = FakeWorksheet()
    ws._rows = [[""] * len(SHEET_HEADER), [""] * len(SHEET_HEADER)]
    store = SheetsStore(ws)

    assert store.read_all() == []


def test_read_all_raises_on_malformed_header():
    """read_all raises StoreIntegrityError when a real header row is present but lacks
    'payload_json' (e.g. a manually edited sheet or wrong tab)."""
    ws = FakeWorksheet()
    # Non-empty header that contains no payload_json column, plus a data row
    ws._rows = [
        ["col_a", "col_b", "col_c"],
        ["val1",  "val2",  "val3"],
    ]
    store = SheetsStore(ws)

    with pytest.raises(StoreIntegrityError, match="payload_json"):
        store.read_all()
