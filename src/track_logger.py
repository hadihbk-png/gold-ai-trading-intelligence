"""
Track-record logger — capture-only core.

Captures ML predictions for Gold / Silver / Platinum before outcomes are known.
Each record is SHA-256 chain-linked so any post-hoc mutation is detectable.
No scoring, no Google Sheets, no UI — those are later steps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

# ── CONFIG ─────────────────────────────────────────────────────────────────────

HORIZON = "next_day_close_to_close_direction"  # confirmed against training target
DIRECTION_THRESHOLD = 0.005    # ±0.5% band; matches src/config.py; used by the FUTURE
                               # scoring pass — recorded here so it can't drift.
                               # Do NOT use the legacy 0.003 (Target_Signal).
FLOAT_PRECISION = 6            # decimal places for all float fields in canonical JSON
GENESIS_HASH = "0" * 64       # chain-head sentinel used as prev_hash for the first record
# Resolution series: next trading day's df["Close"] from the SAME source
# recorded in data_provenance.  No hardcoded exchange.

# Internal constants (not part of the public API)
_NOMINAL_SETTLE_HOUR = "17:00:00"  # nominal UTC time; scoring pass refines with exchange calendar
_CLASS_MAP = {0: "DOWN", 1: "SIDEWAYS", 2: "UP"}  # proba_vector = [P(DOWN), P(SIDEWAYS), P(UP)]

_log = logging.getLogger(__name__)


class StoreIntegrityError(RuntimeError):
    """Raised when a SheetsStore worksheet has a non-empty but malformed header row."""


# ── STORAGE PROTOCOL ───────────────────────────────────────────────────────────

@runtime_checkable
class Store(Protocol):
    def read_all(self) -> list[dict]:
        """Return all rows in append order; never reorder."""
        ...

    def append(self, row: dict) -> None:
        """Append one row to the end; never reorder or sort."""
        ...


class LocalJsonStore:
    """
    JSONL-backed store (one JSON object per line).

    Creates the file and any missing parent directories on construction.
    Intended for local testing only.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        if not os.path.exists(path):
            open(path, "w", encoding="utf-8").close()  # noqa: WPS515

    def read_all(self) -> list[dict]:
        rows: list[dict] = []
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def append(self, row: dict) -> None:
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── SHEETS STORE ───────────────────────────────────────────────────────────────

# Columns written as plain text for human readability in the spreadsheet.
HELPER_COLUMNS = [
    "prediction_id",
    "as_of_date",
    "metal",
    "verdict",
    "confidence_pct",
    "record_hash",
]
# Full header: helper columns followed by the canonical payload column.
SHEET_HEADER = HELPER_COLUMNS + ["payload_json"]


class SheetsStore:
    """
    Google Sheets-backed store satisfying the Store protocol.

    The worksheet object is injected — no gspread import here.
    A credentials factory (make_sheets_store_from_secrets) will be added at
    the wiring step and is not part of this module.
    """

    def __init__(self, worksheet) -> None:
        self._ws = worksheet

    def _ensure_header(self) -> None:
        # A freshly created gspread worksheet returns a grid of empty strings
        # rather than [], so check that the first row actually contains the
        # header sentinel instead of just testing for truthiness.
        vals = self._ws.get_all_values()
        if not vals or vals[0][0:1] != ["prediction_id"]:
            self._ws.append_row(SHEET_HEADER, value_input_option="RAW")

    def append(self, row: dict) -> None:
        self._ensure_header()
        payload_json = json.dumps(row, ensure_ascii=False)
        full_row = [str(row.get(col, "")) for col in HELPER_COLUMNS] + [payload_json]
        self._ws.append_row(full_row, value_input_option="RAW")

    def read_all(self) -> list[dict]:
        vals = self._ws.get_all_values()
        # Treat the sheet as empty if every cell in every row is an empty string
        # (a freshly created gspread worksheet returns such a grid rather than []).
        if not any(cell != "" for row in vals for cell in row):
            return []
        header = vals[0]
        try:
            pj_idx = header.index("payload_json")
        except ValueError:
            raise StoreIntegrityError(
                f"Sheet header row lacks 'payload_json' column; first row: {header!r}"
            )
        rows: list[dict] = []
        for raw_row in vals[1:]:
            if pj_idx < len(raw_row) and raw_row[pj_idx]:
                rows.append(json.loads(raw_row[pj_idx]))
        return rows


# ── CANONICALIZATION ───────────────────────────────────────────────────────────

def _canonicalize(value):  # type: ignore[return]
    """
    Recursively normalize a value for deterministic JSON serialization.

    Rules:
    - float  → rounded to FLOAT_PRECISION decimal places
    - datetime → UTC "YYYY-MM-DDTHH:MM:SSZ" (naive assumed UTC)
    - date   → "YYYY-MM-DD"
    - list   → each element canonicalized
    - dict   → each value canonicalized
    - everything else (str, int, None, bool) → returned unchanged
    """
    if isinstance(value, bool):      # bool is a subclass of int; check before int/float
        return value
    if isinstance(value, float):
        return round(value, FLOAT_PRECISION)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        utc = value.astimezone(timezone.utc)
        return utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, date):      # after datetime — datetime is a subclass of date
        return value.strftime("%Y-%m-%d")
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _canonicalize(v) for k, v in value.items()}
    return value


# Ordered list of prediction fields included in the hash payload.
# Outcome fields (actual_price, realized_direction, hit,
# realized_return_net_of_cost, scored_at) and record_hash are EXCLUDED.
_HASH_FIELDS = (
    "prediction_id",
    "timestamp_utc",
    "as_of_date",
    "metal",
    "horizon",
    "horizon_resolves_at",
    "raw_signal",
    "displayed_signal",
    "verdict",
    "confidence_pct",
    "proba_vector",
    "price_at_decision",
    "regime",
    "atr_pct",
    "filter_state",
    "model_version",
    "data_provenance",
)


def _build_hash_payload(record: dict) -> dict:
    """Extract and canonicalize the frozen prediction fields used for hashing."""
    return {k: _canonicalize(record[k]) for k in _HASH_FIELDS}


def _compute_hash(prev_hash: str, payload: dict) -> str:
    canonical_json = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(
        (prev_hash + "|" + canonical_json).encode("utf-8")
    ).hexdigest()


# ── DERIVATIONS ────────────────────────────────────────────────────────────────

def _next_weekday(d: date) -> date:
    """Return the next Mon–Fri calendar day after d.

    Holiday-calendar precision is deferred to the scoring pass (affects only
    when scoring runs, not the hash or dedup key).
    """
    nwd = d + timedelta(days=1)
    while nwd.weekday() >= 5:  # 5=Sat, 6=Sun
        nwd += timedelta(days=1)
    return nwd


def _derive_horizon_resolves_at(as_of_date: date) -> str:
    """Return an ISO-8601 UTC string for the next weekday after as_of_date.

    Uses a fixed nominal settle time (_NOMINAL_SETTLE_HOUR).  Holiday-calendar
    precision is deferred to the scoring pass — affects only when scoring runs,
    not the hash or dedup key.
    """
    nwd = _next_weekday(as_of_date)
    return f"{nwd.strftime('%Y-%m-%d')}T{_NOMINAL_SETTLE_HOUR}Z"


# ── PUBLIC API ─────────────────────────────────────────────────────────────────

def log_prediction(
    *,
    store: Store,
    metal: str,
    timestamp_utc: datetime,
    as_of_date: date,
    raw_signal: int,
    displayed_signal: str,
    verdict: str,
    proba_vector: list,
    price_at_decision: float,
    atr_pct: float,
    filter_state: str,
    model_version: str,
    data_provenance: str,
    regime: str | None = None,
) -> dict:
    """
    Append a single prediction to *store* with chain-linked SHA-256 integrity.

    store is REQUIRED — there is no default.  Nothing writes until a store is
    explicitly passed by the caller.

    Returns a dict with keys:
      status          "appended" | "duplicate" | "error"
      prediction_id   the computed ID (best-effort on error)
      record_hash     hex SHA-256 (None on error)

    Duplicate detection: if a row with the same prediction_id already exists,
    no write occurs and status="duplicate" is returned immediately.

    Failure isolation: the entire body is wrapped so this function NEVER raises
    to the caller.  Any exception is logged as a warning and status="error" is
    returned.
    """
    best_effort_id: str | None = None
    try:
        if len(proba_vector) != 3:
            raise ValueError(
                f"proba_vector must have exactly 3 elements, got {len(proba_vector)}"
            )
        if raw_signal not in {0, 1, 2}:
            raise ValueError(
                f"raw_signal must be 0, 1, or 2, got {raw_signal!r}"
            )

        # Normalize metal before prediction_id is built; casing typos fragment the dedup key
        metal = metal.strip().lower()
        if metal not in {"gold", "silver", "platinum"}:
            raise ValueError(
                f"Unknown metal {metal!r}; must be one of: gold, silver, platinum"
            )

        # Normalize: datetime → date
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.date()

        as_of_date_str = as_of_date.strftime("%Y-%m-%d")
        # as_of_date = date of the latest completed Close bar (passed in by caller)
        prediction_id = f"{metal}_{as_of_date_str}_{model_version}"
        best_effort_id = prediction_id

        # ── Dedup check ──────────────────────────────────────────────────────
        existing_rows = store.read_all()
        for row in existing_rows:
            if row.get("prediction_id") == prediction_id:
                return {
                    "status":        "duplicate",
                    "prediction_id": prediction_id,
                    "record_hash":   row.get("record_hash"),
                }

        prev_hash = existing_rows[-1]["record_hash"] if existing_rows else GENESIS_HASH

        # confidence_pct derived from proba_vector[raw_signal] — never passed in
        confidence_pct = round(float(proba_vector[raw_signal]) * 100, FLOAT_PRECISION)

        record: dict = {
            # ── Frozen prediction fields (included in hash) ──────────────────
            "prediction_id":       prediction_id,
            "timestamp_utc":       _canonicalize(timestamp_utc),
            "as_of_date":          as_of_date_str,
            "metal":               metal,
            "horizon":             HORIZON,
            "horizon_resolves_at": _derive_horizon_resolves_at(as_of_date),
            "raw_signal":          raw_signal,
            "displayed_signal":    displayed_signal,
            "verdict":             verdict,
            "confidence_pct":      confidence_pct,
            "proba_vector":        [_canonicalize(float(p)) for p in proba_vector],
            "price_at_decision":   _canonicalize(float(price_at_decision)),
            "regime":              regime,
            "atr_pct":             _canonicalize(float(atr_pct)),
            "filter_state":        filter_state,
            "model_version":       model_version,
            "data_provenance":     data_provenance,
            # ── Outcome fields — null until the scoring pass populates them ──
            "actual_price":                None,
            "realized_direction":          None,
            "hit":                         None,
            "realized_return_net_of_cost": None,
            "scored_at":                   None,
        }

        payload = _build_hash_payload(record)
        record_hash = _compute_hash(prev_hash, payload)
        record["record_hash"] = record_hash

        store.append(record)
        return {
            "status":        "appended",
            "prediction_id": prediction_id,
            "record_hash":   record_hash,
        }

    except Exception as exc:
        _log.warning("log_prediction failed: %s", exc, exc_info=True)
        return {
            "status":        "error",
            "prediction_id": best_effort_id,
            "record_hash":   None,
        }


def verify_chain(rows: list[dict]) -> bool:
    """
    Verify the SHA-256 chain of a list of prediction rows.

    Recomputes each record_hash from stored fields + the prior hash using the
    same canonicalization as log_prediction.  Returns True iff every hash in
    the sequence is valid.  An empty list is considered valid (returns True).
    """
    prev_hash = GENESIS_HASH
    for row in rows:
        try:
            payload = _build_hash_payload(row)
            expected = _compute_hash(prev_hash, payload)
            if row.get("record_hash") != expected:
                return False
            prev_hash = expected
        except Exception:
            return False
    return True
