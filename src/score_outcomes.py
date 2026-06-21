"""
Outcome scoring engine for APEX Metals AI track record.

For each logged prediction whose forecast horizon has resolved, fetches the
realized settled close, computes the outcome, and backfills the five outcome
fields via store.update_outcome.

Chain safety: outcome fields are excluded from _HASH_FIELDS, so writing them
does not change any record_hash and verify_chain remains True.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from src.data_loader import download_data, download_metal_ohlcv
from src.track_logger import DIRECTION_THRESHOLD, Store, verify_chain

_log = logging.getLogger(__name__)

# Round-trip transaction cost for the paper trail.
# Confirmed 2026-06-22.  Intentionally lower than the backtest's ~28 bps
# (src/config.py TRANSACTION_COST + EXCHANGE_FEE + BID_ASK_SPREAD + SLIPPAGE × 2 sides),
# which includes slippage and conservative fill assumptions not relevant to
# a next-day close-to-close direction record.
ROUND_TRIP_COST_BPS: int = 5

_METAL_TICKER: dict[str, str] = {
    "silver":   "SI=F",
    "platinum": "PL=F",
}


def _fetch_ohlcv(metal: str) -> pd.DataFrame:
    """Fetch settled OHLCV for *metal* from the SAME source used at decision time.

    Gold  → download_data()                 (GC=F, disk-cached; matches app.py Step 5A)
    Silver/Platinum → download_metal_ohlcv  (SI=F / PL=F; matches _load_metal_data in app.py)
    """
    if metal == "gold":
        return download_data(force_refresh=False)
    return download_metal_ohlcv(_METAL_TICKER[metal], years=2)


def _realized_direction(price_at_decision: float, actual_price: float) -> int:
    """Apply the exact training label rule (DIRECTION_THRESHOLD = ±0.5%).

    Mirrors features.py:
        ret = close.pct_change().shift(-1)
        Target_Direction = np.select([ret < -0.005, ret > 0.005], [0, 2], default=1)
    """
    ret = actual_price / price_at_decision - 1
    if ret > DIRECTION_THRESHOLD:
        return 2  # UP
    if ret < -DIRECTION_THRESHOLD:
        return 0  # DOWN
    return 1      # SIDEWAYS


def _return_net_of_cost(
    raw_signal: int, price_at_decision: float, actual_price: float
) -> float:
    """Directional close-to-close return for raw_signal, minus ROUND_TRIP_COST_BPS.

    raw_signal == 1 (SIDEWAYS) → 0.0 (no trade, no cost incurred).
    raw_signal == 2 (UP)       → long return − cost.
    raw_signal == 0 (DOWN)     → short return − cost.
    """
    if raw_signal == 1:
        return 0.0
    cost = ROUND_TRIP_COST_BPS / 10_000
    if raw_signal == 2:
        return actual_price / price_at_decision - 1 - cost
    return price_at_decision / actual_price - 1 - cost   # raw_signal == 0


def score_all_pending(store: Store) -> dict:
    """Score all resolvable, unscored predictions in *store*.

    Idempotent: rows already scored (scored_at is not None) are skipped.
    Asserts chain integrity after the full pass.

    Returns a summary dict:
      scored              int   — rows successfully scored this run
      skipped_unresolved  int   — horizon has not yet passed
      skipped_no_close    int   — horizon passed but no settled bar found
      already_scored      int   — scored_at was already populated; skipped
      chain_verified      bool  — verify_chain(read_all()) after the pass
    """
    rows = store.read_all()
    now_utc = datetime.now(timezone.utc)

    stats: dict = dict(
        scored=0,
        skipped_unresolved=0,
        skipped_no_close=0,
        already_scored=0,
        chain_verified=False,
    )

    # Lazy per-metal OHLCV cache — one network call per metal per script run
    _ohlcv_cache: dict[str, pd.DataFrame] = {}

    def _get_df(metal: str) -> pd.DataFrame:
        if metal not in _ohlcv_cache:
            _ohlcv_cache[metal] = _fetch_ohlcv(metal)
        return _ohlcv_cache[metal]

    for row in rows:
        # ── Already scored? ───────────────────────────────────────────────────
        if row.get("scored_at") is not None:
            stats["already_scored"] += 1
            continue

        # ── Horizon resolved? ─────────────────────────────────────────────────
        horizon_str = row.get("horizon_resolves_at", "")
        try:
            resolve_dt = datetime.fromisoformat(horizon_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            _log.warning("Unparseable horizon_resolves_at %r — skipping", horizon_str)
            stats["skipped_unresolved"] += 1
            continue

        if now_utc < resolve_dt:
            stats["skipped_unresolved"] += 1
            continue

        resolve_date = resolve_dt.date()
        resolve_ts   = pd.Timestamp(resolve_date)

        # ── Settled close available? ──────────────────────────────────────────
        metal = row["metal"]
        df = _get_df(metal)

        if df.empty or resolve_ts not in df.index:
            stats["skipped_no_close"] += 1
            continue

        actual_price      = float(df.loc[resolve_ts, "Close"])
        price_at_decision = float(row["price_at_decision"])
        raw_signal        = int(row["raw_signal"])

        # ── Compute outcome ───────────────────────────────────────────────────
        rd = _realized_direction(price_at_decision, actual_price)
        outcome = {
            "actual_price":                round(actual_price, 6),
            "realized_direction":          rd,
            "hit":                         rd == raw_signal,
            "realized_return_net_of_cost": round(
                _return_net_of_cost(raw_signal, price_at_decision, actual_price), 8
            ),
            "scored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        store.update_outcome(row["prediction_id"], outcome)
        stats["scored"] += 1

    # ── Post-condition: chain must be intact after all updates ─────────────────
    stats["chain_verified"] = verify_chain(store.read_all())
    return stats
