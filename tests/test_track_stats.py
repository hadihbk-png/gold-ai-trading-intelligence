"""
Unit tests for the pure statistical helpers in src/track_stats.py
and the aggregation logic replicated verbatim from pages/6_Track_Record.py.

These tests are purely arithmetic — no Streamlit, no network, no store I/O.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.track_stats import wilson_ci


# ── Wilson CI ─────────────────────────────────────────────────────────────────

def test_wilson_ci_canonical_case():
    """15/20 — the authoritative reference case from the user spec (2026-06-22).

    Correct formula shifts the center below p̂ via z²/2n correction.
    If center ≈ 0.75 (== p̂) the old naive formula is still in use; must be ≈ 0.710.
    """
    lo, hi = wilson_ci(15, 20)
    assert abs(lo - 0.5313) < 0.001, f"lower bound {lo:.4f}, expected ~0.531"
    assert abs(hi - 0.8881) < 0.001, f"upper bound {hi:.4f}, expected ~0.888"
    # Center must differ from p̂ = 0.75
    center = (lo + hi) / 2
    assert abs(center - 0.75) > 0.01, (
        f"center={center:.4f} equals p̂=0.75 — old (naive) formula still in use"
    )
    assert abs(center - 0.7097) < 0.002, f"center={center:.4f}, expected ~0.710"


def test_wilson_ci_zero_n():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_all_hits():
    """n=10, hits=10 — upper bound must be 1.0, lower bound > 0.7."""
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0
    assert lo > 0.7


def test_wilson_ci_no_hits():
    """n=10, hits=0 — lower bound must be 0.0, upper bound < 0.3."""
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert hi < 0.3


def test_wilson_ci_center_formula():
    """Verify center = (p̂ + z²/2n)/denom, not p̂, for n=5, hits=3."""
    hits, n, z = 3, 5, 1.96
    lo, hi = wilson_ci(hits, n, z)
    p      = hits / n
    denom  = 1.0 + z ** 2 / n
    expected_center = (p + z ** 2 / (2.0 * n)) / denom
    computed_center = (lo + hi) / 2
    assert abs(computed_center - expected_center) < 1e-6


# ── Hit-rate (pandas logic from the page) ─────────────────────────────────────

def _make_scored_df(hits: list[bool]) -> pd.DataFrame:
    """Minimal DataFrame matching the columns the page uses for hit-rate."""
    return pd.DataFrame({
        "metal": ["gold"] * len(hits),
        "hit":   pd.array(hits, dtype="boolean"),
    })


def test_hit_rate_all_correct():
    df = _make_scored_df([True, True, True])
    assert int(df["hit"].sum()) == 3
    assert int(df["hit"].sum()) / len(df) == 1.0


def test_hit_rate_mixed():
    df = _make_scored_df([True, False, True, False, False])
    n_hits = int(df["hit"].sum())
    assert n_hits == 2
    assert abs(n_hits / len(df) - 0.4) < 1e-9


def test_hit_rate_zero():
    df = _make_scored_df([False, False])
    assert int(df["hit"].sum()) == 0


# ── Equity curve compounding ───────────────────────────────────────────────────

def test_equity_compounding_basic():
    """np.cumprod mirrors the page's equity curve logic."""
    rets   = np.array([0.01, -0.005, 0.0, 0.02])
    equity = np.cumprod(1.0 + rets)
    assert abs(equity[0] - 1.01) < 1e-10
    assert abs(equity[1] - 1.01 * 0.995) < 1e-10
    assert abs(equity[2] - 1.01 * 0.995 * 1.0) < 1e-10
    assert abs(equity[-1] - (1.01 * 0.995 * 1.0 * 1.02)) < 1e-10


def test_equity_sideways_flat():
    """SIDEWAYS returns (0.0) leave the equity curve unchanged."""
    rets   = np.array([0.0, 0.0, 0.0])
    equity = np.cumprod(1.0 + rets)
    assert np.allclose(equity, [1.0, 1.0, 1.0])


def test_equity_end_value_matches_product():
    rets   = np.array([-0.00941035, -0.00628973, 0.0, -0.00237233, -0.03148881])
    equity = np.cumprod(1.0 + rets)
    manual = math.prod(1.0 + r for r in rets)
    assert abs(equity[-1] - manual) < 1e-10


# ── Confusion matrix cell ─────────────────────────────────────────────────────

def _confusion_cell(df: pd.DataFrame, pred: int, actual: int) -> int:
    return int(((df["raw_signal"] == pred) & (df["realized_direction"] == actual)).sum())


def test_confusion_matrix_cell():
    df = pd.DataFrame({
        "raw_signal":         [2, 0, 1, 2, 0],
        "realized_direction": [0, 1, 1, 0, 2],
    })
    # pred=UP(2), actual=DOWN(0): rows 0 and 3 → 2
    assert _confusion_cell(df, pred=2, actual=0) == 2
    # pred=DOWN(0), actual=SIDEWAYS(1): row 1 → 1
    assert _confusion_cell(df, pred=0, actual=1) == 1
    # pred=SIDEWAYS(1), actual=SIDEWAYS(1): row 2 → 1
    assert _confusion_cell(df, pred=1, actual=1) == 1
    # pred=DOWN(0), actual=UP(2): row 4 → 1
    assert _confusion_cell(df, pred=0, actual=2) == 1
    # no pred=UP, actual=UP
    assert _confusion_cell(df, pred=2, actual=2) == 0


def test_confusion_matrix_totals():
    """Row sums equal number of predictions per class; col sums equal realized counts."""
    df = pd.DataFrame({
        "raw_signal":         [2, 0, 1, 2, 0],
        "realized_direction": [0, 1, 1, 0, 2],
    })
    # pred=UP appears twice (rows 0,3)
    up_row_sum = sum(_confusion_cell(df, pred=2, actual=c) for c in [0, 1, 2])
    assert up_row_sum == 2
    # realized=SIDEWAYS(1) appears twice (rows 1,2)
    sw_col_sum = sum(_confusion_cell(df, pred=c, actual=1) for c in [0, 1, 2])
    assert sw_col_sum == 2
