"""
Macro regime gate.

Identifies hostile macro environments where model signals are historically
unreliable and raises the confidence floor required to enter a trade.

Three gate conditions (applied in priority order):

  1. Panic / extreme-volatility regime
       VIX > GATE_VIX_PANIC  AND  ATR_Regime > GATE_ATR_REGIME_MAX
       → require P(signal) > GATE_PANIC_CONF for any trade direction.
       Rationale: crisis spikes produce intraday noise that makes next-day
       direction nearly unpredictable even when the model is confident.

  2. Rate-shock regime (suppresses BUY / UP only)
       TNX 20-bar change > GATE_TNX_SHOCK_PP pp  AND  DXY above its SMA20
       → require P(UP) > GATE_SHOCK_UP_CONF to enter a long gold trade.
       Rationale: rapidly rising real yields + strong dollar are the
       classic gold headwind (2022 is the canonical example).
       DOWN signals are NOT suppressed — shorting during rate shock is valid.

  3. Dollar-surge regime (suppresses BUY / UP only)
       DXY > GATE_DXY_BULL_PCT above SMA20  AND  DXY slope positive
       → require P(UP) > GATE_SHOCK_UP_CONF to enter a long gold trade.
       Rationale: gold-dollar inverse relationship means a surging dollar
       makes gold longs require higher conviction.

Each condition raises the confidence threshold rather than blocking outright,
so the model can still trade when its conviction is genuinely high.
"""

import numpy as np
from src.config import (
    REGIME_GATE_ENABLED,
    GATE_TNX_SHOCK_PP, GATE_DXY_BULL_PCT, GATE_SHOCK_UP_CONF,
    GATE_VIX_PANIC, GATE_ATR_REGIME_MAX, GATE_PANIC_CONF,
    ADX_RANGE_FILTER_ENABLED, ADX_RANGE_THRESHOLD, ADX_RANGE_CONF_FLOOR,
)


def check_regime_gate(
    signal_int: int,
    proba_row: "np.ndarray | None",
    feature_snapshot: dict,
) -> "tuple[bool, str | None]":
    """
    Evaluate macro regime gate for a proposed trade entry.

    Parameters
    ----------
    signal_int       : 0=DOWN, 1=SIDEWAYS, 2=UP
    proba_row        : model probability array [p_down, p_side, p_up]
    feature_snapshot : dict of current bar feature values; missing keys are
                       treated as NaN and the corresponding gate is skipped.

    Returns
    -------
    (allow_trade, reason_string_or_None)
    """
    if not REGIME_GATE_ENABLED or signal_int == 1:
        return True, None

    p_signal = float(proba_row[signal_int]) if proba_row is not None else 1.0

    def _get(key: str) -> float:
        v = feature_snapshot.get(key, float("nan"))
        if v is None or (isinstance(v, float) and (v != v or np.isinf(v))):
            return float("nan")
        return float(v)

    vix        = _get("VIX_Close")
    atr_regime = _get("ATR_Regime")
    tnx_slope  = _get("TNX_Slope20")
    dxy_vs_ma  = _get("DXY_vs_SMA20")
    dxy_slope  = _get("DXY_Slope10")

    # ── Gate 1: Panic / extreme volatility ────────────────────────────────────
    panic = (
        not np.isnan(vix)        and vix        > GATE_VIX_PANIC
        and not np.isnan(atr_regime) and atr_regime > GATE_ATR_REGIME_MAX
    )
    if panic and p_signal < GATE_PANIC_CONF:
        return (
            False,
            f"panic-regime gate (VIX={vix:.1f}, ATR×={atr_regime:.2f},"
            f" need conf≥{GATE_PANIC_CONF:.2f}, got {p_signal:.3f})",
        )

    # ── Gate 2: Rate-shock (BUY signals only) ─────────────────────────────────
    if signal_int == 2:
        rate_shock = (
            not np.isnan(tnx_slope) and tnx_slope > GATE_TNX_SHOCK_PP
            and not np.isnan(dxy_vs_ma) and dxy_vs_ma > 0
        )
        if rate_shock and p_signal < GATE_SHOCK_UP_CONF:
            return (
                False,
                f"rate-shock gate (TNX Δ20={tnx_slope:+.2f}pp,"
                f" DXY vs SMA={dxy_vs_ma:+.2%},"
                f" need conf≥{GATE_SHOCK_UP_CONF:.2f}, got {p_signal:.3f})",
            )

    # ── Gate 3: Dollar surge (BUY signals only) ───────────────────────────────
    if signal_int == 2:
        dollar_surge = (
            not np.isnan(dxy_vs_ma) and dxy_vs_ma > GATE_DXY_BULL_PCT
            and not np.isnan(dxy_slope) and dxy_slope > 0
        )
        if dollar_surge and p_signal < GATE_SHOCK_UP_CONF:
            return (
                False,
                f"dollar-surge gate (DXY vs SMA={dxy_vs_ma:+.2%},"
                f" need conf≥{GATE_SHOCK_UP_CONF:.2f}, got {p_signal:.3f})",
            )

    # ── Gate 4: Range / choppy-market filter (disabled — non-determinism
    #    in LightGBM/CatBoost threading makes filter effect unverifiable) ─────
    if False and ADX_RANGE_FILTER_ENABLED and signal_int == 2:
        adx = _get("ADX")
        if not np.isnan(adx) and adx < ADX_RANGE_THRESHOLD:
            if p_signal < ADX_RANGE_CONF_FLOOR:
                return (
                    False,
                    f"range-filter gate (ADX={adx:.1f} < {ADX_RANGE_THRESHOLD:.0f},"
                    f" need conf≥{ADX_RANGE_CONF_FLOOR:.2f}, got {p_signal:.3f})",
                )

    return True, None
