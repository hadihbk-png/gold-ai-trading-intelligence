"""
Quant-grade risk management layer.

RiskManager  – stateful per-simulation object that tracks equity, drawdown,
               daily P&L and enforces hard stop rules.
position_size – ATR-based Kelly-like sizing with confidence scaling.
check_no_trade – explicit filter rules applied before any entry.
"""

import numpy as np
from src.config import (
    MAX_DAILY_LOSS_PCT, MAX_DRAWDOWN_HALT, DRAWDOWN_RESUME_PCT,
    MAX_OPEN_EXPOSURE, RISK_PER_TRADE_PCT,
    ATR_STOP_MULTIPLIER, MIN_CONFIDENCE, MAX_ATR_PCT, MAX_SPREAD_PCT,
    CONF_SIZE_MIN, CONF_SIZE_MAX,
)


class RiskManager:
    """
    Stateful risk manager.  Instantiate once per backtest run and call
    update_equity / begin_day / check_can_trade on every bar.
    """

    def __init__(self, initial_capital: float, params: dict | None = None):
        p = params or {}
        self.capital          = initial_capital
        self.peak_equity      = initial_capital
        self.daily_start_eq   = initial_capital
        self.open_exposure    = 0.0          # fraction of capital currently at risk
        self._halted          = False
        self._halt_reason     = ""

        self.max_daily_loss   = p.get("max_daily_loss_pct",  MAX_DAILY_LOSS_PCT)
        self.max_dd_halt      = p.get("max_drawdown_halt",   MAX_DRAWDOWN_HALT)
        self.dd_resume        = p.get("drawdown_resume_pct", DRAWDOWN_RESUME_PCT)
        self.max_exposure     = p.get("max_open_exposure",   MAX_OPEN_EXPOSURE)
        self.risk_per_trade   = p.get("risk_per_trade_pct",  RISK_PER_TRADE_PCT)

    # ── State updates ─────────────────────────────────────────────────────────

    def update_equity(self, new_equity: float) -> None:
        self.capital     = max(new_equity, 0)
        self.peak_equity = max(self.peak_equity, self.capital)

    def begin_day(self) -> None:
        self.daily_start_eq = self.capital

    def add_exposure(self, fraction: float) -> None:
        self.open_exposure = min(self.open_exposure + fraction, 1.0)

    def remove_exposure(self, fraction: float) -> None:
        self.open_exposure = max(self.open_exposure - fraction, 0.0)

    # ── Derived state ─────────────────────────────────────────────────────────

    @property
    def current_drawdown(self) -> float:
        if self.peak_equity == 0:
            return 0.0
        return (self.capital - self.peak_equity) / self.peak_equity

    @property
    def daily_pnl_pct(self) -> float:
        if self.daily_start_eq == 0:
            return 0.0
        return (self.capital - self.daily_start_eq) / self.daily_start_eq

    # ── Gate: can we trade? ───────────────────────────────────────────────────

    def check_can_trade(self) -> tuple[bool, str]:
        """Returns (allowed, reason_if_blocked)."""

        # Auto-resume check if we were halted
        if self._halted:
            if self.current_drawdown > -self.dd_resume:
                self._halted = False
                self._halt_reason = ""
            else:
                return False, self._halt_reason

        dd = self.current_drawdown
        if dd < -self.max_dd_halt:
            self._halted = True
            self._halt_reason = (
                f"Drawdown {dd:.1%} exceeds max {self.max_dd_halt:.1%}"
            )
            return False, self._halt_reason

        if self.daily_pnl_pct < -self.max_daily_loss:
            return False, (
                f"Daily loss {self.daily_pnl_pct:.1%} exceeds max {self.max_daily_loss:.1%}"
            )

        if self.open_exposure >= self.max_exposure:
            return False, f"Max exposure {self.max_exposure:.1%} already deployed"

        return True, ""

    # ── Position sizing ───────────────────────────────────────────────────────

    def position_size(
        self,
        price: float,
        atr_val: float,
        confidence: float | None = None,
        regime_factor: float = 1.0,
    ) -> float:
        """
        ATR-based position size in units.

        Risk amount  = capital × risk_per_trade
        Stop distance = ATR_STOP_MULTIPLIER × ATR
        N units      = risk_amount / stop_distance

        Adjustments:
          · confidence scaling  (low confidence → smaller size)
          · regime factor       (high-vol regime → smaller size)
          · max-exposure cap    (never deploy more than allowed fraction)
        """
        stop_dist = ATR_STOP_MULTIPLIER * atr_val
        if stop_dist <= 0 or price <= 0:
            return 0.0

        n_units = (self.capital * self.risk_per_trade) / stop_dist

        # Confidence scaling: maps [MIN_CONFIDENCE, 1.0] → [CONF_SIZE_MIN, CONF_SIZE_MAX]
        # Default 0.25→0.5x, ~0.62→1.0x, 1.0→1.5x — oversize high-conviction trades.
        if confidence is not None:
            conf_scale = CONF_SIZE_MIN + (CONF_SIZE_MAX - CONF_SIZE_MIN) * min(max(
                (confidence - MIN_CONFIDENCE) / (1.0 - MIN_CONFIDENCE), 0
            ), 1.0)
            n_units *= conf_scale

        n_units *= regime_factor

        # Hard cap by exposure budget
        remaining = max(self.max_exposure - self.open_exposure, 0)
        cap_units = (self.capital * remaining) / price
        return float(min(n_units, cap_units))


# ── Stateless no-trade filter (used by signals.py and backtest.py) ────────────

def check_no_trade(
    signal_int: int,
    confidence: float | None,
    atr_val: float,
    price: float,
    current_drawdown: float,
    spread_pct: float | None = None,
) -> tuple[bool, str]:
    """
    Returns (trade_ok, reason).  All thresholds are from config.
    """
    if signal_int == 1:
        return False, "Signal is No Trade"

    if confidence is not None and confidence < MIN_CONFIDENCE:
        return False, (
            f"Confidence {confidence:.1%} below threshold {MIN_CONFIDENCE:.1%}"
        )

    atr_pct = atr_val / price if price > 0 else 0.0
    if atr_pct > MAX_ATR_PCT:
        return False, (
            f"Volatility too high: ATR/price={atr_pct:.2%} > {MAX_ATR_PCT:.2%}"
        )

    if spread_pct is not None and spread_pct > MAX_SPREAD_PCT:
        return False, (
            f"Estimated spread {spread_pct:.4%} > threshold {MAX_SPREAD_PCT:.4%}"
        )

    if current_drawdown < -(MAX_DRAWDOWN_HALT * 0.8):
        return False, (
            f"Near drawdown limit ({current_drawdown:.1%})"
        )

    return True, ""
