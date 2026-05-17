"""
Professional walk-forward backtester.

Execution model:
  Fill price = close ± (slippage + spread + commission)
  Overnight financing charged each day position is held.

Risk integration:
  RiskManager gates every entry (drawdown / daily loss / exposure checks).
  No-trade filter applied before entry (confidence / volatility / spread).

Metrics (expanded):
  Sharpe, Sortino, CAGR, Max Drawdown, Win Rate, Profit Factor, Expectancy,
  Calmar Ratio, Avg Hold (days), Total Trades.
"""

import numpy as np
import pandas as pd
from src.config import (
    INITIAL_CAPITAL,
    ATR_STOP_MULTIPLIER, RISK_REWARD_RATIO,
    TOTAL_ENTRY_COST, TOTAL_EXIT_COST, OVERNIGHT_RATE,
    BID_ASK_SPREAD,
    TREND_FILTER_ENABLED, TREND_BULL_DOWN_CONF, TREND_BEAR_UP_CONF,
    TREND_SMA_FAST, TREND_SMA_SLOW, TREND_SLOPE_LOOKBACK,
    BULL_REGIME_ENABLED, BULL_UP_CONF_RELAXED,
    BULL_VIX_CALM_THRESHOLD, BULL_TNX_STABLE_THRESH, BULL_DXY_WEAK_THRESH,
    ATR_TRAIL_ENABLED, ATR_TRAIL_MULTIPLIER,
    CONF_DECAY_ENABLED, CONF_DECAY_THRESHOLD,
    EMA_TREND_EXIT_ENABLED, EMA_TREND_EXIT_PERIOD, EMA_TREND_EXIT_BUFFER,
)
from src.risk import RiskManager, check_no_trade
from src.regime import regime_adjusted_size_factor
from src.regime_gate import check_regime_gate


def _compute_trend_regimes(
    df: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    close: pd.Series,
) -> list | None:
    """
    Pre-compute a per-bar trend regime for the test window.

    Returns a list of integers aligned with test_dates:
      +1  bull  (price > SMA_fast > SMA_slow AND SMA_fast slope rising)
      -1  bear  (price < SMA_fast < SMA_slow AND SMA_fast slope falling)
       0  neutral

    Returns None when the filter is disabled or SMA columns are missing.
    """
    if not TREND_FILTER_ENABLED:
        return None
    fast_col = f"SMA_{TREND_SMA_FAST}"
    slow_col = f"SMA_{TREND_SMA_SLOW}"
    if fast_col not in df.columns or slow_col not in df.columns:
        return None

    df_t   = df.reindex(test_dates)
    sf     = df_t[fast_col]
    ss     = df_t[slow_col]
    sf_lag = sf.shift(TREND_SLOPE_LOOKBACK)

    regimes: list[int] = []
    for i in range(len(test_dates)):
        p  = float(close.iloc[i])
        f  = float(sf.iloc[i])      if not pd.isna(sf.iloc[i])      else float("nan")
        s  = float(ss.iloc[i])      if not pd.isna(ss.iloc[i])      else float("nan")
        fl = float(sf_lag.iloc[i])  if not pd.isna(sf_lag.iloc[i])  else float("nan")

        if any(v != v for v in (f, s, fl)):   # NaN check
            regimes.append(0)
        elif p > f > s and f > fl:
            regimes.append(1)   # bull
        elif p < f < s and f < fl:
            regimes.append(-1)  # bear
        else:
            regimes.append(0)   # neutral
    return regimes


def _compute_bull_regimes(
    df: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    close: pd.Series,
) -> list | None:
    """
    Pre-compute confirmed bull regime for each bar.

    A bar is a confirmed bull if ALL available conditions hold:
      Required: price > SMA50 > SMA200  AND  SMA50 slope > 0
      Macro (skipped when data absent):
        VIX  < BULL_VIX_CALM_THRESHOLD  (calm environment)
        TNX_Slope20 < BULL_TNX_STABLE_THRESH  (yields not surging)
        DXY_vs_SMA20 < BULL_DXY_WEAK_THRESH  OR  DXY_Slope10 < 0  (dollar not surging)

    Returns None when the feature is disabled or SMA columns are missing.
    """
    if not BULL_REGIME_ENABLED:
        return None

    fast_col = f"SMA_{TREND_SMA_FAST}"   # SMA_50
    slow_col = f"SMA_{TREND_SMA_SLOW}"   # SMA_200
    if fast_col not in df.columns or slow_col not in df.columns:
        return None

    df_t = df.reindex(test_dates)
    sf   = df_t[fast_col]
    ss   = df_t[slow_col]

    has_slope  = "SMA50_Slope"  in df_t.columns
    has_vix    = "VIX_Close"    in df_t.columns
    has_tnx    = "TNX_Slope20"  in df_t.columns
    has_dxy_vs = "DXY_vs_SMA20" in df_t.columns
    has_dxy_sl = "DXY_Slope10"  in df_t.columns

    results: list[bool] = []
    for i in range(len(test_dates)):
        p = float(close.iloc[i])
        f = float(sf.iloc[i]) if not pd.isna(sf.iloc[i]) else float("nan")
        s = float(ss.iloc[i]) if not pd.isna(ss.iloc[i]) else float("nan")

        # Core structural condition: price > SMA50 > SMA200
        if f != f or s != s or not (p > f > s):
            results.append(False)
            continue

        # SMA50 slope must be positive
        if has_slope:
            slp = float(df_t["SMA50_Slope"].iloc[i])
            if slp == slp and slp <= 0:
                results.append(False)
                continue

        # VIX must be calm
        if has_vix:
            vix = float(df_t["VIX_Close"].iloc[i])
            if vix == vix and vix >= BULL_VIX_CALM_THRESHOLD:
                results.append(False)
                continue

        # TNX must be stable or falling (not rate-shocked)
        if has_tnx:
            tnx = float(df_t["TNX_Slope20"].iloc[i])
            if tnx == tnx and tnx > BULL_TNX_STABLE_THRESH:
                results.append(False)
                continue

        # DXY must be weakening (below SMA20 OR falling slope)
        if has_dxy_vs or has_dxy_sl:
            dv = float(df_t["DXY_vs_SMA20"].iloc[i]) if has_dxy_vs else float("nan")
            ds = float(df_t["DXY_Slope10"].iloc[i])  if has_dxy_sl else float("nan")
            dv_ok = (dv != dv) or (dv < BULL_DXY_WEAK_THRESH)
            ds_ok = (ds != ds) or (ds < 0)
            if not (dv_ok or ds_ok):
                results.append(False)
                continue

        results.append(True)

    return results


def run_backtest(
    df: pd.DataFrame,
    clf_signals: np.ndarray,
    test_dates: pd.DatetimeIndex,
    initial_capital: float = INITIAL_CAPITAL,
    clf_probas: np.ndarray | None = None,
    regime_series: pd.Series | None = None,
    risk_params: dict | None = None,
):
    """
    Walk-forward simulation over the test period.

    Parameters
    ----------
    df            : Full feature DataFrame (must include ATR, Close columns)
    clf_signals   : Array of 0/1/2 signals aligned with test_dates
    test_dates    : DatetimeIndex for the test period
    clf_probas    : Optional [n_samples × 3] probability matrix for sizing
    regime_series : Optional regime integer series (from regime.py)
    risk_params   : Override dict for RiskManager thresholds

    Returns
    -------
    equity_series, bh_equity, trades_df, metrics_dict
    """
    close = df.reindex(test_dates)["Close"]
    atr   = (df.reindex(test_dates)["ATR"]
             if "ATR" in df.columns else close * 0.01)

    trend_regime_at = _compute_trend_regimes(df, test_dates, close)
    bull_regime_at  = _compute_bull_regimes(df, test_dates, close)

    _ema_col = f"EMA_{EMA_TREND_EXIT_PERIOD}"
    ema_exit = (df.reindex(test_dates)[_ema_col]
                if _ema_col in df.columns
                else pd.Series(np.nan, index=test_dates))

    # Pre-extract macro gate feature snapshots (one dict per bar)
    _GATE_COLS = [
        "VIX_Close", "ATR_Regime", "TNX_Slope20",
        "DXY_vs_SMA20", "DXY_Slope10", "ADX",
    ]
    _gate_df = df.reindex(test_dates)
    _gate_available = {c for c in _GATE_COLS if c in _gate_df.columns}
    gate_snapshots: list[dict] = []
    for _i in range(len(test_dates)):
        gate_snapshots.append(
            {c: float(_gate_df[c].iloc[_i]) for c in _gate_available}
        )

    rm = RiskManager(initial_capital, risk_params)

    equity_curve: list[float] = []
    trades: list[dict] = []

    in_trade       = False
    entry_price    = stop_loss = take_profit = None
    trade_side     = trade_date = None
    n_units        = 0.0
    entry_exposure = 0.0
    days_held      = 0
    trail_stop     = None
    entry_ema      = None

    prev_date = None

    for i, date in enumerate(test_dates):
        price   = float(close.iloc[i])
        atr_val = float(atr.iloc[i]) if not np.isnan(float(atr.iloc[i])) else price * 0.01
        signal  = int(clf_signals[i])

        # ── Daily reset ───────────────────────────────────────────────────────
        if prev_date is None or date.date() != prev_date.date():
            rm.begin_day()
        prev_date = date

        # ── Overnight financing cost ──────────────────────────────────────────
        if in_trade and days_held > 0:
            financing = n_units * price * OVERNIGHT_RATE
            rm.update_equity(rm.capital - financing)
        if in_trade:
            days_held += 1

        # ── Check open-trade exit ─────────────────────────────────────────────
        if in_trade:
            exit_price = None
            exit_reason = None
            last_bar = i == len(test_dates) - 1

            # Ratchet ATR trailing stop toward current price (never retreats)
            if ATR_TRAIL_ENABLED and trail_stop is not None:
                if trade_side == "Buy":
                    trail_stop = max(trail_stop, price - ATR_TRAIL_MULTIPLIER * atr_val)
                else:
                    trail_stop = min(trail_stop, price + ATR_TRAIL_MULTIPLIER * atr_val)

            # Effective stop = best of original stop and trail stop
            if ATR_TRAIL_ENABLED and trail_stop is not None:
                eff_stop = (max(stop_loss, trail_stop) if trade_side == "Buy"
                            else min(stop_loss, trail_stop))
            else:
                eff_stop = stop_loss

            # Priority: Take Profit > Hard Stop (trail or original) > Period End
            if trade_side == "Buy":
                if last_bar:
                    exit_price, exit_reason = price, "Period End"
                elif price >= take_profit:
                    exit_price, exit_reason = take_profit, "Take Profit"
                elif price <= eff_stop:
                    exit_price  = eff_stop
                    exit_reason = ("ATR Trail Stop" if eff_stop > stop_loss
                                   else "Stop Loss")
            else:  # Sell / short
                if last_bar:
                    exit_price, exit_reason = price, "Period End"
                elif price <= take_profit:
                    exit_price, exit_reason = take_profit, "Take Profit"
                elif price >= eff_stop:
                    exit_price  = eff_stop
                    exit_reason = ("ATR Trail Stop" if eff_stop < stop_loss
                                   else "Stop Loss")

            # Confidence decay: exit when model loses conviction in trade direction
            if exit_price is None and days_held >= 1 and CONF_DECAY_ENABLED:
                if clf_probas is not None and i < len(clf_probas):
                    dir_idx = 2 if trade_side == "Buy" else 0
                    if float(clf_probas[i][dir_idx]) < CONF_DECAY_THRESHOLD:
                        exit_price, exit_reason = price, "Conf Decay"

            # EMA trend exit: close when price crosses EMA against trade direction.
            # Only arms when entry was on the correct side of EMA (buffer guards noise).
            if exit_price is None and days_held >= 1 and EMA_TREND_EXIT_ENABLED:
                ema_now = (float(ema_exit.iloc[i])
                           if not pd.isna(ema_exit.iloc[i]) else None)
                if ema_now is not None and entry_ema is not None:
                    if (trade_side == "Buy"
                            and entry_price > entry_ema * (1.0 + EMA_TREND_EXIT_BUFFER)
                            and price < ema_now):
                        exit_price, exit_reason = price, "EMA Exit"
                    elif (trade_side == "Sell"
                            and entry_price < entry_ema * (1.0 - EMA_TREND_EXIT_BUFFER)
                            and price > ema_now):
                        exit_price, exit_reason = price, "EMA Exit"

            if exit_price is not None:
                # Execution cost on exit
                adj_exit = (exit_price * (1 - TOTAL_EXIT_COST) if trade_side == "Buy"
                            else exit_price * (1 + TOTAL_EXIT_COST))
                gross_pnl = (n_units * (adj_exit - entry_price) if trade_side == "Buy"
                             else n_units * (entry_price - adj_exit))
                rm.update_equity(rm.capital + gross_pnl)
                rm.remove_exposure(entry_exposure)

                trades.append({
                    "Entry Date":    trade_date,
                    "Exit Date":     date,
                    "Side":          trade_side,
                    "Entry $":       round(entry_price, 2),
                    "Exit $":        round(adj_exit, 2),
                    "Units":         round(n_units, 4),
                    "PnL $":         round(gross_pnl, 2),
                    "Days Held":     days_held,
                    "Exit Reason":   exit_reason,
                    "Drawdown At Entry": round(rm.current_drawdown * 100, 2),
                })
                in_trade       = False
                n_units        = 0.0
                entry_exposure = 0.0
                days_held      = 0
                trail_stop     = None
                entry_ema      = None

        # ── Entry gate ────────────────────────────────────────────────────────
        if not in_trade:
            # Confirmed bull regime: promote SIDEWAYS/DOWN → BUY when P(UP) ≥ relaxed floor.
            # The OOF threshold (~0.44–0.47) is too tight in strong uptrends; this lowers
            # it to BULL_UP_CONF_RELAXED only when all six bull conditions hold.
            if (BULL_REGIME_ENABLED
                    and bull_regime_at is not None
                    and bull_regime_at[i]
                    and clf_probas is not None and len(clf_probas) > i
                    and signal != 2
                    and float(clf_probas[i][2]) >= BULL_UP_CONF_RELAXED):
                signal = 2

            confidence = None
            if clf_probas is not None and len(clf_probas) > i:
                proba_row = clf_probas[i]
                confidence = float(proba_row[signal]) if len(proba_row) > signal else None

            can_trade, _ = rm.check_can_trade()
            trade_ok, _  = check_no_trade(
                signal_int=signal,
                confidence=confidence,
                atr_val=atr_val,
                price=price,
                current_drawdown=rm.current_drawdown,
                spread_pct=BID_ASK_SPREAD,
            )

            # ── Trend regime filter ───────────────────────────────────────────
            trend_blocked = False
            if (trend_regime_at is not None
                    and clf_probas is not None
                    and len(clf_probas) > i):
                t_reg     = trend_regime_at[i]
                proba_row = clf_probas[i]
                if t_reg == 1 and signal == 0:    # bull regime, DOWN signal
                    trend_blocked = float(proba_row[0]) < TREND_BULL_DOWN_CONF
                elif t_reg == -1 and signal == 2:  # bear regime, UP signal
                    trend_blocked = float(proba_row[2]) < TREND_BEAR_UP_CONF

            # ── Macro regime gate ─────────────────────────────────────────────
            gate_blocked = False
            if not trend_blocked and clf_probas is not None and len(clf_probas) > i:
                _proba = clf_probas[i]
                _snap  = gate_snapshots[i] if i < len(gate_snapshots) else {}
                gate_ok, _ = check_regime_gate(signal, _proba, _snap)
                gate_blocked = not gate_ok

            if can_trade and trade_ok and not trend_blocked and not gate_blocked:
                regime_int = int(regime_series.iloc[i]) if regime_series is not None else 5
                reg_factor = regime_adjusted_size_factor(regime_int)

                n_u = rm.position_size(price, atr_val, confidence, reg_factor)
                if n_u > 0:
                    sl_dist = ATR_STOP_MULTIPLIER * atr_val

                    if signal == 2:  # Buy
                        ep = price * (1 + TOTAL_ENTRY_COST)
                        sl = ep - sl_dist
                        tp = ep + sl_dist * RISK_REWARD_RATIO
                        side = "Buy"
                    else:            # Sell
                        ep = price * (1 - TOTAL_ENTRY_COST)
                        sl = ep + sl_dist
                        tp = ep - sl_dist * RISK_REWARD_RATIO
                        side = "Sell"

                    exp_frac = min((n_u * price) / rm.capital, 1.0)
                    rm.add_exposure(exp_frac)

                    entry_price = ep
                    stop_loss, take_profit = sl, tp
                    trade_side, trade_date = side, date
                    n_units = n_u
                    entry_exposure = exp_frac
                    in_trade = True
                    days_held = 0

                    # Initialise dynamic-exit state
                    trail_stop = (ep - ATR_TRAIL_MULTIPLIER * atr_val if signal == 2
                                  else ep + ATR_TRAIL_MULTIPLIER * atr_val)
                    _ema_at_entry = (float(ema_exit.iloc[i])
                                     if not pd.isna(ema_exit.iloc[i]) else None)
                    entry_ema = _ema_at_entry

        equity_curve.append(rm.capital)

    eq = pd.Series(equity_curve, index=test_dates, name="Strategy")

    # Buy-and-hold benchmark on same capital
    bh = (close / float(close.iloc[0])) * initial_capital
    bh.name = "Buy & Hold"

    trades_df = pd.DataFrame(trades)
    metrics   = _compute_metrics(eq, trades_df, initial_capital)

    return eq, bh, trades_df, metrics


# ── Metrics ────────────────────────────────────────────────────────────────────

def _compute_metrics(eq: pd.Series, trades: pd.DataFrame, capital: float) -> dict:
    n_days   = len(eq)
    final    = float(eq.iloc[-1])
    total_r  = (final - capital) / capital

    daily_r  = eq.pct_change().dropna()
    ann_r    = (1 + total_r) ** (252 / max(n_days, 1)) - 1

    # Sharpe (annualised, rf=0 for simplicity)
    sharpe   = (daily_r.mean() / daily_r.std() * np.sqrt(252)
                if daily_r.std() > 0 else 0.0)

    # Sortino (downside deviation only)
    down_r   = daily_r[daily_r < 0]
    sortino  = (daily_r.mean() / down_r.std() * np.sqrt(252)
                if len(down_r) > 1 and down_r.std() > 0 else 0.0)

    # Max drawdown
    roll_max = eq.cummax()
    dd       = (eq - roll_max) / roll_max
    max_dd   = float(dd.min())

    # Calmar
    calmar   = (ann_r / abs(max_dd)) if max_dd != 0 else 0.0

    m = {
        "Total Return (%)":   round(total_r * 100, 2),
        "CAGR (%)":           round(ann_r * 100, 2),
        "Sharpe Ratio":       round(sharpe, 3),
        "Sortino Ratio":      round(sortino, 3),
        "Calmar Ratio":       round(calmar, 3),
        "Max Drawdown (%)":   round(max_dd * 100, 2),
        "Final Equity ($)":   round(final, 2),
    }

    if not trades.empty:
        wins       = trades[trades["PnL $"] > 0]
        losses     = trades[trades["PnL $"] <= 0]
        n_t        = len(trades)
        wr         = len(wins) / n_t
        gross_win  = float(wins["PnL $"].sum())  if not wins.empty  else 0.0
        gross_loss = float(abs(losses["PnL $"].sum())) if not losses.empty else 1e-9
        pf         = gross_win / gross_loss
        avg_win    = float(wins["PnL $"].mean())   if not wins.empty   else 0.0
        avg_loss   = float(losses["PnL $"].mean()) if not losses.empty else 0.0
        expectancy = wr * avg_win + (1 - wr) * avg_loss
        avg_hold   = (float(trades["Days Held"].mean())
                      if "Days Held" in trades.columns else 0.0)
        m.update({
            "Win Rate (%)":     round(wr * 100, 2),
            "Profit Factor":    round(pf, 3),
            "Expectancy ($)":   round(expectancy, 2),
            "Total Trades":     n_t,
            "Avg Win ($)":      round(avg_win, 2),
            "Avg Loss ($)":     round(avg_loss, 2),
            "Avg Hold (days)":  round(avg_hold, 1),
        })
    else:
        m.update({
            "Win Rate (%)": 0, "Profit Factor": 0, "Expectancy ($)": 0,
            "Total Trades": 0, "Avg Win ($)": 0, "Avg Loss ($)": 0,
            "Avg Hold (days)": 0,
        })

    return m
