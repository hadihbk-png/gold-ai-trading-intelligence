"""
Real-time signal generation for the latest bar.

Uses the stacking ensemble (preferred) or individual model majority vote.
Applies no-trade filters, weekly trend filter, and optional 4H confirmation.
"""

import numpy as np
from src.config import ATR_STOP_MULTIPLIER, RISK_REWARD_RATIO, MIN_CONFIDENCE
from src.risk import check_no_trade, RiskManager

SIGNAL_LABELS = {0: "DOWN",     1: "SIDEWAYS", 2: "UP"}
SIGNAL_COLORS = {0: "#FF4B4B",  1: "#888888",  2: "#00CC88"}
SIGNAL_EMOJI  = {0: "🔴",       1: "⚪",        2: "🟢"}

# ── Phase 4A: Directional override thresholds ─────────────────────────────────
_SIDEWAYS_GAP_PP   = 8.0   # UP–DOWN probability gap (pp) that forces directional
_SIDEWAYS_CONF_MAX = 55.0  # max confidence to remain SIDEWAYS
_SIDEWAYS_ATR_MAX  = 0.8   # max ATR/Price% to remain SIDEWAYS


def _apply_directional_override(
    signal_int: int,
    proba_vec,
    confidence: float,
    atr_pct: float,
) -> tuple[int, float]:
    """
    Phase 4A Enhancement 1 — fix chronic SIDEWAYS bias.

    Override SIDEWAYS → UP or DOWN when the model lacks genuine conviction for
    neutrality (gap too large, confidence too high, or volatility too high for
    a truly sideways day).

    Rules applied only when signal_int == 1 (SIDEWAYS):
      1. Forced directional: |P(UP) - P(DOWN)| > 8pp → use higher direction
      2. SIDEWAYS reservation: only stay SIDEWAYS when ALL hold:
           gap < 8pp AND confidence < 55% AND ATR/Price < 0.8%
    """
    if signal_int != 1 or proba_vec is None:
        return signal_int, confidence

    p_down = float(proba_vec[0]) * 100
    p_up   = float(proba_vec[2]) * 100
    gap    = abs(p_up - p_down)

    stay_sideways = (
        gap < _SIDEWAYS_GAP_PP
        and confidence < _SIDEWAYS_CONF_MAX
        and atr_pct < _SIDEWAYS_ATR_MAX
    )
    if stay_sideways:
        return signal_int, confidence

    # Force directional: use the higher-probability direction
    new_sig = 2 if p_up >= p_down else 0
    new_conf = float(proba_vec[new_sig]) * 100
    return new_sig, new_conf


def generate_latest_signal(
    df,
    reg_results: dict,
    clf_results: dict,
    feature_cols: list,
    stack_reg=None,
    stack_clf=None,
    regime_int: int = 5,
    current_drawdown: float = 0.0,
    use_weekly_filter: bool = True,
    use_4h_confirmation: bool = False,
    ticker: str = "GC=F",
) -> dict | None:
    """
    Predict next-day direction and generate a filtered trading signal.

    Applies in order:
      1. Stacking ensemble (or majority vote fallback)
      2. Custom probability thresholds (if stored in clf_results)
      3. No-trade filters (confidence, ATR, drawdown)
      4. Weekly trend filter (blocks counter-trend entries)
      5. Optional 4H confirmation gate
    """
    available = df.dropna(subset=feature_cols)
    if available.empty:
        return None

    latest = available.iloc[[-1]]
    X = latest[feature_cols].values

    # ── Regression prediction ─────────────────────────────────────────────────
    if stack_reg is not None:
        try:
            ensemble_price = float(stack_reg.predict(X)[0])
            individual_preds = {}
            for name, res in reg_results.items():
                if name in ("Stacking", "Ensemble"):
                    continue
                try:
                    individual_preds[name] = float(res["model"].predict(X)[0])
                except Exception:
                    pass
        except Exception:
            stack_reg = None

    if stack_reg is None:
        individual_preds = {}
        for name, res in reg_results.items():
            if name in ("Stacking", "Ensemble"):
                continue
            try:
                individual_preds[name] = float(res["model"].predict(X)[0])
            except Exception:
                pass
        ensemble_price = (float(np.mean(list(individual_preds.values())))
                          if individual_preds else None)

    # ── Classification prediction ─────────────────────────────────────────────
    if stack_clf is not None:
        try:
            proba_vec  = stack_clf.predict_proba(X)[0]

            # Use optimised thresholds if available
            stk_entry = clf_results.get("Stacking", {})
            t_up   = stk_entry.get("thresholds", {}).get("threshold_up",   0.50)
            t_down = stk_entry.get("thresholds", {}).get("threshold_down", 0.50)

            p_up   = float(proba_vec[2])
            p_down = float(proba_vec[0])
            if p_up > t_up and p_up >= p_down:
                raw_signal = 2
            elif p_down > t_down and p_down > p_up:
                raw_signal = 0
            else:
                raw_signal = 1

            confidence = float(proba_vec[raw_signal] * 100)

            # ── Phase 4A: LSTM-augmented meta-clf (if trained and files present) ──
            _aug_meta  = getattr(stack_clf, "_augmented_meta_clf", None)
            _lstm_pred = (clf_results or {}).get("Stacking", {}).get("lstm_predictor")
            if _aug_meta is not None and _lstm_pred is not None:
                try:
                    _X_recent   = available[feature_cols].values        # (n, n_feat)
                    _lstm_proba = _lstm_pred.predict_proba_from_recent(_X_recent)
                    if _lstm_proba is not None:
                        _l1_proba  = stack_clf._l1_proba(X)             # (1, 9)
                        _aug_input = np.hstack([_l1_proba, _lstm_proba])  # (1, 12)
                        _aug_proba = _aug_meta.predict_proba(_aug_input)[0]  # (3,)
                        proba_vec  = _aug_proba
                        p_up   = float(proba_vec[2])
                        p_down = float(proba_vec[0])
                        if p_up > t_up and p_up >= p_down:
                            raw_signal = 2
                        elif p_down > t_down and p_down > p_up:
                            raw_signal = 0
                        else:
                            raw_signal = 1
                        confidence = float(proba_vec[raw_signal] * 100)
                except Exception:
                    pass  # fall back to tree-only probabilities

        except Exception:
            stack_clf = None

    if stack_clf is None:
        proba_vec = None
        votes, probas = [], []
        for name, res in clf_results.items():
            if name == "Stacking":
                continue
            try:
                v = int(np.array(res["model"].predict(X)).ravel()[0])
                votes.append(v)
                if hasattr(res["model"], "predict_proba"):
                    probas.append(res["model"].predict_proba(X)[0])
            except Exception:
                pass
        raw_signal = (int(np.bincount(votes, minlength=3).argmax())
                      if votes else 1)
        confidence = None
        if probas:
            mean_p = np.mean(probas, axis=0)
            proba_vec = mean_p
            confidence = float(mean_p[raw_signal] * 100)

    # ── No-trade filters ──────────────────────────────────────────────────────
    current_price = float(df["Close"].iloc[-1])
    atr_val       = (float(df["ATR"].iloc[-1])
                     if "ATR" in df.columns else current_price * 0.01)

    # Phase 4A: apply directional bias correction now that ATR/Price is known
    if proba_vec is not None:
        _atr_pct = atr_val / current_price * 100 if current_price else 1.0
        raw_signal, confidence = _apply_directional_override(
            raw_signal, proba_vec, confidence, _atr_pct
        )

    conf_frac     = confidence / 100 if confidence is not None else None

    trade_ok, filter_reason = check_no_trade(
        signal_int=raw_signal,
        confidence=conf_frac,
        atr_val=atr_val,
        price=current_price,
        current_drawdown=current_drawdown,
    )

    filtered_signal = raw_signal if trade_ok else 1

    # ── Weekly trend filter ───────────────────────────────────────────────────
    weekly_trend = 0
    weekly_filter_applied = False
    if use_weekly_filter and "Weekly_Trend" in df.columns:
        wt = df["Weekly_Trend"].iloc[-1]
        if not np.isnan(wt):
            weekly_trend = int(wt)
            try:
                from src.timeframes import apply_weekly_filter
                filtered_signal_w = apply_weekly_filter(filtered_signal, weekly_trend)
                if filtered_signal_w != filtered_signal:
                    weekly_filter_applied = True
                    filter_reason = (filter_reason or "") + " [Counter-trend blocked]"
                filtered_signal = filtered_signal_w
            except Exception:
                pass

    # ── 4H confirmation ───────────────────────────────────────────────────────
    h4_signal_data = None
    h4_confirmed   = True
    if use_4h_confirmation and filtered_signal != 1:
        try:
            from src.timeframes import get_4h_signal
            h4 = get_4h_signal(ticker)
            h4_signal_data = h4
            if h4.get("available") and h4.get("signal") is not None:
                h4_sig = int(h4["signal"])
                # Require 4H to agree with daily direction
                if h4_sig != filtered_signal:
                    h4_confirmed = False
                    filter_reason = (filter_reason or "") + " [4H disagreement]"
                    filtered_signal = 1
        except Exception:
            pass

    final_signal = filtered_signal

    # ── Trade levels ──────────────────────────────────────────────────────────
    stop_loss = take_profit = None
    if final_signal != 1:
        sl_dist = ATR_STOP_MULTIPLIER * atr_val
        if final_signal == 2:
            stop_loss   = current_price - sl_dist
            take_profit = current_price + sl_dist * RISK_REWARD_RATIO
        else:
            stop_loss   = current_price + sl_dist
            take_profit = current_price - sl_dist * RISK_REWARD_RATIO

    return {
        "current_price":         current_price,
        "predicted_price":       ensemble_price,
        "individual_reg_preds":  individual_preds,
        "raw_signal_int":        raw_signal,
        "signal_int":            final_signal,
        "signal_label":          SIGNAL_LABELS[final_signal],
        "signal_color":          SIGNAL_COLORS[final_signal],
        "signal_emoji":          SIGNAL_EMOJI[final_signal],
        "confidence":            confidence,
        "confidence_pct":        round(confidence, 1) if confidence else None,
        "proba_vec":             proba_vec.tolist() if proba_vec is not None else None,
        "filter_reason":         filter_reason if not trade_ok or weekly_filter_applied or not h4_confirmed else None,
        "stop_loss":             stop_loss,
        "take_profit":           take_profit,
        "atr":                   atr_val,
        "regime_int":            regime_int,
        "latest_date":           df.index[-1],
        "using_stacking":        stack_clf is not None,
        "weekly_trend":          weekly_trend,
        "weekly_filter_applied": weekly_filter_applied,
        "h4_signal":             h4_signal_data,
        "h4_confirmed":          h4_confirmed,
    }
