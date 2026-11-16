"""
src/compare.py — Multi-Metal Comparison data builder (Tier 3 · item 2).

Produces a UNIFORM snapshot (price, daily change, signal, confidence,
directional probabilities, regime, verdict) for Gold, Silver and Platinum by
REUSING the exact functions the Dashboard uses:
    - src.train.load_models / load_metal_models   (same trained models)
    - src.data_loader.download_data / download_metal_ohlcv + src.features
    - src.regime.get_current_regime
    - src.signals.generate_latest_signal
    - src.data_loader.get_live_spot_price / get_silver_price / get_platinum_price

PRICE PARITY WITH THE DASHBOARD
    The displayed price and daily-change mirror the Dashboard's headline price
    exactly, including its (intentional) per-metal basis:
      • Gold     : live Twelve Data spot; change = spot vs last futures close.
      • Silver   : live waterfall (Alpha Vantage); change = live vs prior close.
      • Platinum : live Swissquote waterfall; change = live vs prior close.
    If a live feed is unavailable, it falls back to the model's last close (same
    as the Dashboard). The Gold/Silver ratio uses gold's futures last close over
    silver's live price — identical to the Dashboard.

No new modelling, no new data source. Additive and read-only: does NOT touch
models, thresholds, the prediction ledger, or any existing file. The verdict
rule mirrors app.py:compute_decision_verdict (kept in sync here as a pure
function so this module need not import app.py).
"""
from __future__ import annotations

import numpy as np

# Metals to compare, in display order.
METALS = [
    {"key": "gold",     "name": "Gold",     "symbol": "XAU/USD", "ticker": "GC=F"},
    {"key": "silver",   "name": "Silver",   "symbol": "XAG/USD", "ticker": "SI=F"},
    {"key": "platinum", "name": "Platinum", "symbol": "XPT/USD", "ticker": "PL=F"},
]


def _regime_bucket(regime_label: str) -> str:
    """Map a regime label to the bucket compute_decision_verdict expects."""
    rl = (regime_label or "").lower()
    if "vol" in rl or "high" in rl:
        return "high_vol"
    if "trend" in rl:
        return "trending"
    return "neutral"


def _verdict(signal, confidence, regime, rsi, bb_pctb, rolling_accuracy,
             no_trade_score, classifier_consensus):
    """
    Pure mirror of app.py:compute_decision_verdict (verdict label + color only).
    Keep in sync with app.py if the rule changes.
    """
    reasons_for = 0
    reasons_against = 0

    if confidence >= 0.60:
        reasons_for += 1
    elif confidence < 0.50:
        reasons_against += 1

    if rolling_accuracy >= 0.40:
        reasons_for += 1
    elif rolling_accuracy < 0.33:
        reasons_against += 1

    if regime == "high_vol":
        reasons_against += 1
    elif regime == "trending":
        reasons_for += 1

    if signal == "UP" and rsi > 75:
        reasons_against += 1
    if signal == "DOWN" and rsi < 25:
        reasons_against += 1
    if bb_pctb > 0.9:
        reasons_against += 1
    if bb_pctb < 0.1:
        reasons_against += 1

    if classifier_consensus < 0.67:
        reasons_against += 1

    if no_trade_score >= 3:
        reasons_against += 1

    if reasons_against >= 3:
        return ("NO-TRADE", "🔴", "#ef4444")
    elif reasons_against >= 2:
        return ("CAUTION", "🟡", "#f59e0b")
    elif reasons_against <= 1 and reasons_for >= 2:
        return ("GO", "🟢", "#22c55e")
    return ("CAUTION", "🟡", "#f59e0b")


def _load_bundle_and_df(metal_key: str):
    """Return (df, bundle_dict) for a metal, or (None, None) if unavailable."""
    from src.data_loader import download_data, download_metal_ohlcv
    from src.features import add_features, classify_regime

    if metal_key == "gold":
        from src.train import load_models
        reg, clf, feat, sr, sc = load_models()
        if reg is None or clf is None or feat is None:
            return None, None
        bundle = {
            "reg_results": reg, "clf_results": clf, "feature_cols": feat,
            "stack_reg": sr, "stack_clf": sc,
        }
        raw = download_data()
        if raw is None or getattr(raw, "empty", True):
            return None, None
        df = add_features(raw)  # Gold = add_features(download_data()), matches app.py
        return df, bundle

    # Silver / Platinum
    from src.train import load_metal_models
    bundles = load_metal_models() or {}
    bundle = bundles.get(metal_key)
    if bundle is None:
        return None, None
    ticker = "SI=F" if metal_key == "silver" else "PL=F"
    raw = download_metal_ohlcv(ticker, years=2)
    if raw is None or raw.empty:
        return None, None
    df = classify_regime(add_features(raw))
    return df, bundle


def _live_price(metal_key: str, td_key: str, av_key: str):
    """Fetch the live spot price for a metal — SAME feeds as the Dashboard.
    Returns (price_or_None, source_str)."""
    from src.data_loader import (
        get_live_spot_price, get_silver_price, get_platinum_price,
    )
    try:
        if metal_key == "gold":
            return get_live_spot_price(td_key, av_key)
        if metal_key == "silver":
            return get_silver_price(av_key)
        return get_platinum_price(td_key, av_key)
    except Exception as exc:
        return None, f"unavailable ({exc})"


def _snapshot(meta: dict, td_key: str = "", av_key: str = "") -> dict:
    """Build one metal's comparison snapshot. Never raises — errors go in 'error'."""
    from src.regime import get_current_regime
    from src.signals import generate_latest_signal

    out = {"name": meta["name"], "symbol": meta["symbol"], "ok": False, "error": None}
    try:
        df, bundle = _load_bundle_and_df(meta["key"])
        if df is None or bundle is None or df.empty:
            out["error"] = "model or data unavailable"
            return out

        regime_info = get_current_regime(df) or {}
        sig = generate_latest_signal(
            df,
            bundle["reg_results"], bundle["clf_results"], bundle["feature_cols"],
            stack_reg=bundle.get("stack_reg"), stack_clf=bundle.get("stack_clf"),
            regime_int=regime_info.get("regime_int", 5),
        )
        if sig is None:
            out["error"] = "signal unavailable"
            return out

        # ── Price + daily change: mirror the Dashboard headline exactly ──────
        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
        live, price_src = _live_price(meta["key"], td_key, av_key)

        if meta["key"] == "gold":
            # Dashboard: live spot vs last futures close; else last vs prev close.
            if live is not None:
                price = float(live)
                base = last_close
            else:
                price = last_close
                base = prev_close
                price_src = price_src or "last close"
        else:
            # Silver / Platinum: live vs PRIOR close; fallback to last close.
            if live is not None:
                price = float(live)
            else:
                price = last_close
                price_src = "yfinance (last close)"
            base = prev_close
        change_pct = ((price - base) / base * 100) if base else 0.0

        conf_pct = sig.get("confidence_pct")
        conf = (conf_pct if conf_pct is not None else 50) / 100
        proba = sig.get("proba_vec") or [None, None, None]
        signal_label = sig["signal_label"]

        # ── Verdict inputs (same derivation as the Dashboard's DIC) ──────────
        rsi = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 50.0
        bbp = float(df["BB_PctB"].iloc[-1]) if "BB_PctB" in df.columns else 0.5
        regime_bucket = _regime_bucket(regime_info.get("regime_label", ""))

        roll_acc = 0.35
        stk = (bundle.get("clf_results") or {}).get("Stacking", {})
        sy, sp = stk.get("y_test"), stk.get("predictions")
        if sy is not None and sp is not None:
            arr = (np.array(sy) == np.array(sp)).astype(int)
            if len(arr) >= 5:
                roll_acc = float(arr[-30:].mean())

        nt = 0
        if conf < 0.50:
            nt += 1
        if sig.get("filter_reason"):
            nt += 1
        if signal_label == "UP" and rsi > 75:
            nt += 1
        if signal_label == "DOWN" and rsi < 25:
            nt += 1
        if bbp > 0.9 or bbp < 0.1:
            nt += 1

        consensus = 1.0
        try:
            fc = bundle.get("feature_cols") or []
            X = df.dropna(subset=fc).iloc[[-1]][fc].values if fc else None
            if X is not None:
                votes = {}
                for mn in ("XGBoost", "LightGBM", "CatBoost"):
                    mdl = (bundle.get("clf_results") or {}).get(mn, {}).get("model")
                    if mdl is not None:
                        try:
                            votes[mn] = int(np.array(mdl.predict(X)).ravel()[0])
                        except Exception:
                            pass
                if votes:
                    consensus = sum(1 for v in votes.values()
                                    if v == sig["signal_int"]) / len(votes)
        except Exception:
            pass

        verdict, v_emoji, v_color = _verdict(
            signal_label, conf, regime_bucket, rsi, bbp, roll_acc, nt, consensus)

        out.update({
            "ok": True,
            "price": price,                 # Dashboard headline price (live spot)
            "change_pct": change_pct,
            "price_source": price_src,
            "last_close": last_close,       # model's futures last close (for GSR)
            "signal": signal_label,
            "signal_int": sig["signal_int"],
            "signal_color": sig["signal_color"],
            "confidence_pct": conf_pct,
            "p_down": (proba[0] * 100) if proba[0] is not None else None,
            "p_side": (proba[1] * 100) if proba[1] is not None else None,
            "p_up": (proba[2] * 100) if proba[2] is not None else None,
            "regime": regime_info.get("regime_label", "—"),
            "verdict": verdict,
            "verdict_emoji": v_emoji,
            "verdict_color": v_color,
            "filter_reason": sig.get("filter_reason"),
            "as_of": str(df.index[-1])[:10],
        })
        return out
    except Exception as exc:  # never let one metal break the page
        out["error"] = str(exc)
        return out


def build_comparison(td_key: str = "", av_key: str = "") -> list:
    """Return a list of per-metal snapshot dicts for Gold, Silver, Platinum.

    Pass the Dashboard's API keys (TWELVE_DATA_API_KEY, ALPHA_VANTAGE_API_KEY)
    so the price row uses the SAME live feeds the Dashboard shows.
    """
    return [_snapshot(m, td_key, av_key) for m in METALS]
