"""
Gold AI - Live Forward Validation

Captures one prediction snapshot per market date and scores it once the next
gold close is available. This page does not train or alter model logic.
"""
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import DATA_DIR, DIRECTION_THRESHOLD
from src.data_loader import download_data
from src.features import add_features
from src.macro_loader import add_macro_features
from src.regime import get_current_regime
from src.signals import generate_latest_signal
from src.train import load_models

st.set_page_config(
    page_title="Gold AI Decision Intelligence — Live Validation",
    page_icon="OK",
    layout="wide",
)

DARK_BG = "#0e1117"
GRID_CLR = "#1e2130"
LOG_PATH = os.path.join(DATA_DIR, "live_validation_log.csv")

_PLT = dict(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font=dict(color="white"))

# market_bar_date stores the index date of the latest gold bar used for prediction
COLUMNS = [
    "prediction_date",
    "market_bar_date",
    "timestamp_utc",
    "gold_price",
    "signal",
    "confidence",
    "prob_down",
    "prob_sideways",
    "prob_up",
    "regime",
    "atr",
    "vix",
    "actual_date",
    "actual_price",
    "actual_move_pct",
    "correct",
]


def _dark(fig, height=360):
    fig.update_layout(**_PLT, height=height, legend=dict(orientation="h", y=1.02))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
    return fig


_STR_COLS = {"prediction_date", "market_bar_date", "timestamp_utc", "actual_date", "signal", "regime", "correct"}
_NUM_COLS = {"gold_price", "confidence", "prob_down", "prob_sideways", "prob_up", "actual_price", "actual_move_pct"}


def _load_log() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=COLUMNS)
    log = pd.read_csv(LOG_PATH)
    for col in COLUMNS:
        if col not in log.columns:
            log[col] = np.nan
    # Force string/date columns to object so NaN-only columns don't infer as float64
    for col in _STR_COLS:
        if col in log.columns:
            log[col] = log[col].astype(object)
    # Force numeric columns
    for col in _NUM_COLS:
        if col in log.columns:
            log[col] = pd.to_numeric(log[col], errors="coerce")
    if "correct" in log.columns:
        log["correct"] = log["correct"].map({
            True: True,
            False: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            1: True,
            0: False,
        })
        # pandas 3.x: .map() on an all-NaN Series infers float64; re-enforce object
        log["correct"] = log["correct"].astype(object)
    return log[COLUMNS]


def _runtime_prediction_date(now_utc: datetime | None = None) -> str:
    override = os.getenv("LIVE_VALIDATION_DATE", "").strip()
    if override:
        return pd.to_datetime(override).strftime("%Y-%m-%d")
    now = now_utc or datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d")


def _save_log(log: pd.DataFrame) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    log = log.copy()
    log["prediction_date"] = pd.to_datetime(log["prediction_date"]).dt.strftime("%Y-%m-%d")
    if "market_bar_date" in log.columns:
        log["market_bar_date"] = pd.to_datetime(
            log["market_bar_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    log = log.sort_values("prediction_date").drop_duplicates("prediction_date", keep="last")
    log.to_csv(LOG_PATH, index=False)


def _vix_value(df: pd.DataFrame) -> float | None:
    vix_cols = [c for c in df.columns if "VIX" in c.upper() and c.endswith("_Close")]
    if not vix_cols:
        return None
    val = df[vix_cols[0]].iloc[-1]
    return None if pd.isna(val) else float(val)


def _score_signal(signal: str, move_pct: float) -> bool:
    sig = str(signal).upper()
    if sig in ("UP", "BUY", "LONG"):
        return move_pct > DIRECTION_THRESHOLD * 100
    if sig in ("DOWN", "SELL", "SHORT"):
        return move_pct < -DIRECTION_THRESHOLD * 100
    return abs(move_pct) <= DIRECTION_THRESHOLD * 100


def _score_open_predictions(log: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return log

    scored = log.copy()
    # Enforce object dtypes unconditionally — pandas 3.x raises TypeError on dtype-incompatible
    # .at[] assignments (e.g. assigning a string to a float64 column).  astype(object) on an
    # already-object column is a no-op, so this is safe to call unconditionally.
    if "actual_date" not in scored.columns:
        scored["actual_date"] = pd.Series(dtype=object)
    else:
        scored["actual_date"] = scored["actual_date"].astype(object)
    if "correct" not in scored.columns:
        scored["correct"] = pd.Series(dtype=object)
    else:
        scored["correct"] = scored["correct"].astype(object)

    prices = df[["Close"]].copy()
    prices.index = pd.to_datetime(prices.index).normalize()

    for idx, row in scored.iterrows():
        if pd.notna(row.get("correct")):
            continue
        if pd.isna(row.get("signal")):  # BACKFILLED rows have no signal — never score them
            continue

        pred_date = pd.to_datetime(row["prediction_date"]).normalize()
        future = prices[prices.index > pred_date]
        if future.empty:
            continue

        actual_date = future.index[0]
        actual_price = float(future["Close"].iloc[0])
        pred_price = float(row["gold_price"])
        move_pct = (actual_price - pred_price) / pred_price * 100

        scored.at[idx, "actual_date"] = actual_date.strftime("%Y-%m-%d")
        scored.at[idx, "actual_price"] = actual_price
        scored.at[idx, "actual_move_pct"] = move_pct
        scored.at[idx, "correct"] = bool(_score_signal(row["signal"], move_pct))

    return scored


def _check_market_bar_stale(log: pd.DataFrame, market_bar_date: str) -> bool:
    """Return True if the latest market bar is not newer than the last real snapshot bar.
    BACKFILLED rows are excluded so they cannot block new real predictions."""
    if log.empty:
        return False
    real = log[log["timestamp_utc"].astype(str) != "BACKFILLED"]
    if real.empty or "market_bar_date" not in real.columns:
        return False
    last_mbd = pd.to_datetime(
        real.sort_values("prediction_date").iloc[-1].get("market_bar_date", ""),
        errors="coerce",
    )
    current_mbd = pd.to_datetime(market_bar_date, errors="coerce")
    if pd.isna(last_mbd) or pd.isna(current_mbd):
        return False
    return current_mbd <= last_mbd


def _backfill_missing_dates(log: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Insert audit rows for market days absent from the log.
    Prices come from real market data; prediction fields are left NaN —
    nothing is fabricated."""
    if df is None or df.empty or log.empty:
        return log

    prices = df[["Close"]].copy()
    prices.index = pd.DatetimeIndex(prices.index).normalize()
    market_dates = prices.index.unique()

    existing = set(pd.to_datetime(log["prediction_date"], errors="coerce").dt.normalize())
    log_min = pd.to_datetime(log["prediction_date"], errors="coerce").min().normalize()
    today = pd.Timestamp(datetime.now(timezone.utc).date())

    gap_dates = [d for d in market_dates if log_min < d < today and d not in existing]
    if not gap_dates:
        return log

    new_rows = []
    for gap_date in sorted(gap_dates):
        close_price = float(prices.loc[gap_date, "Close"])
        new_rows.append({
            "prediction_date": gap_date.strftime("%Y-%m-%d"),
            "market_bar_date": gap_date.strftime("%Y-%m-%d"),
            "timestamp_utc": "BACKFILLED",
            "gold_price": close_price,
            "signal": np.nan,
            "confidence": np.nan,
            "prob_down": np.nan,
            "prob_sideways": np.nan,
            "prob_up": np.nan,
            "regime": np.nan,
            "atr": np.nan,
            "vix": np.nan,
            "actual_date": np.nan,
            "actual_price": np.nan,
            "actual_move_pct": np.nan,
            "correct": np.nan,
        })

    return pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)


def _append_today_snapshot(
    log: pd.DataFrame,
    df: pd.DataFrame,
    market_bar_date: str,
) -> tuple[pd.DataFrame, str]:
    now_utc = datetime.now(timezone.utc)
    reg, clf, feat, stack_reg, stack_clf = load_models()
    if reg is None:
        return log, "Saved model artifact not found. No snapshot captured."

    regime_info = get_current_regime(df)
    signal = generate_latest_signal(
        df,
        reg,
        clf,
        feat,
        stack_reg=stack_reg,
        stack_clf=stack_clf,
        regime_int=regime_info["regime_int"] if regime_info else 5,
    )
    if not signal:
        return log, "Signal could not be generated. No snapshot captured."

    prediction_date = _runtime_prediction_date(now_utc)
    if not log.empty and prediction_date in set(log["prediction_date"].astype(str)):
        return log, f"Snapshot already exists for {prediction_date}."

    proba = signal.get("proba_vec") or [np.nan, np.nan, np.nan]
    row = {
        "prediction_date": prediction_date,
        "market_bar_date": market_bar_date,
        "timestamp_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "gold_price": float(signal["current_price"]),
        "signal": signal["signal_label"],
        "confidence": signal.get("confidence_pct"),
        "prob_down": float(proba[0]) * 100 if len(proba) > 0 else np.nan,
        "prob_sideways": float(proba[1]) * 100 if len(proba) > 1 else np.nan,
        "prob_up": float(proba[2]) * 100 if len(proba) > 2 else np.nan,
        "regime": regime_info["regime_label"] if regime_info else "Neutral",
        "atr": signal.get("atr"),
        "vix": _vix_value(df),
        "actual_date": np.nan,
        "actual_price": np.nan,
        "actual_move_pct": np.nan,
        "correct": np.nan,
    }
    return pd.concat([log, pd.DataFrame([row])], ignore_index=True), f"Captured snapshot for {prediction_date} (market bar: {market_bar_date})."


def _accuracy(series: pd.Series) -> str:
    valid = series.dropna()
    if valid.empty:
        return "-"
    return f"{valid.astype(bool).mean() * 100:.1f}%"


# ── Session state init ─────────────────────────────────────────────────────────
for _k, _v in [("lv_df", None), ("lv_force_capture", False)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.title("Live Forward Validation")
st.caption("Daily out-of-sample prediction log. No retraining. No model logic changes.")

# ── Force-refresh market data once per session ─────────────────────────────────
# Always fetches fresh bars from yfinance on the first page load so that
# snapshot values reflect the latest available market data, not a stale cache.
if st.session_state.lv_df is None:
    with st.spinner("Downloading latest market data…"):
        try:
            _raw = download_data(force_refresh=True)
            st.session_state.lv_df = add_macro_features(add_features(_raw), None)
        except Exception as exc:
            st.error(f"Failed to load market data: {exc}")
            st.stop()

df = st.session_state.lv_df
market_bar_date = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d")
prediction_date = _runtime_prediction_date()

# ── Load, backfill gaps, and score existing log ────────────────────────────────
log_df = _load_log()
_pre_backfill_len = len(log_df)
log_df = _backfill_missing_dates(log_df, df)
_backfilled = len(log_df) > _pre_backfill_len
log_df = _score_open_predictions(log_df, df)
if _backfilled:
    _save_log(log_df)

today_exists = (
    not log_df.empty
    and prediction_date in set(log_df["prediction_date"].astype(str))
)

# ── Snapshot capture logic ─────────────────────────────────────────────────────
if today_exists:
    st.info(f"Snapshot already exists for {prediction_date}.")
else:
    is_stale = _check_market_bar_stale(log_df, market_bar_date)

    if is_stale and not st.session_state.lv_force_capture:
        st.warning(
            f"Latest market data has not advanced; snapshot would use market bar {market_bar_date}."
        )
        if st.button("Capture anyway"):
            st.session_state.lv_force_capture = True
            st.rerun()
    else:
        st.session_state.lv_force_capture = False
        log_df, capture_msg = _append_today_snapshot(log_df, df, market_bar_date)
        log_df = _score_open_predictions(log_df, df)
        _save_log(log_df)
        st.info(capture_msg)

st.caption(f"Market bar date: {market_bar_date} | Validation threshold: +/-{DIRECTION_THRESHOLD * 100:.2f}%")

scored = log_df[log_df["correct"].notna()].copy()
if not scored.empty:
    scored["correct_bool"] = scored["correct"].astype(bool)

# ── Extended Statistical Validation Summary ───────────────────────────────────
st.divider()
st.subheader("Extended Statistical Validation")

# REAL rows only (BACKFILLED audit rows excluded throughout)
_esv_real         = log_df[log_df["timestamp_utc"].astype(str) != "BACKFILLED"].copy()
_esv_total_preds  = len(_esv_real)

# Scored REAL rows
_esv_scored       = _esv_real[_esv_real["correct"].notna()].copy()
_esv_total_scored = len(_esv_scored)
if not _esv_scored.empty:
    _esv_scored["_cb"] = _esv_scored["correct"].astype(bool)

# Derived stats
_esv_overall_acc = _esv_scored["_cb"].mean() * 100 if _esv_total_scored > 0 else None

if _esv_total_scored > 0:
    _esv_sw           = _esv_scored[_esv_scored["signal"].astype(str).str.upper().eq("SIDEWAYS")]
    _esv_sideways_acc = _esv_sw["_cb"].mean() * 100 if len(_esv_sw) > 0 else None
    _esv_conf_c       = _esv_scored.loc[_esv_scored["_cb"], "confidence"].dropna()
    _esv_conf_i       = _esv_scored.loc[~_esv_scored["_cb"], "confidence"].dropna()
    _esv_avg_conf_c   = _esv_conf_c.mean() if len(_esv_conf_c) > 0 else None
    _esv_avg_conf_i   = _esv_conf_i.mean() if len(_esv_conf_i) > 0 else None
    _esv_last5        = _esv_scored.sort_values("prediction_date").tail(5)
    _esv_last5_acc    = _esv_last5["_cb"].mean() * 100
else:
    _esv_sideways_acc = _esv_avg_conf_c = _esv_avg_conf_i = _esv_last5_acc = None

_esv_fmt = lambda v, s="": f"{v:.1f}{s}" if v is not None else "—"

# Rolling accuracy metrics row
e1, e2, e3, e4, e5, e6 = st.columns(6)
e1.metric("Total Predictions",    _esv_total_preds,
          help="REAL snapshot rows only — BACKFILLED audit rows excluded")
e2.metric("Total Scored",         _esv_total_scored)
e3.metric("Overall Accuracy",     _esv_fmt(_esv_overall_acc,  "%"))
e4.metric("Sideways Accuracy",    _esv_fmt(_esv_sideways_acc, "%"))
e5.metric("Avg Conf — Correct",   _esv_fmt(_esv_avg_conf_c,  "%"))
e6.metric("Avg Conf — Incorrect", _esv_fmt(_esv_avg_conf_i,  "%"))

# Observation status banner
if _esv_total_scored < 30:
    st.warning(
        "Early Stage — Minimum 30 observations needed "
        "for statistically meaningful conclusions."
    )
elif _esv_total_scored < 100:
    st.info("Developing — Results becoming more indicative.")
else:
    st.success("Statistically Meaningful — Results are reliable.")

# Trend indicator: last 5 scored vs overall
if _esv_last5_acc is not None and _esv_overall_acc is not None:
    _esv_delta = _esv_last5_acc - _esv_overall_acc
    _esv_arrow, _esv_tcolor, _esv_tlbl = (
        ("↑", "#00CC88", "Improving") if _esv_delta > 0 else
        ("↓", "#FF4B4B", "Declining") if _esv_delta < 0 else
        ("→", "#888888", "Stable")
    )
    st.markdown(
        f"""<div style="margin:10px 0;padding:10px 16px;background:#1e2130;
            border-radius:8px;display:inline-block;">
            <span style="font-size:0.82em;color:#aaa;">Last 5 scored &nbsp;</span>
            <span style="font-weight:bold;color:white;">{_esv_last5_acc:.1f}%</span>
            <span style="font-size:1.15em;color:{_esv_tcolor};margin-left:10px;">
                {_esv_arrow} {_esv_tlbl}</span>
            <span style="font-size:0.78em;color:#888;margin-left:8px;">
                vs {_esv_overall_acc:.1f}% overall</span>
        </div>""",
        unsafe_allow_html=True,
    )

st.caption(
    "⚠️ Accuracy statistics are based on live out-of-sample predictions only. "
    "Extended observation is required before drawing performance conclusions."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Observations", int(len(scored)))
k2.metric("Accuracy", _accuracy(scored["correct"] if not scored.empty else pd.Series(dtype=object)))
k3.metric("Buy Accuracy", _accuracy(scored.loc[scored["signal"].str.upper().isin(["UP", "BUY"]), "correct"] if not scored.empty else pd.Series(dtype=object)))
k4.metric("Sell Accuracy", _accuracy(scored.loc[scored["signal"].str.upper().isin(["DOWN", "SELL"]), "correct"] if not scored.empty else pd.Series(dtype=object)))
k5.metric("Sideways Accuracy", _accuracy(scored.loc[scored["signal"].str.upper().eq("SIDEWAYS"), "correct"] if not scored.empty else pd.Series(dtype=object)))
st.caption(
    "Accuracy calculated on REAL predictions only. "
    "BACKFILLED audit rows have no signal and are excluded from all accuracy metrics."
)

st.divider()
st.subheader("Validation Table")

display_df = log_df.sort_values("prediction_date", ascending=False).copy()
for col in ["gold_price", "confidence", "prob_down", "prob_sideways", "prob_up", "atr", "vix", "actual_price", "actual_move_pct"]:
    if col in display_df.columns:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(3)

# Tag each row as REAL or BACKFILLED and move to first column (display only, not saved to CSV)
display_df.insert(0, "Source", np.where(
    display_df["timestamp_utc"].astype(str) == "BACKFILLED", "BACKFILLED", "REAL"
))

# Replace NaN with "—" for BACKFILLED rows so the table is readable.
# Cast each column to object first: pandas 3.x raises TypeError when assigning a
# string ("—") into a float64 column, even in a display-only copy.
_backfill_mask = display_df["Source"] == "BACKFILLED"
_blank_cols = [
    "signal", "confidence", "prob_down", "prob_sideways", "prob_up",
    "regime", "atr", "vix", "actual_date", "actual_price", "actual_move_pct", "correct",
]
for _col in _blank_cols:
    if _col in display_df.columns:
        display_df[_col] = display_df[_col].astype(object)
        display_df.loc[_backfill_mask, _col] = display_df.loc[_backfill_mask, _col].fillna("—")

st.dataframe(display_df, hide_index=True, use_container_width=True)
st.caption("REAL = live prediction captured by the model · BACKFILLED = audit gap row (price only, no signal)")

st.divider()
c1, c2 = st.columns(2)

with c1:
    st.subheader("Cumulative Performance")
    if scored.empty:
        st.info("Waiting for at least one scored prediction.")
    else:
        chart_df = scored.sort_values("prediction_date").copy()
        chart_df["cumulative_accuracy"] = chart_df["correct_bool"].expanding().mean() * 100
        chart_df["cumulative_score"] = np.where(chart_df["correct_bool"], 1, -1).cumsum()
        chart_df["observation"] = np.arange(1, len(chart_df) + 1)
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=chart_df["observation"],
            y=chart_df["cumulative_score"],
            mode="lines+markers",
            name="Cumulative Score",
            line=dict(color="#00CC88", width=2.5),
        ))
        fig_acc.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
        fig_acc.update_layout(yaxis_title="Correct - Incorrect", xaxis_title="Scored Observation")
        st.plotly_chart(_dark(fig_acc), use_container_width=True)

with c2:
    st.subheader("Confidence vs Correctness")
    if scored.empty:
        st.info("Waiting for scored predictions.")
    else:
        scatter_df = scored.sort_values("prediction_date").copy()
        scatter_df["result"] = np.where(scatter_df["correct_bool"], "Correct", "Incorrect")
        colors = {"Correct": "#00CC88", "Incorrect": "#FF4B4B"}
        fig_conf = go.Figure()
        for result, part in scatter_df.groupby("result"):
            fig_conf.add_trace(go.Scatter(
                x=part["confidence"],
                y=part["actual_move_pct"],
                mode="markers",
                name=result,
                marker=dict(color=colors[result], size=10, line=dict(color="white", width=0.5)),
                text=part["prediction_date"] + " | " + part["signal"].astype(str),
            ))
        fig_conf.add_hline(y=DIRECTION_THRESHOLD * 100, line_color="#888", line_dash="dot")
        fig_conf.add_hline(y=-DIRECTION_THRESHOLD * 100, line_color="#888", line_dash="dot")
        fig_conf.update_layout(xaxis_title="Confidence (%)", yaxis_title="Actual Move (%)")
        st.plotly_chart(_dark(fig_conf), use_container_width=True)
