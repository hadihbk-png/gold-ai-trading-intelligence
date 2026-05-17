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
    page_title="Gold AI - Live Validation",
    page_icon="OK",
    layout="wide",
)

DARK_BG = "#0e1117"
GRID_CLR = "#1e2130"
LOG_PATH = os.path.join(DATA_DIR, "live_validation_log.csv")

_PLT = dict(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font=dict(color="white"))

COLUMNS = [
    "prediction_date",
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


def _load_log() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=COLUMNS)
    log = pd.read_csv(LOG_PATH)
    for col in COLUMNS:
        if col not in log.columns:
            log[col] = np.nan
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
    return log[COLUMNS]


def _save_log(log: pd.DataFrame) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    log = log.copy()
    log["prediction_date"] = pd.to_datetime(log["prediction_date"]).dt.strftime("%Y-%m-%d")
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
    prices = df[["Close"]].copy()
    prices.index = pd.to_datetime(prices.index).normalize()

    for idx, row in scored.iterrows():
        if pd.notna(row.get("correct")):
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


def _append_today_snapshot(log: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
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

    prediction_date = pd.to_datetime(df.index[-1]).strftime("%Y-%m-%d")
    if not log.empty and prediction_date in set(log["prediction_date"].astype(str)):
        return log, f"Snapshot already exists for {prediction_date}."

    proba = signal.get("proba_vec") or [np.nan, np.nan, np.nan]
    row = {
        "prediction_date": prediction_date,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
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
    return pd.concat([log, pd.DataFrame([row])], ignore_index=True), f"Captured snapshot for {prediction_date}."


def _accuracy(series: pd.Series) -> str:
    valid = series.dropna()
    if valid.empty:
        return "-"
    return f"{valid.astype(bool).mean() * 100:.1f}%"


st.title("Live Forward Validation")
st.caption("Daily out-of-sample prediction log. No retraining. No model logic changes.")

try:
    raw_df = download_data()
    df = add_macro_features(add_features(raw_df), None)
except Exception as exc:
    st.error(f"Failed to load market data: {exc}")
    st.stop()

log_df = _load_log()
log_df = _score_open_predictions(log_df, df)
log_df, capture_msg = _append_today_snapshot(log_df, df)
log_df = _score_open_predictions(log_df, df)
_save_log(log_df)

st.info(capture_msg)
st.caption(f"Validation threshold: +/-{DIRECTION_THRESHOLD * 100:.2f}% next-session move")

scored = log_df[log_df["correct"].notna()].copy()
if not scored.empty:
    scored["correct_bool"] = scored["correct"].astype(bool)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Observations", int(len(scored)))
k2.metric("Accuracy", _accuracy(scored["correct"] if not scored.empty else pd.Series(dtype=object)))
k3.metric("Buy Accuracy", _accuracy(scored.loc[scored["signal"].str.upper().isin(["UP", "BUY"]), "correct"] if not scored.empty else pd.Series(dtype=object)))
k4.metric("Sell Accuracy", _accuracy(scored.loc[scored["signal"].str.upper().isin(["DOWN", "SELL"]), "correct"] if not scored.empty else pd.Series(dtype=object)))
k5.metric("Sideways Accuracy", _accuracy(scored.loc[scored["signal"].str.upper().eq("SIDEWAYS"), "correct"] if not scored.empty else pd.Series(dtype=object)))

st.divider()
st.subheader("Validation Table")

display_df = log_df.sort_values("prediction_date", ascending=False).copy()
for col in ["gold_price", "confidence", "prob_down", "prob_sideways", "prob_up", "atr", "vix", "actual_price", "actual_move_pct"]:
    if col in display_df.columns:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(3)
st.dataframe(display_df, hide_index=True, width="stretch")

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
        st.plotly_chart(_dark(fig_acc), width="stretch")

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
        st.plotly_chart(_dark(fig_conf), width="stretch")
