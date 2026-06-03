"""
APEX Metals AI — Risk Management Calculator
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import (
    PRIMARY_TICKER, ATR_STOP_MULTIPLIER, RISK_REWARD_RATIO,
    INITIAL_CAPITAL, RISK_PER_TRADE_PCT,
)
from src.data_loader import download_data
from src.features import add_features
from src.macro_loader import add_macro_features

st.set_page_config(
    page_title="APEX Metals AI — Risk Management",
    page_icon="🛡️",
    layout="wide",
)

DARK_BG  = "#0e1117"

@st.cache_data(ttl=3600, show_spinner="Loading market data…")
def _load(refresh_key: int) -> pd.DataFrame:
    raw = download_data(force_refresh=(refresh_key > 0))
    return add_features(raw)

rk = st.session_state.get("refresh_key", 0)

try:
    raw_df = _load(rk)
    macro_df = st.session_state.get("macro_df")
    if macro_df is None:
        macro_df = pd.DataFrame()
    df = add_macro_features(raw_df, macro_df)
except Exception as exc:
    st.error(f"Failed to load data: {exc}")
    st.stop()

cur     = float(df["Close"].iloc[-1])
atr_val = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else cur * 0.01
last_dt = str(df.index[-1].date())

# ── Resolve current AI signal direction ───────────────────────────────────────
signal = st.session_state.get("signal")
default_dir = 0  # 0=BUY, 1=SELL, 2=NEUTRAL
if signal:
    if signal.get("signal_int") == 2:   default_dir = 0  # UP → BUY
    elif signal.get("signal_int") == 0: default_dir = 1  # DOWN → SELL
    else:                               default_dir = 2

# ══════════════════════════════════════════════════════════════════════════════
st.title("🛡️ Risk Management")
st.caption("⚠️ NOT financial advice · ATR-based sizing · Gold futures reference")
st.divider()

# ── Live price strip ──────────────────────────────────────────────────────────
p1, p2, p3 = st.columns(3)
p1.metric("Gold Price", f"${cur:,.2f}", help=f"Last bar: {last_dt}")
p2.metric("ATR (14-day)", f"${atr_val:.2f}", help="Average True Range — volatility measure")
p3.metric("ATR / Price", f"{atr_val/cur*100:.2f}%")

st.divider()

# ── Inputs ─────────────────────────────────────────────────────────────────────
st.subheader("Position Inputs")
col_a, col_b, col_c = st.columns(3)

with col_a:
    account = st.number_input(
        "Account Size ($)",
        min_value=1_000, max_value=10_000_000,
        value=int(INITIAL_CAPITAL), step=5_000,
        format="%d",
    )

with col_b:
    risk_pct = st.slider(
        "Risk per Trade (%)",
        min_value=0.25, max_value=3.0,
        value=float(RISK_PER_TRADE_PCT * 100),
        step=0.25,
        format="%.2f%%",
    ) / 100

with col_c:
    direction = st.selectbox(
        "Trade Direction",
        options=["BUY (Long)", "SELL (Short)"],
        index=default_dir if default_dir < 2 else 0,
        help="Pre-filled from AI signal if model is trained",
    )

is_long = direction.startswith("BUY")

# ── ATR multiplier override ────────────────────────────────────────────────────
with st.expander("Advanced: adjust stop multiplier"):
    stop_mult = st.slider(
        "Stop Loss Multiplier (× ATR)",
        min_value=1.0, max_value=4.0,
        value=float(ATR_STOP_MULTIPLIER), step=0.25,
    )
    rr_ratio = st.slider(
        "Risk / Reward Ratio (1 : X)",
        min_value=1.0, max_value=5.0,
        value=float(RISK_REWARD_RATIO), step=0.5,
    )

# ── Calculations ──────────────────────────────────────────────────────────────
max_risk_usd  = account * risk_pct
stop_distance = atr_val * stop_mult
tp_distance   = stop_distance * rr_ratio

entry_price   = cur
stop_price    = entry_price - stop_distance if is_long else entry_price + stop_distance
tp_price      = entry_price + tp_distance   if is_long else entry_price - tp_distance
stop_pct      = stop_distance / entry_price * 100
tp_pct        = tp_distance   / entry_price * 100

# Position size in dollars and oz
pos_size_usd  = max_risk_usd / stop_distance * entry_price if stop_distance > 0 else 0
pos_size_oz   = max_risk_usd / stop_distance if stop_distance > 0 else 0
pos_pct_acct  = pos_size_usd / account * 100

max_loss_usd  = max_risk_usd
expected_gain = max_risk_usd * rr_ratio

# ── Output cards ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Trade Setup")

dir_color = "#00CC88" if is_long else "#FF4B4B"
dir_emoji = "📈" if is_long else "📉"

st.markdown(
    f"""<div style="border:2px solid {dir_color};border-radius:10px;
        padding:14px;text-align:center;margin-bottom:16px">
        <span style="font-size:1.5em;font-weight:bold;color:{dir_color}">
            {dir_emoji} {direction}</span>
        <span style="font-size:0.9em;color:#aaa;margin-left:16px">
            Entry: ${entry_price:,.2f}</span>
    </div>""",
    unsafe_allow_html=True,
)

r1, r2, r3, r4 = st.columns(4)
r1.metric(
    "Stop Loss",
    f"${stop_price:,.2f}",
    f"{'-' if is_long else '+'}{stop_pct:.2f}% from entry",
    delta_color="off",
)
r2.metric(
    "Take Profit",
    f"${tp_price:,.2f}",
    f"{'+' if is_long else '-'}{tp_pct:.2f}% from entry",
)
r3.metric(
    "Risk / Reward",
    f"1 : {rr_ratio:.1f}",
    help="Reward multiple per unit risked",
)
r4.metric(
    "Position Size",
    f"{pos_size_oz:.1f} oz",
    f"${pos_size_usd:,.0f} notional",
)

st.divider()
r5, r6, r7, r8 = st.columns(4)
r5.metric("Max Risk ($)",    f"${max_loss_usd:,.2f}",
          f"{risk_pct*100:.2f}% of account")
r6.metric("Expected Gain ($)", f"${expected_gain:,.2f}",
          f"if TP hit at {rr_ratio:.1f}×")
r7.metric("Position / Account", f"{pos_pct_acct:.1f}%",
          help="Notional exposure relative to account size")
r8.metric("Stop Distance",  f"${stop_distance:.2f}",
          f"{stop_mult:.1f}× ATR")

# ── Scenario table ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("Risk Scenarios")

scenarios = []
for mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    r = risk_pct * mult
    risk_usd = account * r
    ps_oz    = risk_usd / stop_distance if stop_distance > 0 else 0
    ps_usd   = ps_oz * entry_price
    scenarios.append({
        "Risk %":        f"{r*100:.2f}%",
        "Max Loss ($)":  f"${risk_usd:,.0f}",
        "Position (oz)": f"{ps_oz:.1f}",
        "Notional ($)":  f"${ps_usd:,.0f}",
        "Expected Gain": f"${risk_usd*rr_ratio:,.0f}",
        "Acct Exposure": f"{ps_usd/account*100:.1f}%",
    })

st.dataframe(
    pd.DataFrame(scenarios),
    hide_index=True,
    width="stretch",
)

st.caption(
    f"Stop = {stop_mult:.1f}× ATR (${atr_val:.2f}) · "
    f"R:R = 1:{rr_ratio:.1f} · Entry = ${entry_price:,.2f} · "
    f"Data date: {last_dt}"
)
