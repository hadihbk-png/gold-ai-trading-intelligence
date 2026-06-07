"""
APEX Metals AI — AI Assistant (Phase A, Stage 1 skeleton)
"""
import os, sys, warnings
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="APEX Metals AI — AI Assistant",
    page_icon="🤖",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🤖 AI Assistant")
st.info(
    "This AI assistant is a decision-intelligence tool - built to help you make sharper, "
    "better-informed decisions by giving you live data, market context, and clear explanations "
    "of the platform's signals. It can make mistakes and may be incomplete or out of date, so "
    "verify anything important. It does not provide financial or investment advice; it informs "
    "your decision, but the choice to buy, sell, or hold remains yours."
)

st.divider()

# ── Metal selector ─────────────────────────────────────────────────────────────
metal = st.radio("Select metal", options=["Gold", "Silver", "Platinum"], horizontal=True)

st.divider()

# ── Resolve signal + regime from session_state ─────────────────────────────────
if metal == "Gold":
    _signal      = st.session_state.get("signal")
    _regime_info = st.session_state.get("regime_info")
elif metal == "Silver":
    _signal      = st.session_state.get("silver_signal")
    _regime_info = None
else:
    _signal      = st.session_state.get("platinum_signal")
    _regime_info = None

# ── Current readings ───────────────────────────────────────────────────────────
st.subheader(f"Current readings — {metal}")

if _signal is None:
    st.info(
        f"Open the {metal} dashboard once to load current data, "
        f"then return here for live-grounded answers."
    )
else:
    _price      = _signal.get("current_price")
    _sig_lbl    = _signal.get("signal_label", "—")
    _conf_pct   = _signal.get("confidence_pct")
    _proba      = _signal.get("proba_vec")      # [p_down, p_sideways, p_up]
    _filter     = _signal.get("filter_reason")
    _atr        = _signal.get("atr")

    # ── Metrics row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price (USD)",   f"${_price:,.2f}" if _price is not None else "—")
    c2.metric("Signal",        _sig_lbl)
    c3.metric("Confidence",    f"{_conf_pct:.1f}%" if _conf_pct is not None else "—")

    if _regime_info:
        c4.metric("Regime", _regime_info.get("regime_label", "—"))
        if _atr is not None:
            st.caption(f"ATR: ${_atr:,.2f}")
    else:
        c4.metric("ATR", f"${_atr:,.2f}" if _atr is not None else "—")

    # ── Probability split ─────────────────────────────────────────────────────
    if _proba is not None and len(_proba) == 3:
        p_down, p_side, p_up = _proba
        st.markdown("**Directional probabilities**")
        p1, p2, p3 = st.columns(3)
        p1.metric("UP",       f"{p_up   * 100:.1f}%")
        p2.metric("SIDEWAYS", f"{p_side * 100:.1f}%")
        p3.metric("DOWN",     f"{p_down * 100:.1f}%")

    # ── No-trade filter ───────────────────────────────────────────────────────
    if _filter:
        st.warning(f"**No-trade filter active:** {_filter}")
