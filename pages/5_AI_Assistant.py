"""
APEX Metals AI — AI Assistant (Phase A, Stage 2: grounded Claude chat)
"""
import os, sys, warnings
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.explainer import _ANTHROPIC_AVAILABLE, _MODEL
_anthropic = None
if _ANTHROPIC_AVAILABLE:
    from src.explainer import _anthropic


def _md_safe(text: str) -> str:
    return text.replace("$", "\\$")


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

# ── Claude availability check ──────────────────────────────────────────────────
_api_key      = st.secrets.get("ANTHROPIC_API_KEY", "")
_chat_enabled = _ANTHROPIC_AVAILABLE and bool(_api_key)

if not _ANTHROPIC_AVAILABLE:
    st.warning("The `anthropic` package is not installed — chat is unavailable.")
elif not _api_key:
    st.info("Configure `ANTHROPIC_API_KEY` in secrets to enable the AI assistant.")

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

# Extract signal fields once — used by both the readings block and the chat block
_price    = _signal.get("current_price")  if _signal else None
_sig_lbl  = _signal.get("signal_label", "—") if _signal else "—"
_conf_pct = _signal.get("confidence_pct") if _signal else None
_proba    = _signal.get("proba_vec")      if _signal else None   # [p_down, p_sideways, p_up]
_filter   = _signal.get("filter_reason")  if _signal else None
_atr      = _signal.get("atr")            if _signal else None

# ── Current readings ───────────────────────────────────────────────────────────
st.subheader(f"Current readings — {metal}")

if _signal is None:
    st.info(
        f"Open the {metal} dashboard once to load current data, "
        f"then return here for live-grounded answers."
    )
else:
    # ── Metrics row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price (USD)", f"${_price:,.2f}" if _price is not None else "—")
    c2.metric("Signal",      _sig_lbl)
    c3.metric("Confidence",  f"{_conf_pct:.1f}%" if _conf_pct is not None else "—")

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

# ── Chat ───────────────────────────────────────────────────────────────────────
if _chat_enabled and _signal is not None:
    st.divider()
    st.subheader("Ask the assistant")

    # Per-metal history — separate thread per metal, persists across reruns
    _chats   = st.session_state.setdefault("assistant_chat", {})
    _history = _chats.setdefault(metal, [])

    if _history:
        if st.button("Clear conversation", key=f"clear_chat_{metal}"):
            _chats[metal] = []
            st.rerun()

    # Grounded context: only variables already resolved above — no invented numbers
    _ctx_lines = [f"Metal: {metal}"]
    if _price is not None:
        _ctx_lines.append(f"Current price (USD): ${_price:,.2f}")
    _ctx_lines.append(f"AI signal: {_sig_lbl}")
    if _conf_pct is not None:
        _ctx_lines.append(
            f"Model confidence: {_conf_pct:.1f}% "
            f"— the probability of the model's directional read (the UP or DOWN class it leans toward), "
            f"not the displayed signal class; a no-trade filter can force the displayed signal to SIDEWAYS "
            f"while confidence still reflects that directional read"
        )
    if _proba is not None and len(_proba) == 3:
        _p_down, _p_side, _p_up = _proba
        _ctx_lines.append(
            f"Directional probabilities — UP: {_p_up * 100:.1f}%, "
            f"SIDEWAYS: {_p_side * 100:.1f}%, DOWN: {_p_down * 100:.1f}%"
        )
    if _regime_info:
        _ctx_lines.append(f"Market regime: {_regime_info.get('regime_label', '—')}")
    if _atr is not None:
        _ctx_lines.append(f"ATR: ${_atr:,.2f}")
    if _filter:
        _ctx_lines.append(f"No-trade filter active: {_filter}")
    _context_block = "\n".join(_ctx_lines)

    _system_prompt = (
        "You are a decision-intelligence assistant embedded in the APEX Metals AI platform, "
        "a quantitative precious-metals trading research tool.\n\n"
        "STRICT RULES — follow without exception:\n"
        "1. You do NOT give financial or investment advice. Never recommend buying, selling, "
        "or holding any asset, and never suggest trade sizes or entry/exit timing.\n"
        "2. You ground EVERY factual answer exclusively in the live market context block "
        "provided below. Do NOT invent, estimate, or extrapolate any number not present there.\n"
        "3. You cover Gold, Silver, and Platinum only — the three metals tracked by this platform. "
        "You do NOT cover other metals or commodities (e.g. palladium, copper, oil, indices). "
        "Decline questions outside this scope politely and briefly.\n"
        "4. You may explain what signals, regimes, indicators, and probabilities mean, "
        "and what the user should be aware of — but the decision to act remains theirs alone.\n"
        "5. Maintain a professional, measured tone appropriate for a financial decision-intelligence "
        "tool. Use emojis sparingly or not at all.\n"
        "6. When comparing probabilities or any figures, be numerically precise. Do not describe "
        "values as 'close', 'similar', or 'near' unless they genuinely differ by less than 2 "
        "percentage points.\n\n"
        f"LIVE MARKET CONTEXT:\n{_context_block}"
    )

    # Render prior turns
    for _turn in _history:
        with st.chat_message(_turn["role"]):
            st.markdown(_md_safe(_turn["content"]))

    # Accept new input and call Claude
    _user_input = st.chat_input(f"Ask about {metal}…")
    if _user_input:
        _history.append({"role": "user", "content": _user_input})
        with st.chat_message("user"):
            st.markdown(_md_safe(_user_input))

        with st.chat_message("assistant"):
            _placeholder = st.empty()
            _reply       = None
            _err_msg     = None
            _accumulated = ""
            try:
                _client = _anthropic.Anthropic(api_key=_api_key)
                with _client.messages.stream(
                    model=_MODEL,
                    max_tokens=1024,
                    system=[
                        {
                            "type": "text",
                            "text": _system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=_history,
                ) as _stream:
                    for _chunk in _stream.text_stream:
                        _accumulated += _chunk
                        _placeholder.markdown(_md_safe(_accumulated))
                _reply = _accumulated.strip()
            except _anthropic.AuthenticationError:
                _err_msg = "Authentication failed — the API key may be invalid or missing. Check your secrets configuration."
            except _anthropic.RateLimitError:
                _err_msg = "Rate limit reached — please wait a moment and try again."
            except _anthropic.APIConnectionError:
                _err_msg = "Could not reach the Anthropic service — check your connection and try again."
            except _anthropic.APIError:
                _err_msg = "The service returned an error — please try again."
            except Exception:
                _err_msg = "Something went wrong — please try again."

        if _reply is not None:
            _history.append({"role": "assistant", "content": _reply})
        else:
            _placeholder.empty()
            st.error(_err_msg)
            if _history and _history[-1]["role"] == "user":
                _history.pop()
