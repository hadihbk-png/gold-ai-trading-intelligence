"""
APEX Metals AI — Multi-Metal Comparison Panel (Tier 3 · item 2).

Shows Gold, Silver and Platinum side by side in one comparison table: price,
daily change, signal + confidence, directional probabilities, regime, and the
TRADE / CAUTION / NO-TRADE verdict — so the whole precious-metals picture reads
at a glance instead of toggling one metal at a time.

Prices use the SAME live feeds as the Dashboard (Gold via Twelve Data, Silver
via Alpha Vantage, Platinum via Swissquote), so the price row and Gold/Silver
ratio match the Dashboard to the cent. Signals, probabilities and verdicts come
from the same trained models (via src/compare.py).

Additive and READ-ONLY — does not touch models, thresholds, the risk engine,
the verdict gate, or the prediction ledger.
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.compare import build_comparison

st.set_page_config(page_title="APEX — Compare", page_icon="⚖️", layout="wide")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# ⚖️ Multi-Metal Comparison")
st.caption(
    "Gold · Silver · Platinum side by side · NOT financial advice · "
    "prices = live spot (USD), same feeds as the Dashboard"
)

# Same API keys the Dashboard uses, so the live price feeds match exactly.
_TD_KEY = st.secrets.get("TWELVE_DATA_API_KEY", os.environ.get("TWELVE_DATA_API_KEY", ""))
_AV_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")


@st.cache_data(ttl=120, show_spinner="Comparing Gold, Silver & Platinum…")
def _get_comparison(td_key: str, av_key: str):
    return build_comparison(td_key, av_key)


col_a, col_b = st.columns([1, 5])
with col_a:
    if st.button("🔄 Refresh", width="stretch"):
        _get_comparison.clear()
        st.rerun()

metals = _get_comparison(_TD_KEY, _AV_KEY)
ok_metals = [m for m in metals if m.get("ok")]

if not ok_metals:
    st.error(
        "Could not load any metal right now. The models may not be trained yet, "
        "or the data feed is temporarily unavailable. Open the Dashboard to load "
        "the models, then come back."
    )
    st.stop()

# ── Helpers ──────────────────────────────────────────────────────────────────
GREEN, RED, GREY, MUTED = "#00CC88", "#FF4B4B", "#888888", "#aaaaaa"


def _fmt_price(m):
    return f"${m['price']:,.2f}"


def _fmt_change(m):
    c = m["change_pct"]
    color = GREEN if c >= 0 else RED
    arrow = "▲" if c >= 0 else "▼"
    return f"<span style='color:{color}'>{arrow} {c:+.2f}%</span>"


def _fmt_source(m):
    return f"<span style='color:{MUTED};font-size:0.82em'>📡 {m.get('price_source', '—')}</span>"


def _fmt_signal(m):
    return (f"<span style='color:{m['signal_color']};font-weight:800'>"
            f"{m['signal']}</span>")


def _fmt_conf(m):
    c = m.get("confidence_pct")
    return f"{c:.1f}%" if c is not None else "—"


def _fmt_prob(val, color):
    return (f"<span style='color:{color}'>{val:.1f}%</span>"
            if val is not None else "—")


def _fmt_verdict(m):
    return (f"<span style='color:{m['verdict_color']};font-weight:800'>"
            f"{m['verdict_emoji']} {m['verdict']}</span>")


def _fmt_filter(m):
    fr = m.get("filter_reason")
    if not fr:
        return f"<span style='color:{GREEN}'>none</span>"
    return f"<span style='color:#f59e0b;font-size:0.85em'>{fr}</span>"


# Rows: (label, cell-renderer). Each renderer takes a metal dict → HTML string.
ROWS = [
    ("Price (live spot)", _fmt_price),
    ("Daily change", _fmt_change),
    ("Price source", _fmt_source),
    ("Signal (next day)", _fmt_signal),
    ("Confidence", _fmt_conf),
    ("Prob ▲ UP", lambda m: _fmt_prob(m.get("p_up"), GREEN)),
    ("Prob ◼ SIDEWAYS", lambda m: _fmt_prob(m.get("p_side"), GREY)),
    ("Prob ▼ DOWN", lambda m: _fmt_prob(m.get("p_down"), RED)),
    ("Market regime", lambda m: m.get("regime", "—")),
    ("Verdict", _fmt_verdict),
    ("Active filter", _fmt_filter),
]

# ── Build the comparison table (HTML for per-cell colour) ─────────────────────
_header_cells = "".join(
    f"<th style='padding:10px 16px;text-align:center;font-size:1.05em'>"
    f"{m['name']}<div style='color:{MUTED};font-weight:400;font-size:0.8em'>"
    f"{m['symbol']}</div></th>"
    for m in metals
)

_body_rows = []
for label, render in ROWS:
    cells = []
    for m in metals:
        if m.get("ok"):
            cells.append(f"<td style='padding:9px 16px;text-align:center'>{render(m)}</td>")
        else:
            cells.append(f"<td style='padding:9px 16px;text-align:center;color:{MUTED}'>—</td>")
    _body_rows.append(
        f"<tr style='border-top:1px solid #222'>"
        f"<td style='padding:9px 16px;color:{MUTED};font-weight:600;white-space:nowrap'>{label}</td>"
        + "".join(cells) + "</tr>"
    )

_table = (
    "<table style='width:100%;border-collapse:collapse;"
    "background:#0d1218;border-radius:12px;overflow:hidden'>"
    f"<thead><tr style='background:#121821'>"
    f"<th style='padding:10px 16px;text-align:left;color:{MUTED}'>Metric</th>"
    f"{_header_cells}</tr></thead>"
    f"<tbody>{''.join(_body_rows)}</tbody></table>"
)
st.markdown(_table, unsafe_allow_html=True)

# Note any metal that failed to load.
_failed = [m for m in metals if not m.get("ok")]
for m in _failed:
    st.caption(f"⚠️ {m['name']} unavailable: {m.get('error', 'unknown')}")

st.divider()

# ── Cross-metal reads ────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown("##### Gold / Silver Ratio")
    g = next((m for m in ok_metals if m["name"] == "Gold"), None)
    s = next((m for m in ok_metals if m["name"] == "Silver"), None)
    # Parity with the Dashboard: gold FUTURES last close ÷ silver LIVE price.
    if g and s and s.get("price"):
        gsr = g["last_close"] / s["price"]
        _tag = ("silver historically cheap" if gsr > 80
                else "gold historically cheap" if gsr < 50
                else "mid-range")
        st.markdown(
            f"<div style='font-size:2.2em;font-weight:800'>{gsr:,.1f}</div>"
            f"<div style='color:{MUTED}'>{_tag} · GSR &gt; 80 = silver cheap · "
            f"GSR &lt; 50 = gold cheap</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("Needs both Gold and Silver loaded.")

with c2:
    st.markdown("##### Strongest Conviction")
    # Highest-confidence metal with a directional (non-SIDEWAYS) signal.
    directional = [m for m in ok_metals if m["signal"] in ("UP", "DOWN")
                   and m.get("confidence_pct") is not None]
    if directional:
        top = max(directional, key=lambda m: m["confidence_pct"])
        st.markdown(
            f"<div style='font-size:1.6em;font-weight:800'>"
            f"{top['name']} · <span style='color:{top['signal_color']}'>"
            f"{top['signal']}</span></div>"
            f"<div style='color:{MUTED}'>{top['confidence_pct']:.1f}% confidence · "
            f"verdict {top['verdict']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='color:{MUTED}'>No directional signal right now — "
            f"all three are SIDEWAYS. The confidence / no-trade / consensus gates "
            f"are doing their job on a choppy tape.</div>",
            unsafe_allow_html=True,
        )

st.divider()
_asof = next((m["as_of"] for m in ok_metals if m.get("as_of")), "—")
st.caption(
    f"Prices are live spot (same feeds as the Dashboard). Signals, probabilities "
    f"and verdicts are computed on the latest completed daily close "
    f"(as of {_asof}) from the same trained models as the Dashboard · "
    f"research framework only, not financial advice."
)
