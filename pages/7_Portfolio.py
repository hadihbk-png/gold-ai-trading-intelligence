"""
APEX Metals AI — Portfolio Tracker (Tier 3 · item 1 + persistence)

Enter quantity (troy oz) and average cost (USD/oz) per metal; the page pulls
live spot prices (same feeds as the Dashboard) and FX rates, then shows cost
basis, market value, and unrealized P&L in USD and a chosen second currency.

Holdings persist to a 'portfolio' tab in the SAME Google Sheet used for the
track record — separate from the prediction ledger (the hash chain is never
touched). If the sheet is unavailable, the page falls back to session-only.

READ-ONLY on prices. Additive — does NOT touch models, thresholds, signals,
risk, the verdict gate, or the prediction ledger.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import (
    get_live_spot_price,
    get_silver_price,
    get_platinum_price,
    get_fx_rates,
)

st.set_page_config(page_title="APEX — Portfolio", page_icon="📊", layout="wide")

METALS = ["Gold", "Silver", "Platinum"]

# ── Persistence: a 'portfolio' tab in the track-record Google Sheet ───────────
_PORTFOLIO_WS = "portfolio"
_HEADERS = ["metal", "qty_oz", "entry_usd", "updated_utc"]


def _portfolio_worksheet():
    """Open (or create) the 'portfolio' tab in the track-record sheet."""
    import gspread
    try:
        from gspread.exceptions import WorksheetNotFound
    except Exception:
        WorksheetNotFound = Exception

    sa_info = dict(st.secrets["gcp_service_account"])
    tr = st.secrets["track_record"]
    gc = gspread.service_account_from_dict(sa_info)
    sh = gc.open_by_key(tr["sheet_id"])
    try:
        ws = sh.worksheet(_PORTFOLIO_WS)
    except WorksheetNotFound:
        ws = sh.add_worksheet(title=_PORTFOLIO_WS, rows=20, cols=len(_HEADERS))
        ws.append_rows([_HEADERS], value_input_option="RAW")
    return ws


def load_holdings() -> tuple:
    """Return ({metal: {'qty_oz', 'entry_usd'}}, available: bool)."""
    try:
        ws = _portfolio_worksheet()
        out = {}
        for r in ws.get_all_records():
            m = str(r.get("metal", "")).strip()
            if m in METALS:
                out[m] = {
                    "qty_oz": float(r.get("qty_oz", 0) or 0),
                    "entry_usd": float(r.get("entry_usd", 0) or 0),
                }
        return out, True
    except Exception:
        return {}, False


def save_holdings(holdings: dict) -> bool:
    """Overwrite the 'portfolio' tab with current holdings. True on success."""
    try:
        ws = _portfolio_worksheet()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [_HEADERS]
        for m in METALS:
            v = holdings.get(m, {})
            rows.append([m, float(v.get("qty_oz", 0)), float(v.get("entry_usd", 0)), now])
        ws.clear()
        ws.append_rows(rows, value_input_option="RAW")
        return True
    except Exception:
        return False


# ── API keys for live prices ─────────────────────────────────────────────────
def _secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


_TD_KEY = _secret("TWELVE_DATA_API_KEY")
_AV_KEY = _secret("ALPHA_VANTAGE_API_KEY")


@st.cache_data(ttl=60, show_spinner=False)
def _price_gold() -> tuple:
    return get_live_spot_price(_TD_KEY, _AV_KEY)


@st.cache_data(ttl=60, show_spinner=False)
def _price_silver() -> tuple:
    return get_silver_price(_AV_KEY)


@st.cache_data(ttl=60, show_spinner=False)
def _price_platinum() -> tuple:
    return get_platinum_price(_TD_KEY, _AV_KEY)


@st.cache_data(ttl=300, show_spinner=False)
def _fx() -> dict:
    return get_fx_rates()


def _live_prices() -> dict:
    return {"Gold": _price_gold(), "Silver": _price_silver(), "Platinum": _price_platinum()}


# ── Load persisted holdings once per session ─────────────────────────────────
for m in METALS:
    st.session_state.setdefault(f"qty_{m}", 0.0)
    st.session_state.setdefault(f"entry_{m}", 0.0)

if "_portfolio_loaded" not in st.session_state:
    _loaded, _available = load_holdings()
    for m in METALS:
        if m in _loaded:
            st.session_state[f"qty_{m}"] = _loaded[m]["qty_oz"]
            st.session_state[f"entry_{m}"] = _loaded[m]["entry_usd"]
    st.session_state["_portfolio_loaded"] = True
    st.session_state["_portfolio_persist"] = _available

_persist = st.session_state.get("_portfolio_persist", False)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📊 Portfolio Tracker")
st.caption("Personal research only · NOT financial advice · Live prices are delayed / indicative")

if _persist:
    st.success("Holdings persistence is ON — saved to your Google Sheet ('portfolio' tab).", icon="✅")
else:
    st.info(
        "Holdings are session-only right now (Google Sheet not reachable). Your P&L still "
        "works; entries clear on a full reload.", icon="ℹ️"
    )

# ── Currency ─────────────────────────────────────────────────────────────────
_fx_data = _fx()
_rates = _fx_data.get("rates", {"USD": 1.0, "AED": 3.6725})
_ccy_options = list(_rates.keys())
_default_idx = _ccy_options.index("AED") if "AED" in _ccy_options else 0
_ccy = st.selectbox("Second display currency (USD is always shown)", _ccy_options, index=_default_idx)
_rate = float(_rates.get(_ccy, 1.0))

# ── Holdings input ───────────────────────────────────────────────────────────
st.subheader("Your holdings")
st.caption("Quantity in troy ounces · average cost in USD per troy ounce.")

cols = st.columns(3)
for i, m in enumerate(METALS):
    with cols[i]:
        st.markdown(f"**{m}**")
        st.number_input(f"{m} — quantity (oz)", min_value=0.0, step=0.1, key=f"qty_{m}")
        st.number_input(f"{m} — avg cost (USD/oz)", min_value=0.0, step=1.0, key=f"entry_{m}")

# ── Save / Reload controls ───────────────────────────────────────────────────
b1, b2, _sp = st.columns([1, 1, 3])
with b1:
    if st.button("💾 Save holdings", use_container_width=True, disabled=not _persist):
        _h = {m: {"qty_oz": st.session_state[f"qty_{m}"],
                  "entry_usd": st.session_state[f"entry_{m}"]} for m in METALS}
        if save_holdings(_h):
            st.session_state["_portfolio_saved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            st.toast("Holdings saved to Google Sheet.", icon="✅")
        else:
            st.error("Could not save to the sheet — please try again.")
with b2:
    if st.button("↻ Reload from sheet", use_container_width=True, disabled=not _persist):
        _loaded, _available = load_holdings()
        for m in METALS:
            st.session_state[f"qty_{m}"] = _loaded.get(m, {}).get("qty_oz", 0.0)
            st.session_state[f"entry_{m}"] = _loaded.get(m, {}).get("entry_usd", 0.0)
        st.session_state["_portfolio_persist"] = _available
        st.rerun()

if st.session_state.get("_portfolio_saved_at"):
    st.caption(f"Last saved: {st.session_state['_portfolio_saved_at']}")

# ── Compute ──────────────────────────────────────────────────────────────────
prices = _live_prices()
rows = []
tot_cost = 0.0
tot_mv = 0.0
missing_price = False

for m in METALS:
    qty = float(st.session_state[f"qty_{m}"])
    entry = float(st.session_state[f"entry_{m}"])
    if qty <= 0:
        continue
    price, source = prices[m]
    cost = qty * entry
    tot_cost += cost
    if price:
        mv = qty * price
        tot_mv += mv
        pnl = mv - cost
        pnl_pct = (pnl / cost * 100) if cost > 0 else float("nan")
    else:
        mv = pnl = pnl_pct = float("nan")
        missing_price = True
    rows.append({
        "Metal": m,
        "Qty (oz)": round(qty, 4),
        "Avg cost $/oz": round(entry, 2),
        "Live $/oz": round(price, 2) if price else float("nan"),
        "Cost $": round(cost, 2),
        "Value $": round(mv, 2) if price else float("nan"),
        "P&L $": round(pnl, 2) if price else float("nan"),
        "P&L %": round(pnl_pct, 2) if price else float("nan"),
        "Source": source,
    })

# ── Display ──────────────────────────────────────────────────────────────────
st.subheader("Positions")
if not rows:
    st.info("Enter a quantity and average cost above to see your live P&L.")
else:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    tot_pnl = tot_mv - tot_cost
    tot_pnl_pct = (tot_pnl / tot_cost * 100) if tot_cost > 0 else 0.0

    st.markdown("### Portfolio totals")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cost basis (USD)", f"${tot_cost:,.2f}")
    c2.metric("Market value (USD)", f"${tot_mv:,.2f}")
    c3.metric("Unrealized P&L (USD)", f"${tot_pnl:,.2f}", delta=f"{tot_pnl_pct:+.2f}%")

    st.markdown(
        f"**In {_ccy}:**  cost {_ccy} {tot_cost*_rate:,.2f}  ·  "
        f"value {_ccy} {tot_mv*_rate:,.2f}  ·  "
        f"P&L {_ccy} {tot_pnl*_rate:,.2f} ({tot_pnl_pct:+.2f}%)"
    )
    if missing_price:
        st.warning(
            "One or more live prices were unavailable just now; those rows show "
            "blanks and are left out of the totals. Try again shortly."
        )

st.caption(
    f"FX fetched {_fx_data.get('fetched_utc', '—')} · USD→{_ccy} = {_rate:.4f} · "
    f"prices cached ~60s · rendered {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
)
