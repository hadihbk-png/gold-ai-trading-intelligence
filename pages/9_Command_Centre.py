"""
APEX Metals AI — KPI Command Centre (Tier 3 · item 3).

A single one-glance cockpit pulling the platform's headline numbers into four
grouped sections:
  1. Signals        — verdict + signal + confidence for Gold/Silver/Platinum
  2. Track record   — logged / scored / pending, hit-rate, SHA-256 chain status
  3. Model & data   — rolling accuracy per model, gold feature-drift, live feeds
  4. Portfolio      — market value + unrealized P&L (if holdings are saved)

Every number is READ from the same functions/stores the rest of the app uses
(src/compare.py, the track-record Google Sheet, detect_feature_drift, the
portfolio tab). Additive and read-only — touches no models, thresholds, the
verdict gate, or the prediction ledger.
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.compare import build_comparison

st.set_page_config(page_title="APEX — Command Centre", page_icon="🛰️", layout="wide")

GREEN, RED, AMBER, GREY, MUTED = "#00CC88", "#FF4B4B", "#f59e0b", "#888888", "#aaaaaa"

_TD_KEY = st.secrets.get("TWELVE_DATA_API_KEY", os.environ.get("TWELVE_DATA_API_KEY", ""))
_AV_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")


# ── Data builders (cached) ───────────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner="Loading the cockpit…")
def _signals(td_key, av_key):
    return build_comparison(td_key, av_key)


@st.cache_data(ttl=120)
def _track_health():
    """Logged / scored / pending counts, hit-rate, chain status from the store."""
    try:
        import pandas as pd
        from src.track_store_factory import make_sheets_store_from_secrets, dedupe_latest
        from src.track_logger import verify_chain
        store = make_sheets_store_from_secrets()
        rows = store.read_all()
        chain_ok = bool(verify_chain(rows))
        n_total = len(rows)
        if n_total == 0:
            return {"ok": True, "n_total": 0, "n_unique": 0, "n_scored": 0,
                    "n_pending": 0, "hit_rate": None, "chain_ok": chain_ok}
        n_unique = len(dedupe_latest(rows))
        df = pd.DataFrame(rows)
        if "timestamp_utc" in df.columns:
            dd = (df.sort_values("timestamp_utc", na_position="first")
                    .drop_duplicates(subset=["metal", "as_of_date"], keep="last"))
        else:
            dd = df
        scored = dd[dd["scored_at"].notna()] if "scored_at" in dd.columns else dd.iloc[0:0]
        n_scored = len(scored)
        n_pending = n_unique - n_scored
        hit_rate = None
        if "hit" in scored.columns and n_scored > 0:
            hits = pd.to_numeric(scored["hit"].map({True: 1, False: 0}),
                                 errors="coerce").dropna()
            if len(hits) > 0:
                hit_rate = float(hits.mean())
        return {"ok": True, "n_total": n_total, "n_unique": n_unique,
                "n_scored": n_scored, "n_pending": n_pending,
                "hit_rate": hit_rate, "chain_ok": chain_ok}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@st.cache_data(ttl=600)
def _gold_drift():
    """Gold feature-drift score vs training stats (same as the Dashboard)."""
    try:
        import json
        from src.config import MODELS_DIR
        from src.features import detect_feature_drift
        from src.compare import _load_bundle_and_df
        df, bundle = _load_bundle_and_df("gold")
        ts_path = os.path.join(MODELS_DIR, "training_stats.json")
        if df is None or bundle is None or not os.path.exists(ts_path):
            return {"ok": False}
        with open(ts_path) as f:
            stats = json.load(f)
        feat_cols = bundle.get("feature_cols") or []
        live = {c: float(df[c].iloc[-1]) for c in feat_cols if c in df.columns}
        score, drifted = detect_feature_drift(live, stats)
        label = "Normal" if score < 0.2 else "Monitor" if score < 0.5 else "High Drift"
        return {"ok": True, "score": float(score), "label": label,
                "n_drifted": len(drifted)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _portfolio_totals(price_by_metal: dict):
    """Market value + P&L from the 'portfolio' tab, priced at live spot."""
    try:
        import gspread
        sa_info = dict(st.secrets["gcp_service_account"])
        tr = st.secrets["track_record"]
        gc = gspread.service_account_from_dict(sa_info)
        sh = gc.open_by_key(tr["sheet_id"])
        try:
            ws = sh.worksheet("portfolio")
        except Exception:
            return {"ok": True, "has_holdings": False}
        tot_cost = tot_mv = 0.0
        has = False
        priced_all = True
        for r in ws.get_all_records():
            metal = str(r.get("metal", "")).strip()
            qty = float(r.get("qty_oz", 0) or 0)
            entry = float(r.get("entry_usd", 0) or 0)
            if qty <= 0:
                continue
            has = True
            tot_cost += qty * entry
            price = price_by_metal.get(metal) or price_by_metal.get(metal.capitalize())
            if price:
                tot_mv += qty * price
            else:
                priced_all = False
        if not has:
            return {"ok": True, "has_holdings": False}
        pnl = tot_mv - tot_cost
        pnl_pct = (pnl / tot_cost * 100) if tot_cost > 0 else 0.0
        return {"ok": True, "has_holdings": True, "cost": tot_cost, "mv": tot_mv,
                "pnl": pnl, "pnl_pct": pnl_pct, "priced_all": priced_all}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🛰️ KPI Command Centre")
st.caption("One-glance platform status · NOT financial advice")

if st.button("🔄 Refresh"):
    _signals.clear()
    _track_health.clear()
    _gold_drift.clear()
    st.rerun()

metals = _signals(_TD_KEY, _AV_KEY)
ok_metals = [m for m in metals if m.get("ok")]
price_by_metal = {m["name"]: m.get("price") for m in ok_metals}

# ── Section 1: Signals ───────────────────────────────────────────────────────
st.subheader("📡 Signals — Gold · Silver · Platinum")
if ok_metals:
    cols = st.columns(len(metals))
    for col, m in zip(cols, metals):
        with col:
            if not m.get("ok"):
                st.markdown(f"**{m['name']}**")
                st.caption(f"unavailable: {m.get('error', 'unknown')}")
                continue
            conf = m.get("confidence_pct")
            st.markdown(
                f"<div style='border:1px solid #222;border-radius:12px;padding:14px 16px;"
                f"background:#0d1218'>"
                f"<div style='color:{MUTED};font-size:0.85em'>{m['name']} · {m['symbol']}</div>"
                f"<div style='font-size:1.5em;font-weight:800;color:{m['verdict_color']};"
                f"margin:2px 0'>{m['verdict_emoji']} {m['verdict']}</div>"
                f"<div style='font-size:1.05em'>Signal: "
                f"<span style='color:{m['signal_color']};font-weight:700'>{m['signal']}</span></div>"
                f"<div style='color:{MUTED}'>Confidence: "
                f"{(f'{conf:.1f}%' if conf is not None else '—')}</div>"
                f"<div style='color:{MUTED};font-size:0.9em'>${m['price']:,.2f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
else:
    st.error("Signals unavailable — open the Dashboard to load the models, then refresh.")

st.divider()

# ── Section 2: Track record health ───────────────────────────────────────────
st.subheader("🔐 Track Record Health")
th = _track_health()
if th.get("ok"):
    if th["chain_ok"]:
        st.markdown(f"<span style='color:{GREEN};font-weight:700'>✅ Chain verified</span> "
                    f"<span style='color:{MUTED}'>— SHA-256 integrity intact</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:{RED};font-weight:800'>⛔ CHAIN BROKEN</span> "
                    f"<span style='color:{MUTED}'>— integrity check failed</span>",
                    unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Logged", th["n_unique"])
    k2.metric("Scored", th["n_scored"])
    k3.metric("Pending", th["n_pending"])
    hr = th["hit_rate"]
    k4.metric("Hit-rate", f"{hr*100:.0f}%" if hr is not None else "—",
              help="Share of scored predictions that were correct. '—' until outcomes are scored.")
    if th["n_scored"] == 0:
        st.caption("No outcomes scored yet — hit-rate appears after the first scoring pass.")
else:
    st.warning(f"Track-record store unavailable: {th.get('error', 'unknown')}")

st.divider()

# ── Section 3: Model form & data integrity ───────────────────────────────────
st.subheader("🧠 Model Form & Data Integrity")
mc1, mc2, mc3 = st.columns(3)
_model_cols = {"Gold": mc1, "Silver": mc2, "Platinum": mc3}
for m in metals:
    col = _model_cols.get(m["name"])
    if col is None:
        continue
    if m.get("ok"):
        ra = m.get("rolling_accuracy")
        col.metric(f"{m['name']} — model form",
                   f"{ra*100:.0f}%" if ra is not None else "—",
                   help="30-day rolling accuracy on the held-out test set. "
                        "~33% is the random baseline for 3-class.")
    else:
        col.metric(f"{m['name']} — model form", "—")

d1, d2, d3 = st.columns(3)
# Gold feature drift
gd = _gold_drift()
if gd.get("ok"):
    _dc = GREEN if gd["label"] == "Normal" else AMBER if gd["label"] == "Monitor" else RED
    d1.markdown(
        f"<div style='color:{MUTED};font-size:0.85em'>Gold feature drift</div>"
        f"<div style='font-size:1.4em;font-weight:800;color:{_dc}'>{gd['label']}</div>"
        f"<div style='color:{MUTED};font-size:0.85em'>score {gd['score']:.2f} · "
        f"{gd['n_drifted']} features z&gt;3</div>",
        unsafe_allow_html=True)
else:
    d1.markdown(f"<div style='color:{MUTED};font-size:0.85em'>Gold feature drift</div>"
                f"<div style='font-size:1.2em'>—</div>", unsafe_allow_html=True)

# Live feeds status
_live_ok = sum(1 for m in ok_metals if m.get("price_source") and "last close" not in str(m.get("price_source", "")).lower())
d2.metric("Live price feeds", f"{_live_ok}/{len(metals)}",
          help="Metals currently served by a live feed (vs last-close fallback).")

# Last data date
_asof = next((m.get("as_of") for m in ok_metals if m.get("as_of")), "—")
d3.metric("Model data as of", _asof)

# Per-feed detail line
if ok_metals:
    _feeds = " · ".join(f"{m['name']}: {m.get('price_source', '—')}" for m in ok_metals)
    st.caption(f"📡 {_feeds}")

st.divider()

# ── Section 4: Portfolio summary ─────────────────────────────────────────────
st.subheader("💼 Portfolio Summary")
pf = _portfolio_totals(price_by_metal)
if not pf.get("ok"):
    st.caption(f"Portfolio unavailable: {pf.get('error', 'unknown')}")
elif not pf.get("has_holdings"):
    st.caption("No holdings saved. Add positions on the Portfolio page to see value & P&L here.")
else:
    p1, p2, p3 = st.columns(3)
    p1.metric("Cost basis (USD)", f"${pf['cost']:,.2f}")
    p2.metric("Market value (USD)", f"${pf['mv']:,.2f}")
    p3.metric("Unrealized P&L (USD)", f"${pf['pnl']:,.2f}", delta=f"{pf['pnl_pct']:+.2f}%")
    if not pf.get("priced_all", True):
        st.caption("⚠️ Some holdings couldn't be priced live — totals exclude those.")

st.divider()
st.caption(
    "Prices are live spot; signals, model form and verdicts are computed on the latest "
    "completed daily close from the same trained models as the Dashboard. "
    "Research framework only — not financial advice."
)
