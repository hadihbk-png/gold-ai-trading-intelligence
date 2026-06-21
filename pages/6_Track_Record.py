"""
APEX Metals AI — Track Record (Item #8)

READ-ONLY view of the SHA-256 chain-linked prediction ledger. Displays
outcomes as scored by Item #7 without recomputing them.

Scores raw_signal (model output, pre-filter) from the chain store.
Differs by design from 4_Live_Validation, which scores displayed_signal
from a separate CSV store.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import DATA_DIR
from src.signals import SIGNAL_LABELS, SIGNAL_COLORS
from src.track_logger import LocalJsonStore, verify_chain
from src.track_stats import wilson_ci

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="APEX Metals AI — Track Record",
    page_icon="🔐",
    layout="wide",
)

DARK_BG   = "#0e1117"
GRID_CLR  = "#1e2130"
_PLT      = dict(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font=dict(color="white"))
MIN_N     = 30
METALS    = ["gold", "silver", "platinum"]
METAL_CLR = {"gold": "#FFD700", "silver": "#C0C0C0", "platinum": "#E5E4E2"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dark(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(**_PLT, height=height, legend=dict(orientation="h", y=1.02))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
    return fig



def _fmt_sig(v) -> str:
    """Format a signal int (0/1/2) as the class label, or '—' for null."""
    try:
        i = int(float(v))
        return SIGNAL_LABELS.get(i, str(i))
    except (TypeError, ValueError):
        return "—"


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading track record…")
def _load_records() -> tuple[list[dict], bool, str]:
    """Return (rows, chain_ok, source_label). READ-ONLY: never writes."""
    source = "Local JSONL"
    try:
        from src.track_store_factory import make_sheets_store_from_secrets
        store  = make_sheets_store_from_secrets()
        source = "Google Sheets"
    except Exception:
        path  = os.path.join(DATA_DIR, "track_record.jsonl")
        store = LocalJsonStore(path)
    rows = store.read_all()
    return rows, verify_chain(rows), source


# ── Page header ────────────────────────────────────────────────────────────────

st.title("🔐 Track Record")
st.caption(
    "Predictions logged **before** outcomes were known · SHA-256 chain-linked · "
    "auditable, not editable.  "
    "Scores **raw\\_signal** (model output, pre-filter) from the chain store — "
    "differs by design from *4 Live Validation*, which scores displayed\\_signal "
    "from a separate CSV store."
)

rows, chain_ok, store_label = _load_records()
n_total = len(rows)

# ── Section 1: Integrity badge ─────────────────────────────────────────────────

if chain_ok:
    st.success(
        f"**{n_total} prediction{'s' if n_total != 1 else ''} · Chain verified** — "
        "SHA-256 integrity check passed. No record has been altered since logging."
    )
else:
    st.error(
        "**CHAIN BROKEN** — SHA-256 integrity check FAILED. "
        "One or more records may have been tampered with. "
        "Do not rely on these statistics."
    )
st.caption(f"Store: {store_label}  ·  Cache refreshes every 5 min")

if n_total == 0:
    st.info(
        "No predictions logged yet. Predictions are captured automatically from the "
        "Dashboard when the model runs. Check back after the first trading day."
    )
    st.stop()

# ── Parse DataFrame ────────────────────────────────────────────────────────────

df = pd.DataFrame(rows)

df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
df["scored_at"]  = pd.to_datetime(df["scored_at"],  errors="coerce", utc=True)

for _col in ["confidence_pct", "price_at_decision", "actual_price",
             "realized_return_net_of_cost", "atr_pct"]:
    if _col in df.columns:
        df[_col] = pd.to_numeric(df[_col], errors="coerce")

for _col in ["raw_signal", "realized_direction"]:
    if _col in df.columns:
        df[_col] = pd.to_numeric(df[_col], errors="coerce").astype("Int64")

# hit: True/False/None → nullable boolean (map handles NaN→missing automatically)
if "hit" in df.columns:
    df["hit"] = df["hit"].map({True: True, False: False}).astype("boolean")

if "metal" in df.columns:
    df["metal"] = df["metal"].str.lower()

df_scored  = df[df["scored_at"].notna()].copy()
df_pending = df[df["scored_at"].isna()].copy()
n_scored   = len(df_scored)
n_pending  = len(df_pending)

# ── Section 2: Counts ──────────────────────────────────────────────────────────

st.divider()
st.subheader("Prediction Counts")

_rows = []
for _m in METALS:
    _all = df[df["metal"] == _m]
    _sc  = df_scored[df_scored["metal"] == _m]
    _pe  = df_pending[df_pending["metal"] == _m]
    _rows.append({"Metal": _m.capitalize(), "Logged": len(_all),
                  "Scored": len(_sc), "Pending": len(_pe)})
_rows.append({"Metal": "Overall", "Logged": n_total,
              "Scored": n_scored, "Pending": n_pending})
st.dataframe(pd.DataFrame(_rows), hide_index=True, use_container_width=True)

if n_scored == 0:
    st.info(
        "No outcomes scored yet. Run `python scripts/score_outcomes.py` after "
        "the next market close, or wait for the scheduled scoring pass (#9)."
    )
    with st.expander("Raw prediction ledger"):
        st.dataframe(df.sort_values("as_of_date", ascending=False),
                     hide_index=True, use_container_width=True)
    st.stop()

_insufficient = n_scored < MIN_N
if _insufficient:
    st.warning(
        f"**Early stage — {n_scored} scored row{'s' if n_scored != 1 else ''} "
        f"(threshold: {MIN_N}).** "
        "All rates and charts below are **illustrative only** — Wilson CIs are "
        "shown for transparency but should not be treated as evidence of edge."
    )

# ── Section 3: Hit-rate ────────────────────────────────────────────────────────

st.divider()
st.subheader("Hit-Rate  (raw\\_signal vs realized\\_direction)")
st.caption(
    "Hit = raw_signal == realized_direction. 95% Wilson score CI shown. "
    + ("All values illustrative — n < 30." if _insufficient else "")
)

_hr_cols = st.columns(len(METALS) + 1)
for _i, _lbl in enumerate(METALS + ["overall"]):
    _sub  = df_scored if _lbl == "overall" else df_scored[df_scored["metal"] == _lbl]
    _n    = len(_sub)
    _hits = int(_sub["hit"].sum()) if _n > 0 else 0
    with _hr_cols[_i]:
        if _n == 0:
            st.metric(_lbl.capitalize(), "—")
            st.caption("n=0")
        else:
            _p       = _hits / _n
            _lo, _hi = wilson_ci(_hits, _n)
            _warn    = " ⚠" if _n < MIN_N else ""
            st.metric(_lbl.capitalize(), f"{_p * 100:.1f}%{_warn}")
            st.caption(f"n={_n} · 95% CI [{_lo*100:.1f}%–{_hi*100:.1f}%]")

# ── Section 4: Per-class recall + confusion matrix ─────────────────────────────

st.divider()
st.subheader("Per-Class Recall  &  Confusion Matrix")
st.caption(
    "Recall = fraction of realized class X predicted as X. "
    "Per-class n may be very small — low-n recall values are not authoritative."
)

_rc1, _rc2 = st.columns([1, 2])

with _rc1:
    _recall_rows = []
    for _cls in [2, 1, 0]:   # UP → SIDEWAYS → DOWN display order
        _act    = df_scored[df_scored["realized_direction"] == _cls]
        _n_cls  = len(_act)
        _n_corr = int((_act["raw_signal"] == _cls).sum()) if _n_cls > 0 else 0
        _rec    = _n_corr / _n_cls if _n_cls > 0 else None
        _note   = ("⚠ n<10 — illustrative" if 0 < _n_cls < 10
                   else "— no occurrences" if _n_cls == 0 else "")
        _recall_rows.append({
            "Class":      SIGNAL_LABELS[_cls],
            "n realized": _n_cls,
            "n correct":  _n_corr,
            "Recall":     f"{_rec*100:.1f}%" if _rec is not None else "—",
            "Note":       _note,
        })
    st.dataframe(pd.DataFrame(_recall_rows), hide_index=True, use_container_width=True)

with _rc2:
    _cls_lbls = [SIGNAL_LABELS[c] for c in [0, 1, 2]]
    _z = []
    for _pred in [0, 1, 2]:
        _row = []
        for _actual in [0, 1, 2]:
            _row.append(int(
                ((df_scored["raw_signal"] == _pred) &
                 (df_scored["realized_direction"] == _actual)).sum()
            ))
        _z.append(_row)

    _fig_cm = go.Figure(go.Heatmap(
        z=_z,
        x=[f"Realized {l}" for l in _cls_lbls],
        y=[f"Predicted {l}" for l in _cls_lbls],
        colorscale=[[0, "#1e2130"], [1, "#00CC88"]],
        text=[[str(v) for v in r] for r in _z],
        texttemplate="%{text}",
        showscale=False,
        xgap=2, ygap=2,
    ))
    _fig_cm.update_layout(
        **_PLT, height=260,
        title="Confusion matrix (rows = predicted, cols = realized)",
    )
    st.plotly_chart(_fig_cm, use_container_width=True)

# ── Section 5: Equity curve ────────────────────────────────────────────────────

st.divider()
st.subheader("Realized Equity Curve")
st.caption(
    "Raw model signal · 5 bps round-trip modelled · close-to-close · not as-traded. "
    "Indexed to 1.0. SIDEWAYS signals earn 0% (flat). "
    "Equal-weight blend is calendar-aligned: each metal forward-fills on dates "
    "where it has no scored row."
)

_fig_eq = go.Figure()
_eq_by_metal: dict[str, pd.Series] = {}

for _m in METALS:
    _ms = df_scored[df_scored["metal"] == _m].sort_values("as_of_date").copy()
    if _ms.empty:
        continue
    _ret    = _ms["realized_return_net_of_cost"].fillna(0.0).astype(float).values
    _dates  = _ms["as_of_date"].values
    _equity = np.cumprod(1.0 + _ret)
    _s      = pd.Series(_equity, index=pd.DatetimeIndex(_dates))
    _eq_by_metal[_m] = _s
    _fig_eq.add_trace(go.Scatter(
        x=_dates, y=_equity,
        name=_m.capitalize(),
        mode="lines+markers",
        marker=dict(size=5),
        line=dict(color=METAL_CLR[_m], width=2),
    ))

if len(_eq_by_metal) > 1:
    _all_dt  = sorted(set().union(*[set(s.index) for s in _eq_by_metal.values()]))
    _dt_idx  = pd.DatetimeIndex(_all_dt)
    _blend   = (pd.DataFrame({m: s.reindex(_dt_idx).ffill()
                               for m, s in _eq_by_metal.items()})
                  .mean(axis=1, skipna=True))
    _fig_eq.add_trace(go.Scatter(
        x=_dt_idx, y=_blend.values,
        name="Equal-weight blend",
        mode="lines",
        line=dict(color="#FFFFFF", width=1.5, dash="dot"),
    ))

_fig_eq.add_hline(y=1.0, line_color="rgba(255,255,255,0.15)", line_width=1)
st.plotly_chart(_dark(_fig_eq, height=380), use_container_width=True)

if len(_eq_by_metal) > 1:
    st.caption(
        "Equal-weight illustration across metals — not a traded portfolio."
    )

# ── Section 6: Calibration ─────────────────────────────────────────────────────

st.divider()
st.subheader("Confidence Calibration  ·  Quintile Bins")
st.caption(
    "Mean confidence per quintile vs observed hit-rate. "
    "Open markers = bin n < 5 (not reliable)."
)

_cal = df_scored[df_scored["confidence_pct"].notna() & df_scored["hit"].notna()].copy()

if len(_cal) < 5:
    st.info("Insufficient data for calibration chart (need ≥ 5 scored rows with confidence).")
else:
    _cal["_qbin"] = pd.qcut(_cal["confidence_pct"], q=5, labels=False, duplicates="drop")
    _cal_agg = (
        _cal.groupby("_qbin", observed=True)
        .agg(mean_conf=("confidence_pct", "mean"),
             hit_rate=("hit", "mean"),
             n=("hit", "count"))
        .reset_index()
    )

    _conf_min = float(_cal["confidence_pct"].min())
    _conf_max = float(_cal["confidence_pct"].max())

    _fig_cal = go.Figure()
    _fig_cal.add_trace(go.Scatter(
        x=[_conf_min, _conf_max],
        y=[_conf_min / 100.0, _conf_max / 100.0],
        mode="lines", name="Perfect calibration",
        line=dict(color="#555", dash="dot", width=1.5),
    ))
    for _, _r in _cal_agg.iterrows():
        _nb = int(_r["n"])
        _ok = _nb >= 5
        _fig_cal.add_trace(go.Scatter(
            x=[float(_r["mean_conf"])], y=[float(_r["hit_rate"])],
            mode="markers+text",
            text=[f"n={_nb}"],
            textposition="top center",
            marker=dict(
                size=12,
                color="#00CC88" if _ok else "rgba(0,204,136,0)",
                line=dict(color="#00CC88", width=2),
                symbol="circle",
            ),
            showlegend=False,
        ))
    _fig_cal.update_layout(
        **_PLT, height=320,
        xaxis_title="Mean confidence in bin (%)",
        yaxis_title="Observed hit-rate",
        showlegend=True,
    )
    _fig_cal.update_yaxes(range=[0, 1], showgrid=True, gridcolor=GRID_CLR)
    _fig_cal.update_xaxes(showgrid=False)
    st.plotly_chart(_fig_cal, use_container_width=True)

# ── Section 7: Base-vs-meta expander ──────────────────────────────────────────

with st.expander("Model architecture: Base classifiers vs Meta-learner", expanded=False):
    st.markdown(
        """
**Why can "Model Agreement 100%" coexist with a SIDEWAYS or reversed final signal?**

The APEX model is a two-layer stacking ensemble:

1. **Base classifiers** (XGBoost, LightGBM, CatBoost) each produce a 3-class probability
   vector. "Model Agreement" on the Dashboard counts how many base argmaxes agree.
2. **Meta-learner** receives the full 9-dimensional L1 probability vector and has learned
   *when* unanimous base consensus historically fails — e.g. three bases all leaning UP
   with low conviction in a high-volatility regime. In those cases the meta-learner can
   output SIDEWAYS or DOWN despite unanimous base UP votes.
   **This is the stacking layer working as designed, not a contradiction.**

**Can the meta-override edge be measured from this track record?**

Not yet. `raw_signal` stored here is the meta-learner's output. The base classifiers'
individual probability vectors at decision time were **not logged separately** to the
chain store — only the final `proba_vector` (post-meta) was captured. Without a stored
`base_signal` field, rows cannot be split into "meta agreed" vs "meta overrode."
A future enhancement could log `base_proba_vector` as an additional non-hashed field
to enable this analysis.

**Source consistency:** The base votes shown on the Dashboard use the same feature
vector `X` as the meta-learner — same bar, same model objects, deterministic outputs.
The displayed base votes faithfully reflect what the meta-learner's L1 layer received.
        """
    )

# ── Section 8: Prediction ledger ──────────────────────────────────────────────

st.divider()
st.subheader("Prediction Ledger")
st.caption(
    "All logged predictions, newest first. Chain-linked: no row can be added, "
    "removed, or altered without breaking the SHA-256 check at the top of the page."
)

_disp_cols = [
    "as_of_date", "metal", "raw_signal", "displayed_signal", "confidence_pct",
    "price_at_decision", "actual_price", "realized_direction", "hit",
    "realized_return_net_of_cost", "scored_at", "verdict", "regime",
    "model_version", "record_hash",
]
_disp = df[[c for c in _disp_cols if c in df.columns]].copy()

_disp["raw_signal"] = _disp["raw_signal"].apply(
    lambda x: _fmt_sig(x) if pd.notna(x) else "—"
)
if "realized_direction" in _disp.columns:
    _disp["realized_direction"] = _disp["realized_direction"].apply(
        lambda x: _fmt_sig(x) if pd.notna(x) else "—"
    )

_disp = _disp.sort_values("as_of_date", ascending=False)
st.dataframe(_disp, hide_index=True, use_container_width=True)
