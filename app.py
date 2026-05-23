"""
Gold AI Trader Assistant — MVP Dashboard
Streamlit multi-page app  |  NOT FINANCIAL ADVICE  |  Personal research only
"""
import os, sys, warnings, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.config import (
    PRIMARY_TICKER, TRAIN_YEARS, TEST_YEARS, N_TRIALS,
    INITIAL_CAPITAL, MIN_CONFIDENCE, MAX_ATR_PCT,
    MAX_DRAWDOWN_HALT, MAX_DAILY_LOSS_PCT, FRED_API_KEY,
    BULL_UP_CONF_RELAXED, BULL_REGIME_ENABLED,
)
from src.data_loader import download_data, get_train_test_split, get_live_spot_price
from src.features import add_features
from src.macro_loader import download_fred, add_macro_features
from src.regime import get_current_regime, detect_regime, REGIME_LABELS, REGIME_COLORS
from src.signals import generate_latest_signal, SIGNAL_LABELS, SIGNAL_COLORS
from src.train import train_all_models, save_models, load_models
from src.backtest import run_backtest
from src.benchmarks import run_all_benchmarks

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold AI — Dashboard",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG  = "#0e1117"
GRID_CLR = "#1e2130"
_PLT = dict(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font=dict(color="white"))

def _dark(fig, height=420):
    fig.update_layout(**_PLT, height=height, legend=dict(orientation="h", y=1.02))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
    return fig

# ── Session state ──────────────────────────────────────────────────────────────
_KEYS = ("df", "macro_df", "reg_results", "clf_results", "feature_cols",
         "stack_reg", "stack_clf", "backtest_results", "benchmark_results",
         "signal", "regime_info", "refresh_key")
for k in _KEYS:
    if k not in st.session_state:
        st.session_state[k] = None if k != "refresh_key" else 0

# ── Cached data loader ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Downloading market data…")
def _load_price(refresh_key: int) -> pd.DataFrame:
    raw = download_data(force_refresh=(refresh_key > 0))
    return add_features(raw)

@st.cache_data(ttl=3600, show_spinner="Downloading FRED macro data…")
def _load_macro(refresh_key: int, fred_key: str) -> pd.DataFrame:
    if not fred_key:
        return pd.DataFrame()
    os.environ["FRED_API_KEY"] = fred_key
    return download_fred(force_refresh=(refresh_key > 0))

@st.cache_data(ttl=300, show_spinner=False)
def _load_live_price(api_key: str) -> tuple:
    return get_live_spot_price(api_key)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🥇 Gold AI v1")
    st.caption("NOT FINANCIAL ADVICE")
    st.divider()

    col_r, col_t = st.columns(2)
    refresh_btn = col_r.button("🔄 Refresh", width="stretch")
    train_btn   = col_t.button("🚀 Train",   width="stretch", type="primary")

    st.divider()
    fred_key = st.text_input(
        "FRED API Key (optional)", type="password", value=FRED_API_KEY,
        help="Free key at fred.stlouisfed.org — enables macro features",
    )
    if fred_key:
        os.environ["FRED_API_KEY"] = fred_key

    st.divider()
    with st.expander("⚙️ Config"):
        st.caption(f"Ticker: `{PRIMARY_TICKER}`")
        st.caption(f"Train window: {TRAIN_YEARS} years")
        st.caption(f"Optuna trials: {N_TRIALS}")
        st.caption(f"Bull threshold: {BULL_UP_CONF_RELAXED:.2f}")
        st.caption(f"Min confidence: {MIN_CONFIDENCE:.0%}")

# ── Load data ──────────────────────────────────────────────────────────────────
if refresh_btn:
    st.session_state.refresh_key += 1
    st.session_state.signal = None
    st.session_state.backtest_results = None

try:
    raw_df   = _load_price(st.session_state.refresh_key)
    macro_df = _load_macro(
        st.session_state.refresh_key,
        os.environ.get("FRED_API_KEY", ""),
    )
    df = add_macro_features(raw_df, macro_df)
    st.session_state.df = df
    st.session_state.macro_df = macro_df
except Exception as exc:
    st.error(f"Failed to load data: {exc}")
    st.stop()

df = st.session_state.df

# ── Auto-load saved models ─────────────────────────────────────────────────────
if st.session_state.reg_results is None:
    reg, clf, feat, sr, sc = load_models()
    if reg is not None:
        st.session_state.reg_results  = reg
        st.session_state.clf_results  = clf
        st.session_state.feature_cols = feat
        st.session_state.stack_reg    = sr
        st.session_state.stack_clf    = sc

# ── Training flow ──────────────────────────────────────────────────────────────
if train_btn:
    train_df, test_df = get_train_test_split(df)
    with st.status("Training Gold AI v1… (15–30 min first run)", expanded=True) as status:
        log_box = st.empty()
        msgs: list[str] = []

        def _log(msg: str):
            msgs.append(msg)
            log_box.markdown("\n".join(f"• {m}" for m in msgs[-25:]))

        try:
            reg_r, clf_r, feat, sr, sc = train_all_models(
                train_df, test_df,
                n_trials=N_TRIALS,
                progress_callback=_log,
            )
            _log("Running backtest…")
            stk_preds  = clf_r["Stacking"]["predictions"]
            stk_probas = clf_r["Stacking"].get("probabilities")
            test_dates = clf_r["Stacking"]["test_dates"]
            regime_s   = detect_regime(df).reindex(test_dates)
            eq, bh, trades_df, bt_m = run_backtest(
                df, stk_preds, test_dates,
                clf_probas=stk_probas, regime_series=regime_s,
            )

            _log("Running benchmarks…")
            bm = run_all_benchmarks(df, test_dates, initial_capital=INITIAL_CAPITAL)

            _log("Saving models…")
            save_models(reg_r, clf_r, feat, sr, sc)

            st.session_state.reg_results      = reg_r
            st.session_state.clf_results      = clf_r
            st.session_state.feature_cols     = feat
            st.session_state.stack_reg        = sr
            st.session_state.stack_clf        = sc
            st.session_state.backtest_results = (eq, bh, trades_df, bt_m)
            st.session_state.benchmark_results = bm

            _log("Generating signal…")
            ri = get_current_regime(df)
            st.session_state.regime_info = ri
            st.session_state.signal = generate_latest_signal(
                df, reg_r, clf_r, feat,
                stack_reg=sr, stack_clf=sc,
                regime_int=ri["regime_int"] if ri else 5,
            )
            status.update(label="✅ Training complete!", state="complete")
            st.rerun()
        except Exception as exc:
            import traceback
            status.update(label=f"❌ {exc}", state="error")
            st.error(traceback.format_exc())

# ── Auto-generate signal + regime if models loaded but signal missing ──────────
if st.session_state.reg_results is not None:
    if st.session_state.regime_info is None:
        try:
            st.session_state.regime_info = get_current_regime(df)
        except Exception:
            pass

    if st.session_state.backtest_results is None:
        try:
            clf_r = st.session_state.clf_results
            stk_preds  = clf_r["Stacking"]["predictions"]
            stk_probas = clf_r["Stacking"].get("probabilities")
            test_dates = clf_r["Stacking"]["test_dates"]
            regime_s   = detect_regime(df).reindex(test_dates)
            eq, bh, trades_df, bt_m = run_backtest(
                df, stk_preds, test_dates,
                clf_probas=stk_probas, regime_series=regime_s,
            )
            bm = run_all_benchmarks(df, test_dates, initial_capital=INITIAL_CAPITAL)
            st.session_state.backtest_results  = (eq, bh, trades_df, bt_m)
            st.session_state.benchmark_results = bm
        except Exception:
            pass

    if st.session_state.signal is None:
        try:
            ri = st.session_state.regime_info or get_current_regime(df)
            st.session_state.signal = generate_latest_signal(
                df,
                st.session_state.reg_results,
                st.session_state.clf_results,
                st.session_state.feature_cols,
                stack_reg=st.session_state.stack_reg,
                stack_clf=st.session_state.stack_clf,
                regime_int=ri["regime_int"] if ri else 5,
            )
        except Exception:
            pass

# ── Aliases ────────────────────────────────────────────────────────────────────
signal      = st.session_state.signal
regime_info = st.session_state.regime_info
models_ok   = st.session_state.reg_results is not None

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.title("🥇 Gold AI Trader — Dashboard")
st.caption("⚠️ Personal research only · NOT financial advice · Past performance does not guarantee future results")

# ── KPI row ───────────────────────────────────────────────────────────────────
_td_key = st.secrets.get("TWELVE_DATA_API_KEY", os.environ.get("TWELVE_DATA_API_KEY", ""))
_live_price, _price_source = _load_live_price(_td_key)

last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2])
if _live_price is not None:
    cur  = _live_price
    chg  = cur - last_close
    chgp = chg / last_close * 100
else:
    cur  = last_close
    chg  = last_close - prev_close
    chgp = chg / prev_close * 100
last_date = df.index[-1].strftime("%Y-%m-%d %H:%M UTC") if hasattr(df.index[-1], "strftime") else str(df.index[-1])

c1, c2, c3, c4 = st.columns(4)

c1.metric("Gold Price (XAU/USD)", f"${cur:,.2f}",
          f"{chg:+.2f}  ({chgp:+.2f}%)")
c1.caption(f"📡 {_price_source}")

if signal:
    sig_clr  = signal["signal_color"]
    sig_lbl  = signal["signal_label"]
    sig_emi  = signal["signal_emoji"]
    conf_pct = signal.get("confidence_pct", 0)

    c2.markdown(
        f"""<div style="border:2px solid {sig_clr};border-radius:10px;
            padding:14px;text-align:center;height:80px">
            <div style="font-size:0.72em;color:#aaa;margin-bottom:4px">Signal (next day)</div>
            <div style="font-size:1.9em;font-weight:bold;color:{sig_clr}">
                {sig_emi} {sig_lbl}</div></div>""",
        unsafe_allow_html=True,
    )
    c3.metric("Model Confidence", f"{conf_pct:.1f}%",
              help="Calibrated probability for predicted class")
else:
    c2.info("Signal: Train model")
    c3.metric("Model Confidence", "—")

if regime_info:
    rc   = regime_info["regime_color"]
    rlbl = regime_info["regime_label"]
    remi = regime_info.get("regime_emoji", "")
    c4.markdown(
        f"""<div style="border:2px solid {rc};border-radius:10px;
            padding:14px;text-align:center;height:80px">
            <div style="font-size:0.72em;color:#aaa;margin-bottom:4px">Market Regime</div>
            <div style="font-size:1.4em;font-weight:bold;color:{rc}">
                {remi} {rlbl}</div></div>""",
        unsafe_allow_html=True,
    )
else:
    c4.metric("Market Regime", "—")

st.caption(f"Last bar: {last_date}")

# ── Probability breakdown ─────────────────────────────────────────────────────
if signal and signal.get("proba_vec"):
    pv = signal["proba_vec"]
    st.divider()
    st.subheader("Directional Probability")
    pb1, pb2, pb3 = st.columns(3)
    pb1.metric("🔴 DOWN",     f"{pv[0]*100:.1f}%", delta_color="off")
    pb2.metric("⚪ SIDEWAYS", f"{pv[1]*100:.1f}%", delta_color="off")
    pb3.metric("🟢 UP",       f"{pv[2]*100:.1f}%", delta_color="off")

    if signal.get("filter_reason"):
        st.warning(f"⛔ Signal filtered: {signal['filter_reason']}")
    elif signal["signal_int"] != 1:
        st.success("✅ All entry filters passed")

# ── ATR / VIX row ─────────────────────────────────────────────────────────────
st.divider()
m1, m2, m3, m4 = st.columns(4)

atr_val = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else cur * 0.01
atr_pct = atr_val / cur * 100
m1.metric("ATR / Price", f"{atr_pct:.2f}%",
          delta="⚠️ High" if atr_pct > MAX_ATR_PCT * 100 else "✅ Normal",
          delta_color="off")
m2.metric("ATR ($)", f"${atr_val:.2f}")

vix_cols = [c for c in df.columns if "VIX" in c.upper() and c.endswith("_Close")]
if vix_cols:
    vix = float(df[vix_cols[0]].iloc[-1])
    m3.metric("VIX", f"{vix:.1f}",
              delta="🔴 High" if vix > 25 else "🟢 Low" if vix < 15 else "⚪ Normal",
              delta_color="off")

dxy_cols = [c for c in df.columns if "DXY" in c.upper() and c.endswith("_Close")]
if dxy_cols:
    dxy = float(df[dxy_cols[0]].iloc[-1])
    m4.metric("DXY (USD Index)", f"{dxy:.2f}")

# ── Candlestick ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Gold Price — Last 90 Days")
recent = df.tail(90)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.78, 0.22], vertical_spacing=0.03)
fig.add_trace(go.Candlestick(
    x=recent.index,
    open=recent["Open"], high=recent["High"],
    low=recent["Low"],  close=recent["Close"],
    name="OHLC",
    increasing_line_color="#00CC88", decreasing_line_color="#FF4B4B",
), row=1, col=1)
for label, col, color in [("SMA 20", "SMA_20", "#FFA500"),
                            ("SMA 50", "SMA_50", "#00BFFF")]:
    if col in recent.columns:
        fig.add_trace(go.Scatter(x=recent.index, y=recent[col],
                                  name=label, line=dict(color=color, width=1.2)),
                      row=1, col=1)
if "BB_Upper" in recent.columns:
    fig.add_trace(go.Scatter(
        x=list(recent.index) + list(recent.index[::-1]),
        y=list(recent["BB_Upper"]) + list(recent["BB_Lower"][::-1]),
        fill="toself", fillcolor="rgba(128,128,128,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="Bollinger", showlegend=False,
    ), row=1, col=1)
if "Volume" in recent.columns:
    fig.add_trace(go.Bar(x=recent.index, y=recent["Volume"],
                         name="Volume", marker_color="rgba(150,150,200,0.35)",
                         showlegend=False), row=2, col=1)

# Mark signal on chart if available
if signal and signal["signal_int"] != 1:
    arrow_color = "#00CC88" if signal["signal_int"] == 2 else "#FF4B4B"
    arrow_sym   = "triangle-up" if signal["signal_int"] == 2 else "triangle-down"
    fig.add_trace(go.Scatter(
        x=[df.index[-1]], y=[cur],
        mode="markers",
        name=signal["signal_label"],
        marker=dict(color=arrow_color, size=14, symbol=arrow_sym),
    ), row=1, col=1)

fig.update_layout(**_PLT, height=460, xaxis_rangeslider_visible=False,
                  legend=dict(orientation="h", y=1.02))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
st.plotly_chart(fig, width="stretch")

if not models_ok:
    st.info("👆 Press **Train** in the sidebar to generate the AI signal. First run takes 15–30 min.")
