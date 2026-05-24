"""
Gold AI Decision Intelligence Platform — Dashboard
Streamlit multi-page app  |  NOT FINANCIAL ADVICE  |  Personal research only
"""
import json
import os, sys, warnings, time
from datetime import datetime, timedelta, timezone
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
    BULL_UP_CONF_RELAXED, BULL_REGIME_ENABLED, DATA_DIR,
)
from src.data_loader import download_data, get_train_test_split, get_live_spot_price
from src.features import add_features
from src.macro_loader import download_fred, add_macro_features
from src.regime import get_current_regime, detect_regime, REGIME_LABELS, REGIME_COLORS
from src.signals import generate_latest_signal, SIGNAL_LABELS, SIGNAL_COLORS
from src.train import train_all_models, save_models, load_models
from src.backtest import run_backtest
from src.benchmarks import run_all_benchmarks
from src.alerts import send_signal_alert, already_sent_today, send_risk_alert, risk_alert_eligible
from src.wfv import run_wfv

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold AI Decision Intelligence — Dashboard",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG  = "#0e1117"
GRID_CLR = "#1e2130"

_RETRAIN_LOG = os.path.join(DATA_DIR, "model_retrain_log.json")
_UAE_TZ      = timezone(timedelta(hours=4))


def _read_retrain_log() -> dict:
    blank = {"last_retrain_utc": "", "last_bar_date": ""}
    if not os.path.exists(_RETRAIN_LOG):
        return blank
    try:
        with open(_RETRAIN_LOG) as f:
            return {**blank, **json.load(f)}
    except Exception:
        return blank


def _write_retrain_log(last_bar_date: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(_RETRAIN_LOG, "w") as f:
        json.dump({"last_retrain_utc": datetime.now(timezone.utc).isoformat(),
                   "last_bar_date": last_bar_date}, f)
_PLT = dict(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font=dict(color="white"))

def _dark(fig, height=420):
    fig.update_layout(**_PLT, height=height, legend=dict(orientation="h", y=1.02))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
    return fig

# ── Session state ──────────────────────────────────────────────────────────────
_KEYS = ("df", "macro_df", "reg_results", "clf_results", "feature_cols",
         "stack_reg", "stack_clf", "backtest_results", "benchmark_results",
         "signal", "regime_info", "refresh_key", "alert_status", "risk_alert_status",
         "wfv_results")
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
    st.title("🥇 Gold AI Decision Intelligence")
    st.caption("NOT FINANCIAL ADVICE")
    st.divider()

    col_r, col_t = st.columns(2)
    refresh_btn   = col_r.button("🔄 Refresh",     width="stretch")
    train_btn     = col_t.button("🚀 Train",        width="stretch", type="primary")
    auto_retrain_btn = st.button("⚡ Auto-Retrain (fast, no Optuna)", width="stretch")

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
    with st.status("Training Gold AI Decision Intelligence… (15–30 min first run)", expanded=True) as status:
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

# ── Auto-Retrain (3C) ─────────────────────────────────────────────────────────
_retrain_log    = _read_retrain_log()
_cur_bar_date   = (df.index[-1].strftime("%Y-%m-%d")
                   if hasattr(df.index[-1], "strftime") else str(df.index[-1])[:10])
_models_trained = st.session_state.reg_results is not None
_new_bar        = _models_trained and _retrain_log["last_bar_date"] != _cur_bar_date
_hp_available   = (_models_trained
                   and st.session_state.clf_results is not None
                   and "_hyperparams" in st.session_state.clf_results)

if (auto_retrain_btn or _new_bar) and _hp_available:
    _pretrained_hp = st.session_state.clf_results["_hyperparams"]
    _trigger_label = "Manual auto-retrain" if auto_retrain_btn else f"New bar detected ({_cur_bar_date})"
    train_df, test_df = get_train_test_split(df)
    with st.status(f"⚡ {_trigger_label} — retraining with saved hyperparams…", expanded=True) as _ar_status:
        _ar_log_box = st.empty()
        _ar_msgs: list[str] = []

        def _ar_log(msg: str):
            _ar_msgs.append(msg)
            _ar_log_box.markdown("\n".join(f"• {m}" for m in _ar_msgs[-15:]))

        try:
            reg_r, clf_r, feat, sr, sc = train_all_models(
                train_df, test_df,
                n_trials=N_TRIALS,
                pretrained_hyperparams=_pretrained_hp,
                fast_retrain=True,
                progress_callback=_ar_log,
            )
            _ar_log("Running backtest…")
            stk_preds  = clf_r["Stacking"]["predictions"]
            stk_probas = clf_r["Stacking"].get("probabilities")
            test_dates = clf_r["Stacking"]["test_dates"]
            from src.regime import detect_regime
            regime_s   = detect_regime(df).reindex(test_dates)
            eq, bh, trades_df, bt_m = run_backtest(
                df, stk_preds, test_dates,
                clf_probas=stk_probas, regime_series=regime_s,
            )
            bm = run_all_benchmarks(df, test_dates, initial_capital=INITIAL_CAPITAL)
            _ar_log("Saving models…")
            save_models(reg_r, clf_r, feat, sr, sc)
            _write_retrain_log(_cur_bar_date)

            st.session_state.reg_results      = reg_r
            st.session_state.clf_results      = clf_r
            st.session_state.feature_cols     = feat
            st.session_state.stack_reg        = sr
            st.session_state.stack_clf        = sc
            st.session_state.backtest_results = (eq, bh, trades_df, bt_m)
            st.session_state.benchmark_results = bm
            st.session_state.signal           = None  # force regeneration

            _ar_status.update(label="✅ Auto-retrain complete!", state="complete")
            st.rerun()
        except Exception as exc:
            import traceback
            _ar_status.update(label=f"❌ {exc}", state="error")
            st.error(traceback.format_exc())
elif auto_retrain_btn and not _hp_available:
    st.sidebar.warning("Auto-retrain requires a trained model with saved hyperparams. Run full Train first.")

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

# ── Email alert trigger ───────────────────────────────────────────────────────
# Initialise status from file on first page load
if st.session_state.alert_status is None:
    st.session_state.alert_status = "sent" if already_sent_today() else "ready"

# Only attempt if status is still "ready" (not yet sent or errored this session)
if st.session_state.alert_status == "ready" and signal:
    _alert_conditions = (
        signal["signal_int"] in (0, 2)            # UP or DOWN only
        and signal.get("confidence_pct", 0) > 60  # above 60% confidence
        and not signal.get("filter_reason")        # No Trade filter NOT active
    )
    if _alert_conditions:
        _al_sender    = st.secrets.get("GMAIL_SENDER", "")
        _al_password  = st.secrets.get("GMAIL_APP_PASSWORD", "")
        _al_recipient = st.secrets.get("ALERT_RECIPIENT", "")
        if _al_sender and _al_password and _al_recipient:
            _al_ok, _al_msg = send_signal_alert(
                signal, regime_info, _al_sender, _al_password, _al_recipient
            )
            st.session_state.alert_status = (
                "sent" if (_al_ok or _al_msg == "already_sent_today") else "error"
            )
        else:
            st.session_state.alert_status = "not_configured"
    else:
        st.session_state.alert_status = "no_alert"

# ── Risk alert trigger ────────────────────────────────────────────────────────
# Evaluated on every page load; cooldown and daily cap enforced inside module.
_ra_eligible, _ = risk_alert_eligible()
if _ra_eligible and df is not None:
    _ra_sender    = st.secrets.get("GMAIL_SENDER", "")
    _ra_password  = st.secrets.get("GMAIL_APP_PASSWORD", "")
    _ra_recipient = st.secrets.get("ALERT_RECIPIENT", "")
    if _ra_sender and _ra_password and _ra_recipient:
        _ra_ok, _ra_msg = send_risk_alert(
            df, float(df["Close"].iloc[-1]),
            signal, _ra_sender, _ra_password, _ra_recipient,
        )
        st.session_state.risk_alert_status = "sent" if _ra_ok else _ra_msg
    else:
        st.session_state.risk_alert_status = "not_configured"

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.title("🥇 Gold AI Decision Intelligence — Dashboard")
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

# ── Model last-retrained timestamp (3C) ───────────────────────────────────────
_rl = _read_retrain_log()
if _rl["last_retrain_utc"]:
    try:
        _lr_dt  = datetime.fromisoformat(_rl["last_retrain_utc"])
        _lr_uae = _lr_dt.astimezone(_UAE_TZ)
        st.caption(f"Model last retrained: {_lr_uae.strftime('%Y-%m-%d at %H:%M UAE')}")
    except Exception:
        pass

_al_map = {
    "sent":           ("📧", "#00CC88", "Alert sent today"),
    "no_alert":       ("🔕", "#888888", "No alert today — conditions not met"),
    "ready":          ("🔔", "#888888", "Alert system ready"),
    "not_configured": ("⚙️",  "#FFA500", "Alert not configured — add GMAIL_SENDER, GMAIL_APP_PASSWORD, ALERT_RECIPIENT to secrets"),
    "error":          ("⚠️",  "#FF4B4B", "Alert error — check secrets or SMTP settings"),
}
_al_icon, _al_color, _al_text = _al_map.get(
    st.session_state.alert_status or "ready",
    ("🔔", "#888888", "Alert system ready"),
)
st.markdown(
    f'<span style="font-size:0.8em;color:{_al_color}">{_al_icon} {_al_text}</span>',
    unsafe_allow_html=True,
)

_ra_status = st.session_state.risk_alert_status or ""
if not _ra_status:
    _ra_icon, _ra_color, _ra_text = "🔔", "#888888", "Risk monitor: ready"
elif _ra_status == "sent":
    _ra_icon, _ra_color, _ra_text = "🟠", "#FFA500", "Risk alert sent"
elif _ra_status == "not_configured":
    _ra_icon, _ra_color, _ra_text = "⚙️",  "#FFA500", "Risk alert not configured"
elif _ra_status.startswith("insufficient_conditions"):
    _ra_icon, _ra_color, _ra_text = "🟢", "#00CC88", "Risk monitor: no conditions triggered"
elif _ra_status.startswith("cooldown"):
    _ra_icon, _ra_color, _ra_text = "🟠", "#FFA500", f"Risk monitor: {_ra_status}"
elif _ra_status.startswith("daily_cap"):
    _ra_icon, _ra_color, _ra_text = "🟠", "#FFA500", "Risk monitor: daily cap reached (3/day)"
else:
    _ra_icon, _ra_color, _ra_text = "⚠️",  "#FF4B4B", "Risk alert error — check SMTP settings"
st.markdown(
    f'<span style="font-size:0.8em;color:{_ra_color}">{_ra_icon} {_ra_text}</span>',
    unsafe_allow_html=True,
)

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

# ── Signal Strength Indicator ─────────────────────────────────────────────────
if signal:
    _ss_int  = signal["signal_int"]          # 0=DOWN, 1=SIDEWAYS, 2=UP
    _ss_conf = signal.get("confidence_pct", 0)
    _ss_lbl  = signal.get("signal_label", "SIDEWAYS")

    # Bar colour by direction
    if _ss_int == 2:
        _ss_color = "#00CC88"   # green  — UP / BUY
    elif _ss_int == 0:
        _ss_color = "#FF4B4B"   # red    — DOWN / SELL
    else:
        _ss_color = "#888888"   # grey   — SIDEWAYS / No Trade

    # Strength label by confidence band
    if _ss_conf <= 40:
        _ss_grade = "Weak — Exercise Caution"
    elif _ss_conf <= 60:
        _ss_grade = "Moderate — Monitor Closely"
    elif _ss_conf <= 80:
        _ss_grade = "Strong — Signal Worth Considering"
    else:
        _ss_grade = "Very Strong — High Conviction Signal"

    # Plain-English summary
    _ss_grade_word = _ss_grade.split("—")[0].strip().lower()
    if _ss_int == 1:
        _ss_summary = (
            f"AI model favours {_ss_lbl} movement with {_ss_grade_word} confidence. "
            f"No trade recommended at this time."
        )
    else:
        _ss_summary = (
            f"AI model favours {_ss_lbl} movement with {_ss_grade_word} confidence. "
            f"Consider reviewing risk parameters before acting."
        )

    st.divider()
    st.subheader("Signal Strength")
    st.markdown(
        f"""
        <div style="margin-bottom:6px;font-size:0.85em;color:#aaa;">
            Confidence &nbsp;·&nbsp;
            <span style="color:{_ss_color};font-weight:bold;">{_ss_conf:.1f}%</span>
        </div>
        <div style="background:#1e2130;border-radius:6px;height:18px;
                    width:100%;overflow:hidden;">
            <div style="background:{_ss_color};width:{_ss_conf:.1f}%;
                        height:100%;border-radius:6px;"></div>
        </div>
        <div style="margin-top:8px;font-size:0.95em;
                    color:{_ss_color};font-weight:600;">{_ss_grade}</div>
        <div style="margin-top:10px;font-size:0.95em;color:#e0e0e0;">
            {_ss_summary}
        </div>
        <div style="margin-top:10px;font-size:0.78em;color:#888;font-style:italic;">
            Signal strength reflects model confidence only.
            It does not guarantee outcome or constitute financial advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

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

# ── AI 5-Day Price Projection ─────────────────────────────────────────────────
if signal:
    st.divider()
    st.subheader("AI 5-Day Price Projection")

    _CONTEXT_DAYS = 10
    _PROJ_DAYS    = 5

    _ctx        = df.tail(_CONTEXT_DAYS)
    _last_close = float(_ctx["Close"].iloc[-1])
    _last_dt    = _ctx.index[-1]

    # Direction from signal_int: 2=UP, 1=SIDEWAYS, 0=DOWN
    _direction   = {2: +1, 1: 0, 0: -1}.get(signal["signal_int"], 0)
    _conf        = signal.get("confidence_pct", 50) / 100
    _daily_drift = _direction * atr_val * 0.25 * _conf

    _future_dates = pd.bdate_range(start=_last_dt + pd.Timedelta(days=1), periods=_PROJ_DAYS)
    _proj         = [_last_close + _daily_drift * i for i in range(1, _PROJ_DAYS + 1)]
    _upper        = [p + atr_val for p in _proj]
    _lower        = [p - atr_val for p in _proj]

    _fig_proj = go.Figure()

    # Last 10 days actual price (green solid line)
    _fig_proj.add_trace(go.Scatter(
        x=_ctx.index, y=_ctx["Close"],
        name="Actual Price (10d)",
        line=dict(color="#00CC88", width=2),
    ))

    # ATR confidence band (shaded orange fill)
    _fig_proj.add_trace(go.Scatter(
        x=list(_future_dates) + list(_future_dates[::-1]),
        y=_upper + _lower[::-1],
        fill="toself", fillcolor="rgba(255,165,0,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="ATR Confidence Band (±1× ATR)",
    ))

    # Connector from last actual point to first projected point
    _fig_proj.add_trace(go.Scatter(
        x=[_last_dt, _future_dates[0]], y=[_last_close, _proj[0]],
        mode="lines", line=dict(color="#FFA500", width=1.5, dash="dot"),
        showlegend=False,
    ))

    # 5-day projected line (orange dotted)
    _fig_proj.add_trace(go.Scatter(
        x=list(_future_dates), y=_proj,
        name="AI Projection (5d)",
        line=dict(color="#FFA500", width=2, dash="dot"),
    ))

    # Vertical divider between actual and projected regions
    _fig_proj.add_vline(
        x=str(_last_dt.date()),
        line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"),
    )

    # "Indicative Only" watermark annotation
    _fig_proj.add_annotation(
        text="AI Projection — Indicative Only",
        xref="paper", yref="paper", x=0.99, y=0.97,
        showarrow=False, font=dict(color="#FFA500", size=12), align="right",
    )

    _dark(_fig_proj, height=360)
    st.plotly_chart(_fig_proj, width="stretch")

    st.caption(
        "⚠️ This projection is generated from model signals and ATR-based volatility estimates. "
        "It is indicative only and does not constitute a price forecast or financial advice."
    )

if not models_ok:
    st.info("👆 Press **Train** in the sidebar to generate the AI signal. First run takes 15–30 min.")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3A — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📊 Feature Importance (Top 15)", expanded=False):
    if not models_ok:
        st.info("Train the model to see feature importances.")
    else:
        _fi_data, _fi_model_name = None, None
        for _mn in ["XGBoost", "LightGBM", "CatBoost"]:
            _mc = st.session_state.clf_results.get(_mn, {}) if st.session_state.clf_results else {}
            _fi = _mc.get("feature_importance")
            if _fi is not None and len(_fi) > 0:
                _fi_data, _fi_model_name = _fi, _mn
                break

        _feat_cols = st.session_state.feature_cols or []
        if _fi_data is not None and _feat_cols:
            _fi_df = (
                pd.DataFrame({"Feature": _feat_cols[:len(_fi_data)], "Importance": _fi_data})
                .sort_values("Importance", ascending=False)
                .head(15)
            )
            _fi_fig = go.Figure(go.Bar(
                x=_fi_df["Importance"], y=_fi_df["Feature"],
                orientation="h", marker_color="#00CC88",
            ))
            _fi_fig.update_layout(
                **_PLT, height=420,
                yaxis=dict(autorange="reversed"),
                xaxis_title="Importance Score",
                margin=dict(l=180, r=20, t=30, b=40),
            )
            _fi_fig.update_xaxes(showgrid=True, gridcolor=GRID_CLR)
            _fi_fig.update_yaxes(showgrid=False)
            st.caption(f"Source: {_fi_model_name} classifier — top 15 of {len(_feat_cols)} features")
            st.plotly_chart(_fi_fig, width="stretch")
        else:
            st.info("Feature importance data not available — retrain the model.")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3B — Walk-Forward Validation
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔄 Walk-Forward Validation", expanded=False):
    if not models_ok:
        st.info("Train the model first to run walk-forward validation.")
    else:
        _wfv_col1, _wfv_col2 = st.columns([3, 1])
        _wfv_col1.markdown(
            "LightGBM rolling walk-forward: **252-bar train · 21-bar test · 21-bar step**. "
            "Each fold is an independent out-of-sample period. Takes ~1 min."
        )
        _run_wfv = _wfv_col2.button("▶ Run WFV", key="wfv_run_btn")

        if _run_wfv:
            _wfv_feat = [c for c in (st.session_state.feature_cols or []) if c in df.columns]
            with st.spinner("Running walk-forward validation…"):
                st.session_state.wfv_results = run_wfv(df, _wfv_feat)

        _wfv = st.session_state.wfv_results
        if _wfv:
            if "error" in _wfv:
                st.warning(_wfv["error"])
            else:
                _wc1, _wc2, _wc3, _wc4 = st.columns(4)
                _wc1.metric("Folds", _wfv["n_folds"])
                _wc2.metric("Mean Accuracy", f"{_wfv['mean_accuracy']:.1f}%")
                _wc3.metric("Std Accuracy",  f"{_wfv['std_accuracy']:.1f}%")
                _wc4.metric("Folds > 50%",  f"{_wfv['beat_50pct']} / {_wfv['n_folds']}")

                _wfv_fold_df = pd.DataFrame(_wfv["folds"])
                _wfv_fold_df["start"]    = _wfv_fold_df["start"].astype(str)
                _wfv_fold_df["end"]      = _wfv_fold_df["end"].astype(str)
                _wfv_fold_df["accuracy"] = _wfv_fold_df["accuracy"].round(1)
                st.dataframe(_wfv_fold_df.rename(columns={
                    "fold": "Fold", "start": "Test Start", "end": "Test End",
                    "accuracy": "Accuracy (%)", "n_predictions": "Predictions",
                }), hide_index=True, use_container_width=True)

                _wfv_fig = go.Figure()
                _wfv_fig.add_trace(go.Scatter(
                    x=[f["end"] for f in _wfv["folds"]],
                    y=[f["accuracy"] for f in _wfv["folds"]],
                    mode="lines+markers", name="Fold Accuracy",
                    line=dict(color="#00CC88", width=2),
                    marker=dict(size=8),
                ))
                _wfv_fig.add_hline(
                    y=50, line=dict(color="#FFA500", dash="dash", width=1),
                    annotation_text="50% baseline", annotation_position="bottom right",
                )
                _dark(_wfv_fig, height=280)
                _wfv_fig.update_layout(xaxis_title="Fold End Date", yaxis_title="Accuracy (%)")
                st.plotly_chart(_wfv_fig, width="stretch")

                _beat_pct = _wfv["beat_50pct"] / _wfv["n_folds"] * 100
                if _beat_pct >= 70:
                    _wfv_colour, _wfv_verdict = "success", "Strong"
                elif _beat_pct >= 50:
                    _wfv_colour, _wfv_verdict = "info", "Moderate"
                else:
                    _wfv_colour, _wfv_verdict = "warning", "Weak"
                _wfv_interp = (
                    f"**Walk-Forward Verdict — {_wfv_verdict}**: "
                    f"{_wfv['beat_50pct']} of {_wfv['n_folds']} folds beat random chance. "
                    f"Mean accuracy {_wfv['mean_accuracy']:.1f}% ± {_wfv['std_accuracy']:.1f}% "
                    f"(range {_wfv['min_accuracy']:.1f}%–{_wfv['max_accuracy']:.1f}%). "
                    + ("Model generalises well across time." if _beat_pct >= 70
                       else "Results are mixed — consider more data or feature review." if _beat_pct >= 50
                       else "Model struggles out-of-sample — retrain with a larger dataset.")
                )
                getattr(st, _wfv_colour)(_wfv_interp)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3D — Backtesting Deep-Dive
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("📈 Backtesting Deep-Dive", expanded=False):
    _stk = (st.session_state.clf_results or {}).get("Stacking", {})
    _bt_preds  = _stk.get("predictions")
    _bt_y      = _stk.get("y_test")
    _bt_probas = _stk.get("probabilities")
    _bt_dates  = _stk.get("test_dates")

    if _bt_preds is None or _bt_y is None or _bt_dates is None:
        st.info("Train the model to unlock the backtesting deep-dive.")
    else:
        _bt = pd.DataFrame({"y_true": _bt_y, "y_pred": _bt_preds}, index=_bt_dates)
        _bt.index = pd.to_datetime(_bt.index)
        if _bt_probas is not None:
            _bt["conf"] = np.array(_bt_probas).max(axis=1)
        _bt["correct"] = (_bt["y_true"] == _bt["y_pred"]).astype(int)

        # ── Panel 1: Summary stats ────────────────────────────────────────────
        st.markdown("**Summary Statistics**")
        _tot    = len(_bt)
        _acc    = _bt["correct"].mean() * 100
        _dir_bt = _bt[_bt["y_true"] != 1]
        _dir_acc = (_dir_bt["y_true"] == _dir_bt["y_pred"]).mean() * 100 if len(_dir_bt) else 0.0
        _bd1, _bd2, _bd3, _bd4 = st.columns(4)
        _bd1.metric("Total Predictions", _tot)
        _bd2.metric("Overall Accuracy",  f"{_acc:.1f}%")
        _bd3.metric("Directional Accuracy", f"{_dir_acc:.1f}%",
                    help="UP/DOWN only — excludes SIDEWAYS true labels")
        _bd4.metric("Test Period",
                    f"{_bt.index[0].strftime('%b %Y')} – {_bt.index[-1].strftime('%b %Y')}")

        # Class-wise
        _cls_cols = st.columns(3)
        for _ci, (_cls_int, _cls_lbl, _cls_clr) in enumerate(
            [(0, "DOWN", "#FF4B4B"), (1, "SIDEWAYS", "#888888"), (2, "UP", "#00CC88")]
        ):
            _cls_df  = _bt[_bt["y_true"] == _cls_int]
            _cls_acc = (_cls_df["y_true"] == _cls_df["y_pred"]).mean() * 100 if len(_cls_df) else 0.0
            _cls_cols[_ci].metric(f"{_cls_lbl} Accuracy", f"{_cls_acc:.1f}%",
                                  f"{len(_cls_df)} samples")

        st.divider()

        # ── Panel 2: Regime breakdown ─────────────────────────────────────────
        try:
            _reg_s = detect_regime(df).reindex(_bt.index)
            _bt["regime"] = _reg_s.values
            _reg_grp = (
                _bt.groupby("regime")["correct"]
                .agg(accuracy=lambda x: x.mean() * 100, count="count")
                .reset_index()
            )
            _reg_grp.columns = ["Regime", "Accuracy (%)", "Predictions"]
            _reg_grp["Accuracy (%)"] = _reg_grp["Accuracy (%)"].round(1)
            st.markdown("**Accuracy by Market Regime**")
            st.dataframe(_reg_grp, hide_index=True, use_container_width=True)
            st.divider()
        except Exception:
            pass

        # ── Panel 3: Rolling 30-day win rate ──────────────────────────────────
        st.markdown("**Rolling 30-Day Win Rate**")
        _roll_acc = _bt["correct"].rolling(30, min_periods=5).mean() * 100
        _roll_fig = go.Figure()
        _roll_fig.add_trace(go.Scatter(
            x=_bt.index, y=_roll_acc,
            mode="lines", name="30-Day Accuracy",
            line=dict(color="#00CC88", width=2),
            fill="tozeroy", fillcolor="rgba(0,204,136,0.10)",
        ))
        _roll_fig.add_hline(y=50, line=dict(color="#FFA500", dash="dash", width=1))
        _dark(_roll_fig, height=260)
        _roll_fig.update_layout(yaxis_title="Accuracy (%)", xaxis_title=None)
        st.plotly_chart(_roll_fig, width="stretch")

        # ── Panel 4: Confidence calibration ──────────────────────────────────
        if "conf" in _bt.columns:
            st.markdown("**Confidence Calibration**")
            _bins = [0, 0.35, 0.45, 0.55, 0.65, 0.75, 1.01]
            _bin_labels = ["<35%", "35-45%", "45-55%", "55-65%", "65-75%", ">75%"]
            _bt["conf_bin"] = pd.cut(_bt["conf"], bins=_bins, labels=_bin_labels, right=False)
            _cal = (
                _bt.groupby("conf_bin", observed=True)["correct"]
                .agg(accuracy=lambda x: x.mean() * 100, count="count")
                .reset_index()
            )
            _cal_fig = go.Figure()
            _cal_fig.add_trace(go.Bar(
                x=_cal["conf_bin"].astype(str), y=_cal["accuracy"],
                name="Actual Accuracy", marker_color="#00BFFF",
                text=_cal["count"].apply(lambda n: f"n={n}"),
                textposition="outside",
            ))
            _cal_fig.add_hline(y=50, line=dict(color="#FFA500", dash="dash", width=1))
            _dark(_cal_fig, height=280)
            _cal_fig.update_layout(xaxis_title="Model Confidence Bin", yaxis_title="Actual Accuracy (%)")
            st.plotly_chart(_cal_fig, width="stretch")

        # ── Panel 5: Monthly performance bar chart ────────────────────────────
        st.markdown("**Monthly Accuracy**")
        _bt["month"] = _bt.index.to_period("M")
        _mon = (
            _bt.groupby("month")["correct"]
            .agg(accuracy=lambda x: x.mean() * 100, count="count")
            .reset_index()
        )
        _mon["month"] = _mon["month"].astype(str)
        _mon_colors   = ["#00CC88" if a >= 50 else "#FF4B4B" for a in _mon["accuracy"]]
        _mon_fig = go.Figure(go.Bar(
            x=_mon["month"], y=_mon["accuracy"],
            marker_color=_mon_colors,
            text=_mon["count"].apply(lambda n: f"n={n}"),
            textposition="outside",
        ))
        _mon_fig.add_hline(y=50, line=dict(color="#FFA500", dash="dash", width=1))
        _dark(_mon_fig, height=300)
        _mon_fig.update_layout(xaxis_title="Month", yaxis_title="Accuracy (%)")
        st.plotly_chart(_mon_fig, width="stretch")

        # ── Auto-generated interpretation ─────────────────────────────────────
        _best_mon  = _mon.loc[_mon["accuracy"].idxmax(), "month"]
        _worst_mon = _mon.loc[_mon["accuracy"].idxmin(), "month"]
        _green_months = (_mon["accuracy"] >= 50).sum()
        st.info(
            f"**Backtest Insight**: Overall accuracy {_acc:.1f}% on {_tot} test predictions. "
            f"Directional accuracy (UP/DOWN only) is {_dir_acc:.1f}%. "
            f"{_green_months} of {len(_mon)} months beat the 50% baseline. "
            f"Best month: {_best_mon} — worst: {_worst_mon}. "
            + ("Model shows positive edge across most time periods." if _green_months / len(_mon) >= 0.6
               else "Accuracy is inconsistent — walk-forward validation recommended.")
        )
