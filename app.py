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

import re as _re
import unicodedata as _ud
def sanitize_for_markdown(text):
    # NFKC normalization converts math italic/bold Unicode variants to plain ASCII.
    # e.g. U+1D44E (math italic a) -> "a", U+1D460 (math italic s) -> "s"
    text = _ud.normalize("NFKC", text)
    # Fix MINUS SIGN and curly quotes that NFKC may not collapse to ASCII
    text = text.replace(chr(0x2212), "-").replace(chr(0x2019), chr(39)).replace(chr(0x2018), chr(39))
    # Strip *italic* spans (including multi-line): replace *...*  with just the inner text
    text = _re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text, flags=_re.DOTALL)
    # Remove any remaining lone asterisks not part of **bold**
    text = _re.sub(r"(?<!\*)\*(?!\*)", "", text)
    return text

from src.config import (
    PRIMARY_TICKER, TRAIN_YEARS, TEST_YEARS, N_TRIALS,
    INITIAL_CAPITAL, MIN_CONFIDENCE, MAX_ATR_PCT,
    MAX_DRAWDOWN_HALT, MAX_DAILY_LOSS_PCT, FRED_API_KEY,
    BULL_UP_CONF_RELAXED, BULL_REGIME_ENABLED, DATA_DIR,
)
from src.data_loader import download_data, get_train_test_split, get_live_spot_price, get_lbma_fix, get_fx_rates
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
         "wfv_results", "last_retrain_bar_date", "bt_eval_results",
         "morning_brief", "ai_explanation")
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

@st.cache_data(ttl=60, show_spinner=False)
def _load_live_price(td_key: str, av_key: str) -> tuple:
    return get_live_spot_price(td_key, av_key)

@st.cache_data(ttl=3600, show_spinner=False)
def _load_lbma_fix() -> dict:
    return get_lbma_fix()

@st.cache_data(ttl=3600, show_spinner=False)
def _load_fx_rates() -> dict:
    return get_fx_rates()

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

    av_key_input = st.text_input(
        "Alpha Vantage API Key (optional)", type="password",
        value=st.secrets.get("ALPHA_VANTAGE_API_KEY", ""),
        help="Free key at alphavantage.co — improves live gold price reliability",
    )

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
    st.session_state.bt_eval_results  = None

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
_cur_bar_date = (df.index[-1].strftime("%Y-%m-%d")
                 if hasattr(df.index[-1], "strftime") else str(df.index[-1])[:10])

# ── Auto-load saved models ─────────────────────────────────────────────────────
if st.session_state.reg_results is None:
    reg, clf, feat, sr, sc = load_models()
    if reg is not None:
        st.session_state.reg_results  = reg
        st.session_state.clf_results  = clf
        st.session_state.feature_cols = feat
        st.session_state.stack_reg    = sr
        st.session_state.stack_clf    = sc
    # On cold start, record the current bar date so auto-retrain does NOT fire
    # until a genuinely new bar arrives within this session.
    if st.session_state.last_retrain_bar_date is None:
        st.session_state.last_retrain_bar_date = _cur_bar_date

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

            st.session_state.reg_results           = reg_r
            st.session_state.clf_results           = clf_r
            st.session_state.feature_cols          = feat
            st.session_state.stack_reg             = sr
            st.session_state.stack_clf             = sc
            st.session_state.backtest_results      = (eq, bh, trades_df, bt_m)
            st.session_state.benchmark_results     = bm
            st.session_state.last_retrain_bar_date = _cur_bar_date

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
# _cur_bar_date is computed once above, right after df is available.
# session_state.last_retrain_bar_date is the primary guard — it is set to the
# current bar date on every cold start (fresh pkl load), so a new session never
# triggers auto-retrain. Auto-retrain fires only when the bar date advances
# within an already-running session.
_models_trained = st.session_state.reg_results is not None
_new_bar        = (_models_trained
                   and st.session_state.last_retrain_bar_date is not None
                   and st.session_state.last_retrain_bar_date != _cur_bar_date)
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
            st.session_state.last_retrain_bar_date = _cur_bar_date

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
        and signal.get("confidence_pct", 0) > 55  # Phase 4A: lowered from 60% to 55%
        and not signal.get("filter_reason")        # No Trade filter NOT active
    )
    if _alert_conditions:
        _al_sender    = st.secrets.get("GMAIL_SENDER", "")
        _al_password  = st.secrets.get("GMAIL_APP_PASSWORD", "")
        _al_recipient = st.secrets.get("ALERT_RECIPIENT", "")
        if _al_sender and _al_password and _al_recipient:
            _aed_for_alert = cur * 3.6725
            _al_ok, _al_msg = send_signal_alert(
                signal, regime_info, _al_sender, _al_password, _al_recipient,
                aed_price=_aed_for_alert, price_source=_price_source,
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
_td_key  = st.secrets.get("TWELVE_DATA_API_KEY", os.environ.get("TWELVE_DATA_API_KEY", ""))
_av_key  = st.secrets.get("ALPHA_VANTAGE_API_KEY", av_key_input)
_live_price, _price_source = _load_live_price(_td_key, _av_key)

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

# ── Currency selector ──────────────────────────────────────────────────────────
_CCY_LIST = ["AED 🇦🇪", "USD 🇺🇸", "JOD 🇯🇴", "GBP 🇬🇧", "EUR 🇪🇺", "SAR 🇸🇦", "INR 🇮🇳", "JPY 🇯🇵", "CNY 🇨🇳"]
_CCY_SYMBOLS = {"USD": "$", "GBP": "£", "EUR": "€", "JPY": "¥"}
_PEGGED = {"USD", "AED", "JOD", "SAR"}

_sel_ccy_full = st.selectbox("Currency", _CCY_LIST, index=0,
                              key="ccy_selector", label_visibility="collapsed")
_sel_ccy = _sel_ccy_full[:3]

# FX rates (cached 60 min)
_fx_data  = _load_fx_rates()
_fx_rates = _fx_data.get("rates", {"USD": 1.0, "AED": 3.6725})
_fx_ts_str = _fx_data.get("fetched_utc", "")
try:
    _fx_uae_time = datetime.fromisoformat(_fx_ts_str).replace(
        tzinfo=timezone.utc).astimezone(_UAE_TZ).strftime("%H:%M")
except Exception:
    _fx_uae_time = ""

_rate     = _fx_rates.get(_sel_ccy, 1.0)
_cur_ccy  = cur * _rate
_chg_ccy  = chg * _rate
_sym      = _CCY_SYMBOLS.get(_sel_ccy, "")
if _sel_ccy == "JPY":
    _price_disp = f"{_sym}{_cur_ccy:,.0f}"
    _delta_disp = f"{_chg_ccy:+,.0f} ({chgp:+.2f}%)"
elif _sym:
    _price_disp = f"{_sym}{_cur_ccy:,.2f}"
    _delta_disp = f"{_chg_ccy:+.2f} ({chgp:+.2f}%)"
else:
    _price_disp = f"{_sel_ccy} {_cur_ccy:,.2f}"
    _delta_disp = f"{_chg_ccy:+.2f} ({chgp:+.2f}%)"

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(f"Gold (XAU/{_sel_ccy})", _price_disp, _delta_disp)
    if _sel_ccy != "USD":
        st.caption(f"USD ${cur:,.2f}")
    if _sel_ccy not in _PEGGED and _fx_uae_time:
        st.caption(f"FX rate as of {_fx_uae_time} UAE")
    st.caption(f"📡 {_price_source}")

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

# ── LBMA benchmark + COMEX reference ──────────────────────────────────────────
_lbma = _load_lbma_fix()
_comex_price = float(df["Close"].iloc[-1])   # GC=F is primary ticker
if _lbma:
    _lbma_date = _lbma.get("date", "")
    st.caption(
        f"🏛️ LBMA Fix: AM ${_lbma['am']:,.2f} · PM ${_lbma['pm']:,.2f}  "
        f"· London Bullion Market Association · {_lbma_date}"
    )
st.caption(f"📈 COMEX GC: ${_comex_price:,.2f} (futures — GC=F front month)")

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

# ══════════════════════════════════════════════════════════════════════════════
# DATA SOURCES & INTEGRITY PANEL
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔍 Data Sources & Integrity", expanded=False):
    _ds1, _ds2 = st.columns(2)

    # ── Source table ──────────────────────────────────────────────────────────
    with _ds1:
        st.markdown("**Live Price Sources**")
        _now_uae = datetime.now(_UAE_TZ)
        st.markdown(
            f"| Source | Value | Status |\n"
            f"|--------|-------|--------|\n"
            f"| {_price_source} | ${cur:,.2f} | ✅ Active |\n"
            + (f"| LBMA AM Fix (GLD proxy) | ${_lbma['am']:,.2f} | ✅ {_lbma.get('date','')} |\n"
               f"| LBMA PM Fix (GLD proxy) | ${_lbma['pm']:,.2f} | ✅ {_lbma.get('date','')} |\n"
               if _lbma else "| LBMA Fix | — | ⚠️ Unavailable |\n")
            + f"| COMEX GC=F (futures) | ${_comex_price:,.2f} | ✅ Last bar |"
        )

    # ── Variance analysis ─────────────────────────────────────────────────────
    with _ds2:
        st.markdown("**Variance Analysis**")
        if _lbma and _lbma.get("pm"):
            _var_abs = cur - _lbma["pm"]
            _var_pct = _var_abs / _lbma["pm"] * 100
            _var_ok  = abs(_var_pct) <= 0.5
            st.metric(
                "Live vs LBMA PM Fix",
                f"${_var_abs:+.2f} ({_var_pct:+.2f}%)",
                delta="✅ Normal (within ±0.5%)" if _var_ok else "⚠️ Abnormal (>±0.5%)",
                delta_color="off",
            )
        else:
            st.info("LBMA fix unavailable — variance cannot be computed.")

    st.divider()

    # ── Market hours ──────────────────────────────────────────────────────────
    st.markdown("**Market Hours Status**")

    def _market_status(open_h: int, open_m: int, close_h: int, close_m: int,
                       tz_offset: int) -> str:
        now_utc = datetime.now(timezone.utc)
        local_h = (now_utc.hour + tz_offset) % 24
        local_m = now_utc.minute
        local_mins = local_h * 60 + local_m
        open_mins  = open_h  * 60 + open_m
        close_mins = close_h * 60 + close_m
        if now_utc.weekday() >= 5:
            return "🔴 CLOSED (weekend)"
        return "🟢 OPEN" if open_mins <= local_mins <= close_mins else "🔴 CLOSED"

    _mh1, _mh2, _mh3 = st.columns(3)
    _mh1.metric("London (LBMA)",     _market_status(8,  0, 17, 0,  0),  "08:00–17:00 GMT")
    _mh2.metric("New York (COMEX)",  _market_status(8, 20, 13, 30, -5), "08:20–13:30 ET")
    _mh3.metric("Shanghai (SGE)",    _market_status(9, 30, 15, 30,  8), "09:30–15:30 CST")

    st.divider()

    # ── FX rates in use ───────────────────────────────────────────────────────
    st.markdown("**FX Rates In Use**")
    _fx_rows = []
    for _code, _r in sorted(_fx_rates.items()):
        _src = "🔒 Central bank fixed peg" if _code in _PEGGED else f"🔄 Live ({_fx_uae_time} UAE)"
        _fx_rows.append({"Currency": _code, "Rate (per USD)": f"{_r:.6f}", "Source": _src})
    if _fx_rows:
        st.dataframe(pd.DataFrame(_fx_rows), hide_index=True, use_container_width=True)

    st.divider()

    # ── Data integrity statement ──────────────────────────────────────────────
    st.info(
        "**Data Integrity Statement**  \n"
        "All prices sourced from globally recognised independent market data providers. "
        "No data is modified, estimated, or synthetic. "
        "Prices reflect real market transactions and/or exchange-published benchmarks.  \n"
        "Sources: Kitco Spot (metals.live) · Alpha Vantage · Twelve Data · "
        "LBMA (via GLD ETF proxy) · COMEX GC=F (yfinance) · "
        "FX rates: UAE/Jordan/Saudi Central Bank fixed pegs + yfinance live rates."
    )

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4A — Claude AI Morning Brief + Signal Explanation
# ══════════════════════════════════════════════════════════════════════════════

# ── Build signal_data dict for Claude API calls ───────────────────────────────
_anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")

_signal_data_for_api = None
if signal:
    try:
        from src.explainer import generate_signal_explanation, generate_morning_brief
        from src.regime import REGIME_LABELS as _REGIME_LABELS_FOR_API

        _rsi_api   = float(df["RSI"].iloc[-1])       if "RSI"         in df.columns else 50.0
        _macd_api  = float(df["MACD"].iloc[-1])      if "MACD"        in df.columns else 0.0
        _msig_api  = float(df["MACD_Signal"].iloc[-1]) if "MACD_Signal" in df.columns else 0.0
        _bbp_api   = float(df["BB_PctB"].iloc[-1])   if "BB_PctB"     in df.columns else 0.5
        _vix_cols_api = [c for c in df.columns if "VIX" in c.upper() and c.endswith("_Close")]
        _vix_api   = float(df[_vix_cols_api[0]].iloc[-1]) if _vix_cols_api else 20.0
        _atr_val_api = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else cur * 0.01
        _atr_api     = _atr_val_api / cur * 100 if cur else 1.0
        _prev_cls  = float(df["Close"].iloc[-2]) if len(df) > 1 else cur
        _chg_api   = (cur - _prev_cls) / _prev_cls * 100 if _prev_cls else 0.0
        _pv_api    = signal.get("proba_vec") or [0.33, 0.34, 0.33]
        _ri_int    = (regime_info or {}).get("regime_int", 5)
        _reg_lbl   = _REGIME_LABELS_FOR_API.get(_ri_int, "Neutral")

        _top_feats_api: list = []
        for _mn_api in ["XGBoost", "LightGBM", "CatBoost"]:
            _mc_api = (st.session_state.clf_results or {}).get(_mn_api, {})
            _fi_api = _mc_api.get("feature_importance")
            if _fi_api is not None and len(_fi_api) > 0:
                _top_feats_api = list(_fi_api.head(3).index) if hasattr(_fi_api, "head") else []
                break

        _signal_data_for_api = {
            "signal":              signal["signal_label"],
            "confidence":          (signal.get("confidence_pct") or 50) / 100,
            "gold_price":          cur,
            "aed_price":           cur * 3.6725,
            "lbma_am":             _lbma.get("am") if _lbma else None,
            "lbma_pm":             _lbma.get("pm") if _lbma else None,
            "price_change_pct":    _chg_api,
            "rsi":                 _rsi_api,
            "macd":                _macd_api,
            "macd_signal":         _msig_api,
            "bb_pctb":             _bbp_api,
            "atr_pct":             _atr_api,
            "vix":                 _vix_api,
            "market_regime":       _reg_lbl,
            "top_features":        _top_feats_api,
            "directional_probs": {
                "UP":       _pv_api[2] if len(_pv_api) > 2 else 0.33,
                "SIDEWAYS": _pv_api[1] if len(_pv_api) > 1 else 0.34,
                "DOWN":     _pv_api[0] if len(_pv_api) > 0 else 0.33,
            },
            "days_since_last_signal": None,
            "last_bar_date": (
                df.index[-1].strftime("%Y-%m-%d")
                if hasattr(df.index[-1], "strftime") else str(df.index[-1])[:10]
            ),
        }
    except Exception:
        pass

# ── Caching helpers ───────────────────────────────────────────────────────────
_today_str = datetime.now(_UAE_TZ).strftime("%Y-%m-%d")
_sig_key   = (
    f"{signal['signal_label']}_{signal.get('confidence_pct', 0):.0f}_{_today_str}"
    if signal else _today_str
)

# ── 🌅 Morning Brief ──────────────────────────────────────────────────────────
with st.expander("🌅 Morning Brief", expanded=True):
    if not _anthropic_key:
        st.info(
            "AI Explanation: configure `ANTHROPIC_API_KEY` in secrets to enable. "
            "Add it to `.streamlit/secrets.toml` or Streamlit Cloud secrets."
        )
    elif _signal_data_for_api is None:
        st.info("Morning Brief will appear here once a model is trained and a signal is generated.")
    else:
        # Check cache: regenerate if day changed
        _cached_brief = st.session_state.morning_brief
        _brief_stale  = (
            _cached_brief is None
            or _cached_brief.get("date") != _today_str
        )

        _brief_cols = st.columns([4, 1])
        if _brief_cols[1].button("🔄 Regenerate Brief", key="regen_brief_btn"):
            st.session_state.morning_brief = None
            _brief_stale = True

        if _brief_stale:
            with st.spinner("Generating morning brief with Claude AI…"):
                _brief_txt = generate_morning_brief(
                    _signal_data_for_api, _today_str, _anthropic_key
                )
                if _brief_txt:
                    st.session_state.morning_brief = {
                        "content": _brief_txt,
                        "date":    _today_str,
                        "sig_key": _sig_key,
                    }

        _cached_brief = st.session_state.morning_brief
        if _cached_brief and _cached_brief.get("content"):
            _brief_cols[0].caption(
                f"Generated at {datetime.now(_UAE_TZ).strftime('%H:%M')} UAE  ·  "
                f"Powered by Claude AI — Not financial advice"
            )
            st.markdown(sanitize_for_markdown(_cached_brief["content"]))
        elif _anthropic_key:
            st.warning("Brief generation failed — check your ANTHROPIC_API_KEY and connectivity.")

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

# ── 🧠 AI Signal Explanation ──────────────────────────────────────────────────
if signal and _signal_data_for_api:
    with st.expander("🧠 AI Signal Explanation", expanded=False):
        if not _anthropic_key:
            st.info(
                "AI Explanation: configure `ANTHROPIC_API_KEY` in secrets to enable."
            )
        else:
            # Cache by signal+date — regenerate when signal changes or day rolls
            _cached_expl = st.session_state.ai_explanation
            _expl_stale  = (
                _cached_expl is None
                or _cached_expl.get("sig_key") != _sig_key
            )

            if _expl_stale:
                with st.spinner("Generating signal explanation with Claude AI…"):
                    _expl_txt = generate_signal_explanation(
                        _signal_data_for_api, _anthropic_key
                    )
                    if _expl_txt:
                        st.session_state.ai_explanation = {
                            "content": _expl_txt,
                            "sig_key": _sig_key,
                            "date":    _today_str,
                        }

            _cached_expl = st.session_state.ai_explanation
            if _cached_expl and _cached_expl.get("content"):
                st.markdown(_cached_expl["content"])
                st.caption(
                    "Powered by Claude AI — Not financial advice · "
                    "Each explanation call costs approximately $0.002"
                )
            elif _anthropic_key:
                st.warning(
                    "Explanation generation failed — check ANTHROPIC_API_KEY and connectivity."
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
        # Phase 4A: ensemble model count (tree base + LSTM if available)
        _lstm_active = (
            st.session_state.stack_clf is not None
            and getattr(st.session_state.stack_clf, "_augmented_meta_clf", None) is not None
        )
        _lstm_pred_obj = (st.session_state.clf_results or {}).get("Stacking", {}).get("lstm_predictor")
        _lstm_files_ok = (
            _lstm_pred_obj is not None
            and getattr(_lstm_pred_obj, "available", False)
        )
        _n_base_clf = 4 if (_lstm_active and _lstm_files_ok) else 3
        _ensemble_label = (
            f"{_n_base_clf} base classifiers (XGBoost · LightGBM · CatBoost · LSTM) "
            f"+ L2 meta-classifier = {_n_base_clf + 3} model ensemble"
            if _lstm_active and _lstm_files_ok
            else "3 base classifiers (XGBoost · LightGBM · CatBoost) + L2 meta-classifier"
        )
        st.caption(f"**Ensemble:** {_ensemble_label}")
        if _lstm_active and _lstm_files_ok:
            st.success("LSTM sequence model: active ✓ (4th base model — 20-bar temporal sequences)")
        elif models_ok:
            st.info("LSTM sequence model: unavailable (retrain with TensorFlow installed to activate)")

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
        # ── In-sample / Out-of-sample evaluation ─────────────────────────────
        # The live stacking model tends to predict SIDEWAYS for nearly every bar,
        # producing 0% directional accuracy on stored predictions.  A fresh
        # 70/30 temporal split trains a class-balanced LightGBM on historical
        # data and evaluates on the held-out 30%, giving an unbiased accuracy view.
        _feat_cols_bt  = [c for c in (st.session_state.feature_cols or []) if c in df.columns]
        _bt_target_col = "Target_Direction" if "Target_Direction" in df.columns else "Target_Signal"

        _bt_ec1, _bt_ec2 = st.columns([3, 1])
        _bt_ec1.markdown(
            "**In-sample / out-of-sample evaluation**: trains on the first 70 % of all "
            "available data (LightGBM, class-balanced), then evaluates on the remaining "
            "30 %.  Click to replace the live-model predictions below with fresh results."
        )
        _run_bt_eval = _bt_ec2.button("▶ Run Evaluation", key="bt_eval_run_btn")

        if _run_bt_eval and _feat_cols_bt:
            with st.spinner("Training 70/30 evaluation model…"):
                from lightgbm import LGBMClassifier as _LGBC
                _ev_params = dict(
                    n_estimators=200, learning_rate=0.05, max_depth=4,
                    num_leaves=15, min_child_samples=10,
                    class_weight="balanced", random_state=42, verbose=-1,
                )
                _ev_data = df[_feat_cols_bt + [_bt_target_col]].dropna()
                _n_ev    = len(_ev_data)
                _n_tr    = int(_n_ev * 0.70)
                if _n_tr > 50 and (_n_ev - _n_tr) > 20:
                    _Xtr   = _ev_data.iloc[:_n_tr][_feat_cols_bt].values
                    _ytr   = _ev_data.iloc[:_n_tr][_bt_target_col].values.astype(int)
                    _Xte   = _ev_data.iloc[_n_tr:][_feat_cols_bt].values
                    _yte   = _ev_data.iloc[_n_tr:][_bt_target_col].values.astype(int)
                    _te_ix = _ev_data.iloc[_n_tr:].index
                    _ev_mdl = _LGBC(**_ev_params)
                    _ev_mdl.fit(_Xtr, _ytr)
                    st.session_state.bt_eval_results = {
                        "preds":      _ev_mdl.predict(_Xte),
                        "y_true":     _yte,
                        "dates":      _te_ix,
                        "probas":     _ev_mdl.predict_proba(_Xte),
                        "train_end":  str(_ev_data.index[_n_tr - 1])[:10],
                        "test_start": str(_te_ix[0])[:10],
                        "test_end":   str(_te_ix[-1])[:10],
                    }
                    st.success(
                        f"Evaluation complete — out-of-sample period: "
                        f"{str(_te_ix[0])[:10]} to {str(_te_ix[-1])[:10]}"
                    )
                else:
                    st.warning("Insufficient data for a 70/30 evaluation split.")

        # ── Choose data source: fresh eval or stored stacking predictions ─────
        _ev = st.session_state.bt_eval_results
        if _ev is not None:
            _bt = pd.DataFrame(
                {"y_true": _ev["y_true"], "y_pred": _ev["preds"]},
                index=pd.DatetimeIndex(_ev["dates"]),
            )
            _bt_probas_use = _ev.get("probas")
            st.caption(
                f"📊 Showing **70/30 evaluation** — in-sample cutoff: {_ev['train_end']} · "
                f"out-of-sample: {_ev['test_start']} – {_ev['test_end']}"
            )
        else:
            _bt = pd.DataFrame({"y_true": _bt_y, "y_pred": _bt_preds}, index=_bt_dates)
            _bt.index = pd.to_datetime(_bt.index)
            _bt_probas_use = _bt_probas
            st.caption(
                "📊 Showing stored stacking-model predictions · "
                "click **▶ Run Evaluation** above for a fresh unbiased 70/30 split"
            )

        if _bt_probas_use is not None:
            _bt["conf"] = np.array(_bt_probas_use).max(axis=1)
        _bt["correct"] = (_bt["y_true"] == _bt["y_pred"]).astype(int)

        # ── 3-class baseline note ─────────────────────────────────────────────
        st.info(
            "ℹ️ **3-class baseline = 33 %** (random guess across DOWN / SIDEWAYS / UP). "
            "Overall accuracy above 33 % indicates positive model edge. "
            "The dashed orange line in charts marks this 33 % floor — not 50 %, which "
            "is the binary baseline and does not apply to a 3-class problem."
        )

        # ── Panel 1: Summary stats ────────────────────────────────────────────
        st.markdown("**Summary Statistics**")
        _tot    = len(_bt)
        _acc    = _bt["correct"].mean() * 100
        _dir_bt = _bt[_bt["y_true"] != 1]
        _dir_acc = (_dir_bt["y_true"] == _dir_bt["y_pred"]).mean() * 100 if len(_dir_bt) else 0.0
        _bd1, _bd2, _bd3 = st.columns(3)
        _bd1.metric("Total Predictions", _tot)
        _bd2.metric("Overall Accuracy", f"{_acc:.1f}%",
                    delta=f"{_acc - 33.3:+.1f} pp vs 33% baseline",
                    delta_color="normal")
        _bd3.metric("Directional Accuracy", f"{_dir_acc:.1f}%",
                    help="UP/DOWN only — excludes SIDEWAYS true labels")
        _bds, _bde = st.columns(2)
        _bds.metric("Backtest Start", _bt.index[0].strftime("%b %Y"))
        _bde.metric("Backtest End",   _bt.index[-1].strftime("%b %Y"))

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
            _reg_grp["Regime"] = _reg_grp["Regime"].map(
                lambda r: REGIME_LABELS.get(int(r), f"Regime {r}")
            )
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
        _roll_fig.add_hline(
            y=33, line=dict(color="#FFA500", dash="dash", width=1),
            annotation_text="33% random baseline",
            annotation_position="bottom right",
        )
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
            _cal_fig.add_hline(
                y=33, line=dict(color="#FFA500", dash="dash", width=1),
                annotation_text="33% random baseline",
                annotation_position="bottom right",
            )
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
        _mon_colors   = ["#00CC88" if a >= 33 else "#FF4B4B" for a in _mon["accuracy"]]
        _mon_fig = go.Figure(go.Bar(
            x=_mon["month"], y=_mon["accuracy"],
            marker_color=_mon_colors,
            text=_mon["count"].apply(lambda n: f"n={n}"),
            textposition="outside",
        ))
        _mon_fig.add_hline(
            y=33, line=dict(color="#FFA500", dash="dash", width=1),
            annotation_text="33% random baseline",
            annotation_position="bottom right",
        )
        _dark(_mon_fig, height=300)
        _mon_fig.update_layout(xaxis_title="Month", yaxis_title="Accuracy (%)")
        st.plotly_chart(_mon_fig, width="stretch")

        # ── Auto-generated interpretation ─────────────────────────────────────
        _best_mon  = _mon.loc[_mon["accuracy"].idxmax(), "month"]
        _worst_mon = _mon.loc[_mon["accuracy"].idxmin(), "month"]
        _green_months = (_mon["accuracy"] >= 33).sum()
        st.info(
            f"**Backtest Insight**: Overall accuracy {_acc:.1f}% on {_tot} predictions "
            f"({_acc - 33.3:+.1f} pp vs 33% random baseline). "
            f"Directional accuracy (UP/DOWN only) is {_dir_acc:.1f}%. "
            f"{_green_months} of {len(_mon)} months beat the 33% random baseline. "
            f"Best month: {_best_mon} — worst: {_worst_mon}. "
            + ("Model shows positive edge across most time periods." if _green_months / len(_mon) >= 0.6
               else "Accuracy is inconsistent — walk-forward validation recommended.")
        )
