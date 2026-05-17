"""
Gold AI — Historical Performance
Walk-forward validation summary + live backtest results.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import INITIAL_CAPITAL
from src.wfv_parser import parse_validation_log, latest_validation_log
from src.benchmarks import benchmark_metrics_table

st.set_page_config(
    page_title="Gold AI — Historical Performance",
    page_icon="📈",
    layout="wide",
)

DARK_BG  = "#0e1117"
GRID_CLR = "#1e2130"
_PLT = dict(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font=dict(color="white"))

def _dark(fig, height=380):
    fig.update_layout(**_PLT, height=height, legend=dict(orientation="h", y=1.02))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
    return fig

def _fmt_pf(v: float) -> str:
    if v > 999:
        return ">999"
    return f"{v:.3f}"

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

st.title("📈 Historical Performance")
st.caption("⚠️ NOT financial advice · Walk-forward validation 2019–2024 · Backtest ≠ live results")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Walk-Forward Validation (6 windows)
# ══════════════════════════════════════════════════════════════════════════════
st.header("Walk-Forward Validation — Gold AI v1 Baseline")
st.caption("6 independent test windows · Train = prior 4 years · Test = target year · 100 Optuna trials")

log_path = latest_validation_log(PROJECT_ROOT)
wfv = parse_validation_log(log_path) if log_path else {}

if not wfv.get("ok"):
    if wfv.get("partial"):
        st.warning("⏳ Validation run in progress — partial results not shown. Check back when complete.")
    else:
        st.info(
            "No walk-forward validation log found.  "
            "Run `python run_rolling_validation.py` to generate results."
        )
else:
    windows = wfv["windows"]
    agg     = wfv.get("aggregates", {})
    elapsed = wfv.get("elapsed_min", 0)

    # ── Aggregate KPIs ────────────────────────────────────────────────────────
    prof_n = agg.get("profitable_n", sum(1 for w in windows if w.get("return", 0) > 0))
    total_n = agg.get("total_n", len(windows))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Profitable Windows", f"{prof_n} / {total_n}",
              help="Windows with positive total return")
    k2.metric("Avg Return",
              f"{agg.get('avg_return', np.mean([w.get('return',0) for w in windows])):+.2f}%")
    k3.metric("Avg Sharpe",
              f"{agg.get('avg_sharpe', np.mean([w.get('sharpe',0) for w in windows])):.3f}")
    k4.metric("Avg Max Drawdown",
              f"{agg.get('avg_maxdd', np.mean([w.get('maxdd',0) for w in windows])):.2f}%")
    k5.metric("Avg Win Rate",
              f"{agg.get('avg_winrate', np.mean([w.get('winrate',0) for w in windows])):.1f}%")

    st.divider()

    # ── Per-window table ──────────────────────────────────────────────────────
    st.subheader("Per-Window Results")

    rows = []
    for w in windows:
        ret = w.get("return", 0)
        bh  = w.get("bh_return", 0)
        rows.append({
            "Year":       w.get("label", "—"),
            "Regime":     w.get("regime", "—"),
            "AI Return":  f"{ret:+.2f}%",
            "B&H Return": f"{bh:+.2f}%",
            "vs B&H":     f"{ret - bh:+.2f}pp",
            "Sharpe":     f"{w.get('sharpe', 0):.3f}",
            "Max DD":     f"{w.get('maxdd', 0):.2f}%",
            "Win Rate":   f"{w.get('winrate', 0):.1f}%",
            "Profit Factor": _fmt_pf(w.get("pf", 0)),
            "Trades":     int(w.get("trades", 0)),
        })

    wfv_df = pd.DataFrame(rows)

    def _color_return(val):
        try:
            v = float(val.replace("%", "").replace("+", "").replace("pp",""))
            return f"color: {'#00CC88' if v > 0 else '#FF4B4B' if v < 0 else '#888'}"
        except Exception:
            return ""

    style_cols = ["AI Return", "B&H Return", "vs B&H"]
    styler = wfv_df.style
    if hasattr(styler, "map"):
        styled = styler.map(_color_return, subset=style_cols)
    elif hasattr(styler, "applymap"):
        styled = styler.applymap(_color_return, subset=style_cols)
    else:
        styled = wfv_df
    st.dataframe(styled, hide_index=True, width="stretch")

    # ── Return bar chart ──────────────────────────────────────────────────────
    st.subheader("Return by Window")
    bar_df = pd.DataFrame({
        "Window":   [w.get("label", "") for w in windows],
        "AI Return": [w.get("return", 0) for w in windows],
        "B&H":       [w.get("bh_return", 0) for w in windows],
    })
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="AI Strategy",
        x=bar_df["Window"], y=bar_df["AI Return"],
        marker_color=[("#00CC88" if v >= 0 else "#FF4B4B") for v in bar_df["AI Return"]],
    ))
    fig_bar.add_trace(go.Scatter(
        name="Buy & Hold",
        x=bar_df["Window"], y=bar_df["B&H"],
        mode="lines+markers",
        line=dict(color="#FFA500", dash="dot", width=2),
        marker=dict(size=7),
    ))
    fig_bar.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    fig_bar.update_layout(**_PLT, height=320,
                          yaxis_title="Return (%)",
                          legend=dict(orientation="h", y=1.05))
    fig_bar.update_xaxes(showgrid=False)
    fig_bar.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
    st.plotly_chart(fig_bar, width="stretch")

    # ── Sharpe + MaxDD side by side ───────────────────────────────────────────
    sc1, sc2 = st.columns(2)
    with sc1:
        fig_sh = go.Figure(go.Bar(
            x=[w.get("label","") for w in windows],
            y=[w.get("sharpe", 0) for w in windows],
            marker_color=[("#00CC88" if v >= 0 else "#FF4B4B")
                          for v in [w.get("sharpe",0) for w in windows]],
            name="Sharpe",
        ))
        fig_sh.add_hline(y=0, line_color="rgba(255,255,255,0.2)")
        fig_sh.update_layout(**_PLT, height=260, title="Sharpe Ratio by Window",
                             yaxis_title="Sharpe", showlegend=False)
        fig_sh.update_xaxes(showgrid=False)
        fig_sh.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
        st.plotly_chart(fig_sh, width="stretch")

    with sc2:
        fig_dd = go.Figure(go.Bar(
            x=[w.get("label","") for w in windows],
            y=[abs(w.get("maxdd", 0)) for w in windows],
            marker_color="#FF4B4B",
            name="Max DD",
        ))
        fig_dd.update_layout(**_PLT, height=260, title="Max Drawdown % by Window",
                             yaxis_title="Max Drawdown (%)", showlegend=False)
        fig_dd.update_xaxes(showgrid=False)
        fig_dd.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
        st.plotly_chart(fig_dd, width="stretch")

    st.caption(
        f"Source: {os.path.basename(log_path)}  ·  "
        f"Elapsed: {elapsed:.0f} min  ·  "
        f"Gold AI v1 baseline (TUNE_ONCE=True, 100 trials, 117 features, bull_thr=0.42)"
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Live Backtest (current training period)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Live Backtest — Current Training Period")
st.caption("Single in-sample test period · Uses most recently trained model from Dashboard")

bt = st.session_state.get("backtest_results")
bm = st.session_state.get("benchmark_results")

if bt is None:
    st.info("No backtest results available. Train the model on the Dashboard page first.")
else:
    eq, bh_eq, trades_df, bt_m = bt

    bh_ret  = (float(bh_eq.iloc[-1]) - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    bh_days = len(bh_eq)
    bh_cagr = ((1 + bh_ret / 100) ** (252 / max(bh_days, 1)) - 1) * 100

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Return",
              f"{bt_m['Total Return (%)']:+.2f}%",
              f"B&H: {bh_ret:+.2f}%")
    k2.metric("CAGR",
              f"{bt_m['CAGR (%)']:+.2f}%",
              f"B&H: {bh_cagr:+.2f}%")
    k3.metric("Sharpe",   f"{bt_m['Sharpe Ratio']:.3f}")
    k4.metric("Max DD",   f"{bt_m['Max Drawdown (%)']:.2f}%", delta_color="inverse")
    k5.metric("Win Rate", f"{bt_m['Win Rate (%)']:.1f}%")

    k6, k7, k8, k9, k10 = st.columns(5)
    k6.metric("Sortino",       f"{bt_m['Sortino Ratio']:.3f}")
    k7.metric("Profit Factor", _fmt_pf(bt_m.get("Profit Factor", 0)))
    k8.metric("Expectancy",    f"${bt_m.get('Expectancy ($)', 0):,.2f}")
    k9.metric("Total Trades",  bt_m["Total Trades"])
    k10.metric("Avg Hold",     f"{bt_m.get('Avg Hold (days)', 0):.1f} days")

    st.divider()

    # ── Equity curve ──────────────────────────────────────────────────────────
    st.subheader("Equity Curve vs Buy & Hold")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values,
                                name="AI Strategy",
                                line=dict(color="#00CC88", width=2.5)))
    fig_eq.add_trace(go.Scatter(x=bh_eq.index, y=bh_eq.values,
                                name="Buy & Hold Gold",
                                line=dict(color="#FFA500", width=1.5, dash="dot")))
    fig_eq.update_layout(**_PLT, height=360, yaxis_title="Portfolio Value ($)",
                         legend=dict(orientation="h", y=1.02))
    fig_eq.update_xaxes(showgrid=False)
    fig_eq.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
    st.plotly_chart(fig_eq, width="stretch")

    # ── Drawdown ──────────────────────────────────────────────────────────────
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    fig_dd2 = go.Figure(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", fillcolor="rgba(255,75,75,0.18)",
        line=dict(color="#FF4B4B"), name="Drawdown",
    ))
    fig_dd2.update_layout(**_PLT, height=200, yaxis_title="Drawdown (%)")
    fig_dd2.update_xaxes(showgrid=False)
    fig_dd2.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
    st.plotly_chart(fig_dd2, width="stretch")

    # ── Benchmark comparison ──────────────────────────────────────────────────
    if bm is not None:
        st.subheader("Strategy vs Benchmarks")
        bm_tbl = benchmark_metrics_table(bm)
        if not bm_tbl.empty:
            ai_row = pd.DataFrame([{
                "Total Return (%)":  bt_m.get("Total Return (%)", 0),
                "CAGR (%)":          bt_m.get("CAGR (%)", 0),
                "Sharpe":            bt_m.get("Sharpe Ratio", 0),
                "Sortino":           bt_m.get("Sortino Ratio", 0),
                "Max DD (%)":        bt_m.get("Max Drawdown (%)", 0),
                "Win Rate (%)":      bt_m.get("Win Rate (%)", 0),
                "Profit Factor":     min(bt_m.get("Profit Factor", 0), 999),
                "Trades":            bt_m.get("Total Trades", 0),
            }], index=["AI Strategy"])
            combined = pd.concat([ai_row, bm_tbl])
            st.dataframe(
                combined.style
                    .highlight_max(subset=["Total Return (%)", "CAGR (%)", "Sharpe"],
                                   color="#1a3a2a")
                    .highlight_min(subset=["Max DD (%)"], color="#1a3a2a"),
                width="stretch",
            )

        # Multi-strategy equity curves
        bm_colors = {
            "RSI Strategy": "#00BFFF",
            "MA Crossover":  "#FFA500",
            "Breakout":      "#FF69B4",
            "Buy & Hold":    "#888888",
        }
        fig_bm = go.Figure()
        fig_bm.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name="AI Strategy", line=dict(color="#00CC88", width=2.5),
        ))
        for name, res in bm.items():
            if "equity" in res:
                fig_bm.add_trace(go.Scatter(
                    x=res["equity"].index, y=res["equity"].values,
                    name=name,
                    line=dict(color=bm_colors.get(name, "#888"), width=1.2, dash="dot"),
                ))
        fig_bm.update_layout(**_PLT, height=360, yaxis_title="Portfolio Value ($)",
                             legend=dict(orientation="h", y=1.02))
        fig_bm.update_xaxes(showgrid=False)
        fig_bm.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
        st.plotly_chart(fig_bm, width="stretch")

    # ── Trade log ─────────────────────────────────────────────────────────────
    st.subheader(f"Trade Log  ({len(trades_df)} trades)")
    if not trades_df.empty:
        def _color_pnl(v):
            try:
                return (f"color: {'#00CC88' if float(v) > 0 else '#FF4B4B' if float(v) < 0 else ''}")
            except Exception:
                return ""
        st.dataframe(
            trades_df.style.applymap(_color_pnl, subset=["PnL $"]),
            width="stretch",
        )
    else:
        st.write("No trades executed in the test period.")
