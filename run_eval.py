"""
Full retraining + evaluation script for the directional gold trading model.
Run with: python run_eval.py
"""

import sys, os, warnings, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from src.config import (
    N_TRIALS, DIRECTION_THRESHOLD, INITIAL_CAPITAL,
    TRAIN_YEARS, TEST_YEARS,
)
from src.data_loader import download_data, get_train_test_split
from src.features import add_features, get_feature_columns
from src.macro_loader import download_fred, add_macro_features
from src.regime import detect_regime, get_current_regime
from src.train import train_all_models, save_models
from src.backtest import run_backtest
from src.benchmarks import run_all_benchmarks, benchmark_metrics_table
from sklearn.metrics import recall_score
from src.calibration import (
    compute_brier_scores, compute_reliability_curves, optimize_threshold,
)
from src.signals import generate_latest_signal, SIGNAL_LABELS
from src.timeframes import get_4h_signal

# ─── helpers ──────────────────────────────────────────────────────────────────

SEP  = "=" * 68
SEP2 = "-" * 68

def h(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def sub(title):
    print(f"\n{SEP2}\n  {title}\n{SEP2}")


# ─── 1. Data ──────────────────────────────────────────────────────────────────
h("1 / 7  LOADING DATA")

df_raw  = download_data(force_refresh=True)
macro   = download_fred() if os.getenv("FRED_API_KEY") else pd.DataFrame()
df_feat = add_features(df_raw)
df      = add_macro_features(df_feat, macro)

train_df, test_df = get_train_test_split(df)

print(f"  Total bars    : {len(df):,}")
print(f"  Training bars : {len(train_df):,}  ({train_df.index[0].date()} → {train_df.index[-1].date()})")
print(f"  Test bars     : {len(test_df):,}  ({test_df.index[0].date()} → {test_df.index[-1].date()})")
print(f"  Features      : {len(get_feature_columns(df))}")

td = df["Target_Direction"].dropna()
print(f"\n  Target_Direction distribution (±{DIRECTION_THRESHOLD*100:.0f}%):")
print(f"    DOWN     (0) : {np.mean(td==0)*100:.1f}%  ({int((td==0).sum())} bars)")
print(f"    SIDEWAYS (1) : {np.mean(td==1)*100:.1f}%  ({int((td==1).sum())} bars)")
print(f"    UP       (2) : {np.mean(td==2)*100:.1f}%  ({int((td==2).sum())} bars)")


# ─── 2. Train ─────────────────────────────────────────────────────────────────
h("2 / 7  TRAINING ALL MODELS")
print(f"  Optuna trials per model : {N_TRIALS}")
print(f"  Class balancing         : XGBoost sample_weight · LGB balanced · CB class_weights")
print(f"  Stacking OOF folds      : 5 (TimeSeriesSplit)")
print(f"  Calibration             : isotonic · 3-fold")
print()

t0 = time.time()
log_msgs = []

def _log(msg):
    log_msgs.append(msg)
    print(f"  [{time.time()-t0:5.0f}s]  {msg}", flush=True)

reg_results, clf_results, feature_cols, stack_reg, stack_clf = train_all_models(
    train_df, test_df,
    n_trials=N_TRIALS,
    progress_callback=_log,
)

elapsed = time.time() - t0
print(f"\n  Training complete in {elapsed/60:.1f} min")
save_models(reg_results, clf_results, feature_cols, stack_reg, stack_clf)
print("  Models saved to models/models.pkl")


# ─── 3. Classification metrics ────────────────────────────────────────────────
h("3 / 7  DIRECTIONAL MODEL METRICS")

stk = clf_results["Stacking"]
y_true  = stk["y_test"]
y_pred  = stk["predictions"]
probas  = stk["probabilities"]
thr_res = stk.get("thresholds", {})

overall_acc  = accuracy_score(y_true, y_pred)
act_mask   = (y_pred != 1)
dir_acc    = float(np.mean(y_pred[act_mask] == y_true[act_mask])) if act_mask.sum() > 0 else 0.0
down_rec   = float(recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0))
up_rec     = float(recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0))

sub("Overall accuracy")
print(f"  Overall accuracy          : {overall_acc:.1%}")
print(f"  Directional accuracy (*)  : {dir_acc:.1%}  (* on predicted non-SIDEWAYS bars only)")
print(f"  DOWN recall               : {down_rec:.1%}")
print(f"  UP   recall               : {up_rec:.1%}")

sub("Per-class precision / recall / F1")
report = classification_report(
    y_true, y_pred,
    target_names=["DOWN", "SIDEWAYS", "UP"],
    digits=3, zero_division=0,
)
# Indent each line
for line in report.splitlines():
    print("  " + line)

sub("Optimised confidence thresholds (OOF)")
print(f"  UP   threshold  : {thr_res.get('threshold_up',  0.5):.3f}")
print(f"  DOWN threshold  : {thr_res.get('threshold_down', 0.5):.3f}")
print(f"  OOF dir. acc.   : {thr_res.get('best_score', 0):.1%}")

sub("Brier scores (lower = better calibrated)")
brier = compute_brier_scores(y_true, probas)
for cls, score in brier.items():
    bar = "#" * int(score * 40)
    print(f"  {cls:<10} : {score:.4f}  {bar}")

sub("Stacking lift vs individual models")
ind_accs = {n: clf_results[n]["metrics"]["Accuracy"]
            for n in clf_results if n not in ("Stacking", "_hyperparams")}
for name, acc in ind_accs.items():
    print(f"  {name:<10} : {acc:.1%}")
print(f"  {'Stacking':<10} : {overall_acc:.1%}  (+{(overall_acc - np.mean(list(ind_accs.values())))*100:.2f}pp vs avg)")


# ─── 4. Backtest ──────────────────────────────────────────────────────────────
h("4 / 7  STRATEGY BACKTEST")

# Use thresholded predictions for backtest
from src.calibration import apply_threshold
t_up   = thr_res.get("threshold_up",   0.50)
t_down = thr_res.get("threshold_down", 0.50)
thresh_preds = apply_threshold(probas, t_up, t_down)

test_dates = stk["test_dates"]
regime_s   = detect_regime(df).reindex(test_dates)

eq, bh, trades_df, bt_metrics = run_backtest(
    df, thresh_preds, test_dates,
    clf_probas=probas,
    regime_series=regime_s,
)

bh_ret  = (float(bh.iloc[-1]) / INITIAL_CAPITAL - 1) * 100
bh_days = len(bh)
bh_cagr = ((1 + bh_ret/100) ** (252 / max(bh_days, 1)) - 1) * 100

sub("AI Directional vs Buy & Hold")
metrics_order = [
    ("Total Return (%)",  "Total Return"),
    ("CAGR (%)",          "CAGR"),
    ("Sharpe Ratio",      "Sharpe"),
    ("Sortino Ratio",     "Sortino"),
    ("Calmar Ratio",      "Calmar"),
    ("Max Drawdown (%)",  "Max Drawdown"),
    ("Win Rate (%)",      "Win Rate"),
    ("Profit Factor",     "Profit Factor"),
    ("Expectancy ($)",    "Expectancy"),
    ("Total Trades",      "Total Trades"),
    ("Avg Hold (days)",   "Avg Hold"),
]
print(f"  {'Metric':<22} {'AI Model':>12}  {'Buy & Hold':>12}")
print(f"  {'-'*48}")
for key, label in metrics_order:
    ai_val = bt_metrics.get(key, "—")
    if key == "Total Return (%)":
        bh_val = f"{bh_ret:+.2f}%"
        ai_str = f"{ai_val:+.2f}%"
    elif key == "CAGR (%)":
        bh_val = f"{bh_cagr:+.2f}%"
        ai_str = f"{ai_val:+.2f}%"
    elif key in ("Max Drawdown (%)",):
        bh_dd = (bh / bh.cummax() - 1).min() * 100
        bh_val = f"{bh_dd:.2f}%"
        ai_str = f"{ai_val:.2f}%"
    else:
        bh_val = "—"
        if isinstance(ai_val, float):
            ai_str = f"{ai_val:.3f}"
        else:
            ai_str = str(ai_val)
    print(f"  {label:<22} {ai_str:>12}  {bh_val:>12}")

sub("Trade log summary")
if not trades_df.empty:
    wins   = trades_df[trades_df["PnL $"] > 0]
    losses = trades_df[trades_df["PnL $"] <= 0]
    print(f"  Total trades  : {len(trades_df)}")
    print(f"  Winners       : {len(wins)}  (avg ${wins['PnL $'].mean():,.0f})")
    print(f"  Losers        : {len(losses)}  (avg ${losses['PnL $'].mean():,.0f})")
    if "Side" in trades_df.columns:
        print(f"  Buy trades    : {(trades_df['Side']=='Buy').sum()}")
        print(f"  Sell trades   : {(trades_df['Side']=='Sell').sum()}")
    print(f"\n  Last 5 trades:")
    cols = ["Entry Date","Exit Date","Side","Entry $","Exit $","PnL $","Days Held","Exit Reason"]
    cols = [c for c in cols if c in trades_df.columns]
    print(trades_df[cols].tail(5).to_string(index=False))
else:
    print("  No trades executed during test period.")


# ─── 5. Benchmark comparison ──────────────────────────────────────────────────
h("5 / 7  BENCHMARK COMPARISON")

bm_results = run_all_benchmarks(df, test_dates, initial_capital=INITIAL_CAPITAL)

rows = []
# AI model row first
rows.append({
    "Strategy":       "AI Directional",
    "Total Ret (%)":  bt_metrics.get("Total Return (%)", 0),
    "CAGR (%)":       bt_metrics.get("CAGR (%)", 0),
    "Sharpe":         bt_metrics.get("Sharpe Ratio", 0),
    "Sortino":        bt_metrics.get("Sortino Ratio", 0),
    "Max DD (%)":     bt_metrics.get("Max Drawdown (%)", 0),
    "Win Rate (%)":   bt_metrics.get("Win Rate (%)", 0),
    "Prof Factor":    bt_metrics.get("Profit Factor", 0),
    "Expectancy ($)": bt_metrics.get("Expectancy ($)", 0),
    "Trades":         bt_metrics.get("Total Trades", 0),
})

for name, res in bm_results.items():
    if "error" in res:
        print(f"  {name}: ERROR — {res['error']}")
        continue
    m = res["metrics"]
    rows.append({
        "Strategy":       name,
        "Total Ret (%)":  m.get("Total Return (%)", 0),
        "CAGR (%)":       m.get("CAGR (%)", 0),
        "Sharpe":         m.get("Sharpe Ratio", 0),
        "Sortino":        m.get("Sortino Ratio", 0),
        "Max DD (%)":     m.get("Max Drawdown (%)", 0),
        "Win Rate (%)":   m.get("Win Rate (%)", 0),
        "Prof Factor":    m.get("Profit Factor", 0),
        "Expectancy ($)": m.get("Expectancy ($)", 0),
        "Trades":         m.get("Total Trades", 0),
    })

bm_df = pd.DataFrame(rows).set_index("Strategy")
# print with alignment
header = f"  {'Strategy':<18} {'Total Ret':>10} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'WinRate':>8} {'PF':>7} {'Exp $':>8} {'Trades':>7}"
print(header)
print("  " + "-" * 100)
for idx, row in bm_df.iterrows():
    marker = " <-- AI" if idx == "AI Directional" else ""
    print(
        f"  {idx:<18} "
        f"{row['Total Ret (%)']:>+9.1f}% "
        f"{row['CAGR (%)']:>+7.1f}% "
        f"{row['Sharpe']:>8.3f} "
        f"{row['Sortino']:>8.3f} "
        f"{row['Max DD (%)']:>7.1f}% "
        f"{row['Win Rate (%)']:>7.1f}% "
        f"{row['Prof Factor']:>7.3f} "
        f"{row['Expectancy ($)']:>8.0f} "
        f"{int(row['Trades']):>7}"
        f"{marker}"
    )


# ─── 6. Reliability curves (text summary) ─────────────────────────────────────
h("6 / 7  CALIBRATION SUMMARY")

rel = compute_reliability_curves(y_true, probas, n_bins=8)
print("  Reliability curve alignment (predicted prob → actual fraction positive):")
for cls_name, data in rel.items():
    if not data["mean_pred"]:
        continue
    print(f"\n  {cls_name}:")
    for mp, fp in zip(data["mean_pred"], data["fraction_pos"]):
        bar_len = int(fp * 30)
        ideal   = int(mp * 30)
        bar     = "#" * bar_len
        gap     = " " * abs(ideal - bar_len)
        direction = ("over" if bar_len > ideal else "under") + "-confident"
        print(f"    pred={mp:.2f}  actual={fp:.2f}  {'|' + bar:<33} {direction}")


# ─── 7. Latest live signal ────────────────────────────────────────────────────
h("7 / 7  LATEST LIVE SIGNAL")

regime_info = get_current_regime(df)
h4          = get_4h_signal("GC=F")

sig = generate_latest_signal(
    df, reg_results, clf_results, feature_cols,
    stack_reg=stack_reg, stack_clf=stack_clf,
    regime_int=regime_info["regime_int"],
    use_weekly_filter=True,
    use_4h_confirmation=False,
)

if sig:
    print(f"  Date           : {str(sig['latest_date'])[:10]}")
    print(f"  Current price  : ${sig['current_price']:,.2f}")
    print(f"  Predicted close: ${sig['predicted_price']:,.2f}" if sig.get("predicted_price") else "  Predicted close: N/A")
    print()
    print(f"  SIGNAL         : {sig['signal_emoji']}  {sig['signal_label']}")
    print(f"  Confidence     : {sig['confidence_pct']:.1f}%")
    if sig.get("proba_vec"):
        pv = sig["proba_vec"]
        print(f"  P(DOWN)        : {pv[0]*100:.1f}%")
        print(f"  P(SIDEWAYS)    : {pv[1]*100:.1f}%")
        print(f"  P(UP)          : {pv[2]*100:.1f}%")
    print()
    print(f"  Weekly trend   : {sig['weekly_trend']:+d}  ({'+1=uptrend, -1=downtrend, 0=neutral'})")
    print(f"  ATR            : ${sig['atr']:,.2f}  ({sig['atr']/sig['current_price']*100:.2f}% of price)")
    print(f"  Regime         : {regime_info['regime_label']}")
    if sig.get("filter_reason"):
        print(f"  Filter reason  : {sig['filter_reason']}")
    if sig["signal_int"] != 1:
        print(f"  Stop loss      : ${sig['stop_loss']:,.2f}")
        print(f"  Take profit    : ${sig['take_profit']:,.2f}")
        rr = abs(sig["take_profit"] - sig["current_price"]) / abs(sig["stop_loss"] - sig["current_price"])
        print(f"  R/R            : {rr:.1f} : 1")
    print()
    print(f"  4H EMA9/21     : {h4.get('ema9', '—')} / {h4.get('ema21', '—')}")
    print(f"  4H RSI(14)     : {h4.get('rsi14', '—')}")
    print(f"  4H Signal      : {SIGNAL_LABELS.get(h4.get('signal', 1), '—')}  (confidence {h4.get('confidence','—')})")

print(f"\n{SEP}")
print("  EVALUATION COMPLETE")
print(SEP)
