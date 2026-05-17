"""
Threshold sweep: runs full retrain + eval at ±0.7%.

Changes vs prior run:
  - Trend regime filter: DOWN signals in bull regime require P(DOWN)>=0.45
  - DOWN threshold grid raised to 0.35–0.45 (was 0.25–0.42)
  - UP threshold grid unchanged (0.33–0.65)
"""
import sys, os, warnings, time, importlib
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, recall_score

# Pre-import modules we'll reload per iteration
import src.config as _cfg

from src.data_loader import download_data, get_train_test_split
from src.macro_loader import download_fred, add_macro_features

SEP  = "=" * 68
SEP2 = "-" * 68

THRESHOLDS_TO_TEST = [0.007]   # ±0.7% — confirmed winner from prior sweep

# ── Load raw data once ────────────────────────────────────────────────────────
print(f"\n{SEP}\n  PRE-LOADING RAW DATA\n{SEP}")
df_raw = download_data(force_refresh=False)
macro  = download_fred() if os.getenv("FRED_API_KEY") else pd.DataFrame()

summary_rows = []

for thr in THRESHOLDS_TO_TEST:
    pct = f"±{thr*100:.1f}%"
    print(f"\n\n{'#'*68}")
    print(f"  RUNNING FULL RETRAIN   threshold = {pct}")
    print(f"{'#'*68}\n")

    # ── Patch config ──────────────────────────────────────────────────────────
    _cfg.DIRECTION_THRESHOLD = thr
    # Reload features so Target_Direction is recomputed with new threshold
    import src.features as _feat
    importlib.reload(_feat)
    from src.features import add_features, get_feature_columns

    from src.train import train_all_models, save_models
    from src.backtest import run_backtest
    from src.benchmarks import run_all_benchmarks
    from src.calibration import (
        compute_brier_scores, optimize_threshold, apply_threshold,
    )
    from src.signals import generate_latest_signal, SIGNAL_LABELS
    from src.regime import detect_regime, get_current_regime
    from src.timeframes import get_4h_signal

    # ── Features + split ─────────────────────────────────────────────────────
    df_feat = add_features(df_raw)
    df      = add_macro_features(df_feat, macro)
    train_df, test_df = get_train_test_split(df)

    feat_cols = get_feature_columns(df)
    td = df["Target_Direction"].dropna()
    n_bars = len(df)

    print(f"  Target_Direction distribution ({pct}):")
    for cls, label in [(0,"DOWN"),(1,"SIDEWAYS"),(2,"UP")]:
        n = int((td==cls).sum())
        print(f"    {label:<10}: {n} bars ({n/len(td)*100:.1f}%)")

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    log_msgs = []
    def _log(msg):
        log_msgs.append(msg)
        print(f"  [{time.time()-t0:5.0f}s]  {msg}", flush=True)

    from src.config import N_TRIALS
    reg_results, clf_results, feature_cols, stack_reg, stack_clf = train_all_models(
        train_df, test_df, n_trials=N_TRIALS, progress_callback=_log,
    )
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed/60:.1f} min")

    # ── Classification metrics ─────────────────────────────────────────────────
    stk      = clf_results["Stacking"]
    y_true   = stk["y_test"]
    y_pred   = stk["predictions"]
    probas   = stk["probabilities"]
    thr_res  = stk.get("thresholds", {})

    t_up   = thr_res.get("threshold_up",   0.50)
    t_down = thr_res.get("threshold_down", 0.50)
    thresh_preds = apply_threshold(probas, t_up, t_down)

    overall_acc = accuracy_score(y_true, thresh_preds)
    act_mask    = thresh_preds != 1
    dir_acc = (float(np.mean(thresh_preds[act_mask] == y_true[act_mask]))
               if act_mask.sum() > 0 else 0.0)
    down_rec = float(recall_score(y_true, thresh_preds, labels=[0],
                                  average="macro", zero_division=0))

    print(f"\n  Overall accuracy     : {overall_acc:.1%}")
    print(f"  Directional accuracy : {dir_acc:.1%}")
    print(f"  DOWN recall          : {down_rec:.1%}")
    print(f"  UP  threshold        : {t_up:.3f}")
    print(f"  DOWN threshold       : {t_down:.3f}")
    print(f"  OOF dir acc          : {thr_res.get('best_score',0):.1%}")

    rep = classification_report(y_true, thresh_preds,
                                target_names=["DOWN","SIDEWAYS","UP"],
                                digits=3, zero_division=0)
    print()
    for line in rep.splitlines():
        print("  " + line)

    # ── Backtest ───────────────────────────────────────────────────────────────
    test_dates = stk["test_dates"]
    regime_s   = detect_regime(df).reindex(test_dates)

    eq, bh, trades_df, bt_metrics = run_backtest(
        df, thresh_preds, test_dates,
        clf_probas=probas,
        regime_series=regime_s,
    )

    print(f"\n  Backtest results:")
    for key in ["Total Return (%)","CAGR (%)","Sharpe Ratio","Sortino Ratio",
                "Calmar Ratio","Max Drawdown (%)","Win Rate (%)","Profit Factor",
                "Expectancy ($)","Total Trades","Avg Hold"]:
        val = bt_metrics.get(key, "—")
        print(f"    {key:<22}: {val}")

    if not trades_df.empty:
        buy_n  = int((trades_df.get("Side","") == "Buy").sum())  if "Side" in trades_df.columns else "—"
        sell_n = int((trades_df.get("Side","") == "Sell").sum()) if "Side" in trades_df.columns else "—"
        print(f"    Buy  trades         : {buy_n}")
        print(f"    Sell trades         : {sell_n}")

    # ── Benchmarks ────────────────────────────────────────────────────────────
    from src.config import INITIAL_CAPITAL
    bm_results = run_all_benchmarks(df, test_dates, initial_capital=INITIAL_CAPITAL)

    print(f"\n  Benchmark comparison:")
    hdr = f"  {'Strategy':<18} {'TotRet':>9} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'WinRate':>8} {'PF':>7} {'Trades':>7}"
    print(hdr)
    print("  " + "-" * 78)

    def _bm_row(label, m):
        print(f"  {label:<18} "
              f"{m.get('Total Return (%)',0):>+8.1f}% "
              f"{m.get('Sharpe Ratio',0):>8.3f} "
              f"{m.get('Sortino Ratio',0):>8.3f} "
              f"{m.get('Max Drawdown (%)',0):>7.1f}% "
              f"{m.get('Win Rate (%)',0):>7.1f}% "
              f"{m.get('Profit Factor',0):>7.3f} "
              f"{int(m.get('Total Trades',0)):>7}")

    _bm_row(f"AI {pct}", bt_metrics)
    for name, res in bm_results.items():
        if "error" not in res:
            _bm_row(name, res["metrics"])

    # ── Collect summary ───────────────────────────────────────────────────────
    summary_rows.append({
        "Threshold":       pct,
        "Overall Acc":     f"{overall_acc:.1%}",
        "Dir Acc":         f"{dir_acc:.1%}",
        "DOWN Recall":     f"{down_rec:.1%}",
        "Trades":          bt_metrics.get("Total Trades", 0),
        "Sharpe":          bt_metrics.get("Sharpe Ratio", 0),
        "Max DD (%)":      bt_metrics.get("Max Drawdown (%)", 0),
        "Profit Factor":   bt_metrics.get("Profit Factor", 0),
        "Total Ret (%)":   bt_metrics.get("Total Return (%)", 0),
        "t_up":            t_up,
        "t_down":          t_down,
        "OOF Dir Acc":     thr_res.get("best_score", 0),
    })

    # Save models for the threshold that performs better (overwritten each iter;
    # final save will be the last threshold — user can re-run chosen one)
    save_models(reg_results, clf_results, feature_cols, stack_reg, stack_clf)

# ── Side-by-side summary ──────────────────────────────────────────────────────
print(f"\n\n{'#'*68}")
print(f"  SIDE-BY-SIDE COMPARISON")
print(f"{'#'*68}\n")

keys = ["Threshold","Overall Acc","Dir Acc","DOWN Recall","Trades","Sharpe",
        "Max DD (%)","Profit Factor","Total Ret (%)","t_up","t_down","OOF Dir Acc"]
for k in keys:
    vals = "   ".join(f"{r[k]}" if not isinstance(r[k], float) else f"{r[k]:.4f}"
                      for r in summary_rows)
    print(f"  {k:<20}: {vals}")

print(f"\n{'#'*68}\n  SWEEP COMPLETE\n{'#'*68}\n")
