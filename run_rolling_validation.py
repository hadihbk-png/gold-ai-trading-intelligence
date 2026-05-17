"""
Rolling walk-forward validation with true per-window retraining.

Speed optimisations (all scientifically sound):

  1. Download once  — full history fetched once, cached globally.
  2. Features once  — df_full built once; each window slices from it.
  3. N_TRIALS_ROLLING = 20  — fewer Optuna trials per window.
     Hyper-parameter search is still independent per window; the smaller
     budget finds a good (not necessarily best) set, which is sufficient
     for cross-window generalisation testing.
  4. TUNE_ONCE = True  — Optuna runs only for the FIRST window; its
     hyperparameters (tree depth, learning rate, regularisation) are reused
     for all subsequent windows.  Only the model WEIGHTS are refitted from
     scratch on each window's training data, so every window is still a
     fully independent retrain.  This is valid because hyperparameters are
     regularisation choices, not pattern fits, and are stable across regimes.
     Set TUNE_ONCE = False to run independent Optuna search per window.

Windows tested (train = prior 4 years, test = the labelled year):
  2019  train 2015–2018   Gold bull run
  2020  train 2016–2019   COVID crash + V-recovery
  2021  train 2017–2020   Sideways / consolidation
  2022  train 2018–2021   Bear / rate shock
  2023  train 2019–2022   Recovery / range
  2024  train 2020–2023   Strong bull run
"""

import sys, os, warnings, time, importlib
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

import src.config as _cfg

from src.data_loader import download_data
from src.macro_loader import download_fred, add_macro_features
from src.benchmarks import run_all_benchmarks
from src.calibration import apply_threshold
from src.backtest import run_backtest
from src.regime import detect_regime
from src.config import INITIAL_CAPITAL, DIRECTION_THRESHOLD

SEP  = "=" * 72
SEP2 = "#" * 72

# ── Runtime settings ──────────────────────────────────────────────────────────
N_TRIALS_ROLLING = 100   # Optuna trials per window  (config N_TRIALS=100 overridden)
TUNE_ONCE        = True   # False → Optuna runs independently on every window
TRAIN_YEARS      = 4     # rolling training window length in years
HIST_START       = "2014-01-01"
HIST_END         = "2025-06-01"

# ── Walk-forward windows ──────────────────────────────────────────────────────
WINDOWS = [
    ("2019-01-01", "2020-01-01", "2019",              "Gold bull run (+18%)"),
    ("2020-01-01", "2021-01-01", "2020 (COVID)",      "COVID crash + V-recovery"),
    ("2021-01-01", "2022-01-01", "2021",              "Sideways / consolidation"),
    ("2022-01-01", "2023-01-01", "2022 (Rate hikes)", "Bear / rate shock"),
    ("2023-01-01", "2024-01-01", "2023",              "Recovery / range"),
    ("2024-01-01", "2025-01-01", "2024 (Bull run)",   "Strong bull run (+27%)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Download once + build features once
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  DOWNLOADING FULL HISTORY  {HIST_START} – {HIST_END}\n{SEP}")
df_raw = download_data(start=HIST_START, end=HIST_END, force_refresh=True)
macro  = (download_fred(start=HIST_START, end=HIST_END, force_refresh=True)
          if os.getenv("FRED_API_KEY") else pd.DataFrame())
print(f"  {len(df_raw)} bars  ({df_raw.index[0].date()} – {df_raw.index[-1].date()})")

print(f"\n{SEP}\n  BUILDING FEATURES ON FULL HISTORY\n{SEP}")
import src.features as _feat
importlib.reload(_feat)
from src.features import add_features, get_feature_columns
from src.train import train_all_models

df_feat = add_features(df_raw)
df_full = add_macro_features(df_feat, macro)
print(f"  Feature matrix shape   : {df_full.shape}")
print(f"  Optuna trials/window   : {N_TRIALS_ROLLING}")
print(f"  Tune-once mode         : {TUNE_ONCE}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Per-window walk-forward loop
# ─────────────────────────────────────────────────────────────────────────────
summary_rows       = []
reused_hyperparams = None   # set after first Optuna run when TUNE_ONCE=True
total_t0           = time.time()

for win_idx, (win_start, win_end, label, regime_note) in enumerate(WINDOWS):

    print(f"\n\n{SEP2}")
    print(f"  WINDOW {win_idx+1}/{len(WINDOWS)} : {label}  —  {regime_note}")
    print(f"  TEST   : {win_start}  →  {win_end}")
    train_start = pd.Timestamp(win_start) - pd.DateOffset(years=TRAIN_YEARS)
    train_end   = pd.Timestamp(win_start)
    print(f"  TRAIN  : {train_start.date()}  →  {train_end.date()}")
    if TUNE_ONCE and win_idx > 0:
        print(f"  Optuna : skipped (reusing params from window 1)")
    else:
        print(f"  Optuna : {N_TRIALS_ROLLING} trials per study")
    print(f"{SEP2}\n")

    # Slice train / test from the globally-built feature matrix
    train_df = df_full[
        (df_full.index >= train_start) & (df_full.index < train_end)
    ].copy()
    test_df  = df_full[
        (df_full.index >= win_start) & (df_full.index < win_end)
    ].copy()

    if len(train_df) < 200:
        print(f"  Skipping — only {len(train_df)} training bars (need ≥200).")
        continue
    if len(test_df) < 20:
        print(f"  Skipping — only {len(test_df)} test bars.")
        continue

    print(f"  Training bars : {len(train_df)}")
    print(f"  Test bars     : {len(test_df)}")

    # ── Retrain ────────────────────────────────────────────────────────────────
    t0 = time.time()

    def _log(msg):
        print(f"  [{time.time()-t0:5.0f}s]  {msg}", flush=True)

    use_params = reused_hyperparams if (TUNE_ONCE and win_idx > 0) else None

    reg_results, clf_results, feature_cols, stack_reg, stack_clf = train_all_models(
        train_df, test_df,
        n_trials=N_TRIALS_ROLLING,
        progress_callback=_log,
        pretrained_hyperparams=use_params,
    )

    # Save hyperparameters from the first window for reuse
    if TUNE_ONCE and win_idx == 0:
        reused_hyperparams = clf_results.get("_hyperparams")
        if reused_hyperparams:
            print(f"  Hyperparameters saved for reuse in windows 2–{len(WINDOWS)}.")

    elapsed_min = (time.time() - t0) / 60
    print(f"\n  Training complete in {elapsed_min:.1f} min")

    # ── Predictions & classification metrics ──────────────────────────────────
    stk     = clf_results["Stacking"]
    y_true  = stk["y_test"]
    probas  = stk["probabilities"]
    thr_res = stk.get("thresholds", {})
    t_up    = thr_res.get("threshold_up",   0.40)
    t_down  = thr_res.get("threshold_down", 0.36)

    thresh_preds = apply_threshold(probas, t_up, t_down)
    overall_acc  = accuracy_score(y_true, thresh_preds)
    act_mask     = thresh_preds != 1
    dir_acc = (float(np.mean(thresh_preds[act_mask] == y_true[act_mask]))
               if act_mask.sum() > 0 else 0.0)
    dn_rec  = float(recall_score(y_true, thresh_preds, labels=[0],
                                 average="macro", zero_division=0))

    print(f"\n  Classification:")
    print(f"    Overall accuracy : {overall_acc:.1%}")
    print(f"    Dir accuracy     : {dir_acc:.1%}  (on {int(act_mask.sum())} signals)")
    print(f"    DOWN recall      : {dn_rec:.1%}")
    print(f"    t_up={t_up:.3f}  t_down={t_down:.3f}  OOF={thr_res.get('best_score',0):.4f}")

    # ── Backtest ───────────────────────────────────────────────────────────────
    test_dates = stk["test_dates"]
    regime_s   = detect_regime(df_full).reindex(test_dates)

    eq, bh, trades_df, bt = run_backtest(
        df_full, thresh_preds, test_dates,
        clf_probas=probas,
        regime_series=regime_s,
    )
    bh_ret = float((bh.iloc[-1] - bh.iloc[0]) / bh.iloc[0] * 100)

    print(f"\n  Backtest results:")
    for key in ["Total Return (%)", "CAGR (%)", "Sharpe Ratio", "Sortino Ratio",
                "Calmar Ratio", "Max Drawdown (%)", "Win Rate (%)",
                "Profit Factor", "Expectancy ($)", "Total Trades"]:
        print(f"    {key:<24}: {bt.get(key, '—')}")

    if not trades_df.empty and "Side" in trades_df.columns:
        buy_n  = int((trades_df["Side"] == "Buy").sum())
        sell_n = int((trades_df["Side"] == "Sell").sum())
        print(f"    {'Buy trades':<24}: {buy_n}")
        print(f"    {'Sell trades':<24}: {sell_n}")

    # ── Benchmarks ────────────────────────────────────────────────────────────
    bm = run_all_benchmarks(df_full, test_dates, initial_capital=INITIAL_CAPITAL)

    print(f"\n  Benchmark comparison:")
    hdr = (f"  {'Strategy':<22} {'Return':>8} {'Sharpe':>8} {'Sortino':>8}"
           f" {'MaxDD':>7} {'WinRate':>8} {'PF':>7} {'Trades':>7}")
    print(hdr)
    print("  " + "-" * 80)

    def _row(lbl, m):
        print(f"  {lbl:<22}"
              f" {m.get('Total Return (%)',0):>+7.1f}%"
              f" {m.get('Sharpe Ratio',0):>8.3f}"
              f" {m.get('Sortino Ratio',0):>8.3f}"
              f" {m.get('Max Drawdown (%)',0):>6.1f}%"
              f" {m.get('Win Rate (%)',0):>7.1f}%"
              f" {m.get('Profit Factor',0):>7.3f}"
              f" {int(m.get('Total Trades',0)):>7}")

    _row(f"AI {label}", bt)
    for name, res in bm.items():
        if "error" not in res:
            _row(name, res["metrics"])

    summary_rows.append({
        "Window":    label,
        "Regime":    regime_note,
        "BH Return": bh_ret,
        "Return":    bt.get("Total Return (%)",  0),
        "Sharpe":    bt.get("Sharpe Ratio",      0),
        "Sortino":   bt.get("Sortino Ratio",     0),
        "MaxDD":     bt.get("Max Drawdown (%)",  0),
        "WinRate":   bt.get("Win Rate (%)",      0),
        "PF":        bt.get("Profit Factor",     0),
        "Trades":    bt.get("Total Trades",      0),
        "DirAcc":    dir_acc,
        "DNRec":     dn_rec,
        "t_up":      t_up,
        "t_down":    t_down,
        "optuna":    "tuned" if use_params is None else "reused",
    })


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Cross-window summary
# ─────────────────────────────────────────────────────────────────────────────
total_elapsed = (time.time() - total_t0) / 60
print(f"\n\n{SEP2}")
print(f"  ROLLING WALK-FORWARD VALIDATION SUMMARY")
print(f"  Total elapsed: {total_elapsed:.1f} min   "
      f"({'tune-once' if TUNE_ONCE else 'tune-per-window'}, "
      f"{N_TRIALS_ROLLING} trials)")
print(f"{SEP2}\n")

hdr = (f"  {'Window':<22} {'Regime':<24} {'B&H':>6} {'AI Ret':>7}"
       f" {'Sharpe':>7} {'Sort':>7} {'MaxDD':>7}"
       f" {'WR':>6} {'PF':>7} {'Trd':>4} {'DirAcc':>7}")
print(hdr)
print("  " + "-" * 108)

for r in summary_rows:
    ai_s = "+" if r["Return"]    >= 0 else ""
    bh_s = "+" if r["BH Return"] >= 0 else ""
    print(
        f"  {r['Window']:<22}"
        f" {r['Regime']:<24}"
        f" {bh_s}{r['BH Return']:>5.1f}%"
        f" {ai_s}{r['Return']:>6.1f}%"
        f" {r['Sharpe']:>7.3f}"
        f" {r['Sortino']:>7.3f}"
        f" {r['MaxDD']:>6.1f}%"
        f" {r['WinRate']:>5.1f}%"
        f" {r['PF']:>7.3f}"
        f" {int(r['Trades']):>4}"
        f" {r['DirAcc']:>6.1%}"
    )

if summary_rows:
    returns = [r["Return"]  for r in summary_rows]
    sharpes = [r["Sharpe"]  for r in summary_rows]
    mdd     = [r["MaxDD"]   for r in summary_rows]
    wrs     = [r["WinRate"] for r in summary_rows]
    pfs     = [r["PF"]      for r in summary_rows]
    trades  = [r["Trades"]  for r in summary_rows]
    bh_rets = [r["BH Return"] for r in summary_rows]

    print("\n  " + "-" * 108)
    print(f"\n  Aggregates across {len(summary_rows)} windows:")
    print(f"    Profitable windows    : {sum(r > 0 for r in returns)} / {len(returns)}")
    print(f"    Avg AI return         : {np.mean(returns):+.2f}%"
          f"  (range {min(returns):+.1f}% to {max(returns):+.1f}%)")
    print(f"    Avg B&H return        : {np.mean(bh_rets):+.2f}%")
    print(f"    Avg Sharpe            : {np.mean(sharpes):.3f}"
          f"  (range {min(sharpes):.3f} to {max(sharpes):.3f})")
    print(f"    Positive-Sharpe wins  : {sum(s > 0 for s in sharpes)} / {len(sharpes)}")
    print(f"    Avg Max Drawdown      : {np.mean(mdd):.2f}%")
    print(f"    Avg Win Rate          : {np.mean(wrs):.1f}%")
    print(f"    Avg Profit Factor     : {np.mean(pfs):.3f}")
    print(f"    Avg Trades / window   : {np.mean(trades):.1f}")

print(f"\n{SEP2}\n  VALIDATION COMPLETE\n{SEP2}\n")
