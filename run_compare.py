"""
Structural comparison: Individual best · Soft-vote · Stacking
Loads saved models.pkl — no retraining.
"""
import sys, os, warnings, pickle
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score

from src.data_loader import download_data, get_train_test_split
from src.features import add_features, get_feature_columns
from src.macro_loader import download_fred, add_macro_features
from src.calibration import apply_threshold, optimize_threshold
from src.backtest import run_backtest
from src.config import INITIAL_CAPITAL, DIRECTION_THRESHOLD

SEP  = "=" * 68
SEP2 = "-" * 68

# ── 1. Reload data (no force-refresh) ─────────────────────────────────────────
print(f"\n{SEP}\n  LOADING DATA (cached)\n{SEP}")
df_raw  = download_data(force_refresh=False)
macro   = download_fred() if os.getenv("FRED_API_KEY") else pd.DataFrame()
df_feat = add_features(df_raw)
df      = add_macro_features(df_feat, macro)
_, test_df = get_train_test_split(df)

# ── 2. Load saved models ───────────────────────────────────────────────────────
print(f"\n{SEP}\n  LOADING SAVED MODELS\n{SEP}")
with open("models/models.pkl", "rb") as f:
    d = pickle.load(f)

clf_results  = d["clf_results"]
feature_cols = d["feature_cols"]
stack_clf    = d["stack_clf"]

clf_target = "Target_Direction"
te   = test_df[feature_cols + [clf_target]].dropna()
Xte  = te[feature_cols].values
yte  = te[clf_target].values.astype(int)
test_dates = te.index

print(f"  Test bars    : {len(yte)}")
print(f"  Features     : {len(feature_cols)}")
print(f"\n  Class distribution in test set (±{DIRECTION_THRESHOLD*100:.1f}%):")
for cls, label in [(0,"DOWN"),(1,"SIDEWAYS"),(2,"UP")]:
    n = (yte == cls).sum()
    print(f"    {label:<10}: {n:3d} bars  ({n/len(yte)*100:.1f}%)")

# Shared thresholds from stacking OOF optimisation
t_up   = clf_results["Stacking"]["thresholds"]["threshold_up"]
t_down = clf_results["Stacking"]["thresholds"]["threshold_down"]
print(f"\n  Shared OOF thresholds: UP>{t_up:.3f}  DOWN>{t_down:.3f}")

# ── 3. Build prediction sets ───────────────────────────────────────────────────

def directional_acc(preds, y):
    act = preds != 1
    if act.sum() == 0:
        return 0.0
    return float(np.mean(preds[act] == y[act]))

def metrics_for(preds, probas, y, test_dates, df, label):
    """Compute all requested metrics for a prediction array."""
    act_mask = preds != 1
    n_trades_pred = int(act_mask.sum())
    dir_acc  = directional_acc(preds, y)
    down_rec = float(recall_score(y, preds, labels=[0], average="macro", zero_division=0))

    # Backtest
    _, _, trades_df, bt = run_backtest(
        df, preds, test_dates,
        clf_probas=probas,
        regime_series=None,
    )
    total_ret  = bt.get("Total Return (%)", 0)
    sharpe     = bt.get("Sharpe Ratio", 0)
    pf         = bt.get("Profit Factor", 0)
    n_trades_bt = int(bt.get("Total Trades", 0))

    return {
        "label":       label,
        "dir_acc":     dir_acc,
        "down_rec":    down_rec,
        "trade_count": n_trades_bt,
        "profit_factor": pf,
        "sharpe":      sharpe,
        "total_ret":   total_ret,
    }

results = []

# ── Individual models (use saved predictions & probas from clf_results) ────────
best_ind = None
best_ind_acc = -1
for name in ["XGBoost", "LightGBM", "CatBoost"]:
    res    = clf_results[name]
    probas = res.get("probabilities")
    if probas is None:
        continue
    preds = apply_threshold(probas, t_up, t_down)
    m = metrics_for(preds, probas, yte, test_dates, df, f"Individual – {name}")
    results.append(m)
    if m["dir_acc"] > best_ind_acc:
        best_ind_acc = m["dir_acc"]
        best_ind     = m

# ── Soft-vote (average of calibrated L1 probas) ────────────────────────────────
try:
    sv_proba = stack_clf.soft_vote_proba(Xte)
    sv_preds = apply_threshold(sv_proba, t_up, t_down)
    m = metrics_for(sv_preds, sv_proba, yte, test_dates, df, "Soft-vote ensemble")
    results.append(m)
except Exception as e:
    print(f"  Soft-vote failed: {e}")
    sv_proba = None
    sv_preds = None

# ── Stacking (current) ─────────────────────────────────────────────────────────
stk_preds  = clf_results["Stacking"]["predictions"]
stk_probas = clf_results["Stacking"]["probabilities"]
# Apply optimised thresholds (stored predictions used default 0.5)
stk_preds_thr = apply_threshold(stk_probas, t_up, t_down)
m = metrics_for(stk_preds_thr, stk_probas, yte, test_dates, df, "Stacking (current)")
results.append(m)

# ── 4. Print comparison table ──────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STRUCTURAL COMPARISON")
print(SEP)
print(f"  {'Strategy':<26} {'DirAcc':>8} {'DwnRec':>8} {'Trades':>7} "
      f"{'PF':>7} {'Sharpe':>8} {'TotRet':>9}")
print(f"  {'-'*73}")

for r in results:
    marker = " <-- BEST" if r["label"] == (best_ind["label"] if best_ind else "") else ""
    print(
        f"  {r['label']:<26} "
        f"{r['dir_acc']:>7.1%} "
        f"{r['down_rec']:>7.1%} "
        f"{r['trade_count']:>7d} "
        f"{r['profit_factor']:>7.3f} "
        f"{r['sharpe']:>8.3f} "
        f"{r['total_ret']:>+8.2f}%"
        f"{marker}"
    )

# ── 5. Verdict ────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  VERDICT\n{SEP2}")
soft_r  = next((r for r in results if "Soft" in r["label"]), None)
stack_r = next((r for r in results if "Stacking" in r["label"]), None)

if soft_r and stack_r:
    soft_better  = soft_r["dir_acc"] > stack_r["dir_acc"]
    soft_more    = soft_r["trade_count"] >= stack_r["trade_count"]
    soft_sharpe  = soft_r["sharpe"] > stack_r["sharpe"]
    score = sum([soft_better, soft_more, soft_sharpe])
    if score >= 2:
        print(f"  Soft-vote BEATS stacking on {score}/3 metrics "
              f"(dir-acc, trades, Sharpe).")
        print(f"  RECOMMENDATION: replace meta-learner with soft-vote.")
    else:
        print(f"  Stacking holds edge on {3-score}/3 metrics.")
        print(f"  RECOMMENDATION: investigate meta-learner overfitting before switching.")

print(f"\n{SEP}\n  DONE\n{SEP}\n")
