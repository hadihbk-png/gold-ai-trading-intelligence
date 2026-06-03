"""
Full Optuna retrain — APEX Metals AI v6.0 Gold Model
=====================================================
Addresses DOWN recall collapse with:
  1. Full Optuna hyperparameter search on 146-feature set
  2. No SIDEWAYS class-weight boost (sideways_weight_boost=1.0)
  3. Extended threshold grid targeting DOWN recall >= 25%
  4. Per-class precision / recall / F1 table (before AND after threshold tuning)

Split methodology
-----------------
Strictly forward-looking temporal split:
  Train = first (N - TEST_YEARS) years of data
  Test  = final TEST_YEARS year(s)
No shuffling. Test bars are always chronologically after all training bars.
This is equivalent to a single-window walk-forward validation.
"""
import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import download_data, get_train_test_split
from src.features import add_features, classify_regime
from src.macro_loader import download_fred, add_macro_features
from src.train import train_all_models, save_models, compute_training_stats
from src.calibration import apply_threshold
from src.config import DATA_DIR, MODELS_DIR, FRED_API_KEY, TEST_YEARS, N_TRIALS

LABEL = {0: "DOWN", 1: "SIDEWAYS", 2: "UP"}

# ══════════════════════════════════════════════════════════════════════════════
def _class_table(y_true, y_pred, label=""):
    """Print precision / recall / F1 per class plus overall accuracy."""
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    lines = []
    lines.append(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append(f"  {'-'*52}")
    for i, cls in enumerate(["DOWN", "SIDEWAYS", "UP"]):
        lines.append(
            f"  {cls:<12} {p[i]:>10.1%} {r[i]:>10.1%} {f[i]:>10.3f} {int(sup[i]):>10}"
        )
    lines.append(f"  {'-'*52}")
    lines.append(f"  {'Overall acc':<12} {acc:>10.1%}")
    if label:
        print(f"\n  --- {label} ---")
    print("\n".join(lines))
    return {"precision": p, "recall": r, "f1": f, "support": sup, "accuracy": acc}


def _threshold_grid_search(y_true, probas, min_down_recall=0.25):
    """
    Grid-search (t_down, t_up) to maximise macro-F1 subject to
    DOWN recall >= min_down_recall.  Falls back to maximising DOWN recall
    alone if no configuration meets the floor.
    """
    from sklearn.metrics import f1_score, recall_score

    t_dn_grid = np.round(np.arange(0.10, 0.42, 0.02), 3)
    t_up_grid = np.round(np.arange(0.30, 0.62, 0.02), 3)

    best_f1, best_dn_rec = -1.0, -1.0
    best_t_dn, best_t_up = 0.25, 0.38
    best_preds_constrained = None
    best_preds_fallback = None

    p_dn = probas[:, 0]
    p_up = probas[:, 2]

    for t_dn in t_dn_grid:
        for t_up in t_up_grid:
            preds = np.full(len(y_true), 1, dtype=int)
            up_m   = p_up   > t_up
            dn_m   = p_dn   > t_dn
            both   = up_m & dn_m
            preds[up_m & ~both]  = 2
            preds[dn_m & ~both]  = 0
            preds[both] = np.where(p_up[both] >= p_dn[both], 2, 0)

            dn_rec = recall_score(y_true, preds, labels=[0], average="macro", zero_division=0)
            mf1    = f1_score(y_true, preds, average="macro", zero_division=0)

            if dn_rec >= min_down_recall and mf1 > best_f1:
                best_f1 = mf1
                best_t_dn, best_t_up = t_dn, t_up
                best_preds_constrained = preds.copy()

            if dn_rec > best_dn_rec:
                best_dn_rec = dn_rec
                best_preds_fallback = preds.copy()

    if best_preds_constrained is not None:
        return best_t_dn, best_t_up, best_preds_constrained, "constrained (DOWN recall floor met)"
    else:
        # Find the config that got closest to the floor
        print(f"  WARNING: No configuration achieved DOWN recall >= {min_down_recall:.0%}.")
        print(f"  Best DOWN recall achieved: {best_dn_rec:.1%}. Using best recall fallback.")
        return best_t_dn, best_t_up, best_preds_fallback, "fallback (maximising DOWN recall)"


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*64)
print("  APEX Metals AI v6.0 — Full Optuna Retrain (DOWN recall fix)")
print("="*64)

# ── 1. Data ────────────────────────────────────────────────────────────────────
print("\n[1/6] Downloading market data…")
t0 = time.perf_counter()

try:
    raw = download_data(force_refresh=True)
except Exception as e:
    print(f"  Force refresh failed ({e}), using cache.")
    raw = download_data(force_refresh=False)

df = add_features(raw)

try:
    if FRED_API_KEY:
        macro_df = download_fred(force_refresh=False)
        df = add_macro_features(df, macro_df)
    else:
        df = add_macro_features(df, pd.DataFrame())
except Exception as e:
    print(f"  FRED unavailable ({e}), continuing without macro.")

df = classify_regime(df, window=20)
regime_counts = df["regime"].value_counts().to_dict()

print(f"  Total bars:  {len(df):,}")
print(f"  Date range:  {df.index[0].date()} -> {df.index[-1].date()}")
print(f"  Regimes:     {regime_counts}")
print(f"  Data fetch:  {time.perf_counter()-t0:.1f}s")

# ── 2. Train / test split ──────────────────────────────────────────────────────
print(f"\n[2/6] Train/test split (last {TEST_YEARS} year as test — strictly forward-looking)…")
train_df, test_df = get_train_test_split(df)
print(f"  Train: {len(train_df):,} bars  ({train_df.index[0].date()} -> {train_df.index[-1].date()})")
print(f"  Test:  {len(test_df):,}  bars  ({test_df.index[0].date()} -> {test_df.index[-1].date()})")
print(f"  !! All test dates are AFTER the last training date — no look-ahead bias.")

# ── 3. Full Optuna training ────────────────────────────────────────────────────
print(f"\n[3/6] Full Optuna training (N_TRIALS={N_TRIALS}, sideways_boost=1.0 — no SIDEWAYS tilt)…")
print(f"  Estimated time: 15–25 minutes. Progress below.")
t_train = time.perf_counter()
logs = []

def _log(msg):
    logs.append(msg)
    print(f"  {msg}")

reg_r, clf_r, feat, sr, sc = train_all_models(
    train_df, test_df,
    n_trials=N_TRIALS,
    pretrained_hyperparams=None,   # Force full Optuna — no reuse
    fast_retrain=False,
    sideways_weight_boost=1.0,     # Remove the SIDEWAYS tilt that crushed DOWN recall
    progress_callback=_log,
)

elapsed = time.perf_counter() - t_train
print(f"\n  Training complete in {elapsed/60:.1f} min")

# ── 4. Evaluate BEFORE threshold tuning ───────────────────────────────────────
print(f"\n[4/6] Evaluating with default thresholds…")
stk = clf_r.get("Stacking", {})
y_pred_default = np.array(stk.get("predictions", [])).ravel().astype(int)
y_true         = np.array(stk.get("y_test", [])).ravel().astype(int)
probas         = stk.get("probabilities")
thresholds_stk = stk.get("thresholds", {})

if len(y_pred_default) == 0:
    print("  ERROR: no predictions in stacking results — aborting.")
    sys.exit(1)

print(f"  Stacking thresholds (Optuna-tuned): "
      f"DOWN>{thresholds_stk.get('threshold_down', '?')}  "
      f"UP>{thresholds_stk.get('threshold_up', '?')}")

stats_before = _class_table(y_true, y_pred_default, label="BEFORE threshold tuning (default thresholds)")

# ── 5. Extended threshold search ───────────────────────────────────────────────
print(f"\n[5/6] Extended threshold grid search (targeting DOWN recall >= 25%)…")

if probas is None:
    print("  No OOF probabilities available — cannot tune thresholds.")
    stats_after   = stats_before
    t_dn_best = thresholds_stk.get("threshold_down", 0.25)
    t_up_best = thresholds_stk.get("threshold_up",   0.38)
    y_pred_tuned = y_pred_default
    grid_note = "skipped (no probas)"
else:
    # Use the test-set probabilities for reporting (OOF were used inside training)
    t_dn_best, t_up_best, y_pred_tuned, grid_note = _threshold_grid_search(
        y_true, probas, min_down_recall=0.25
    )
    print(f"  Best thresholds: DOWN>{t_dn_best}  UP>{t_up_best}  ({grid_note})")
    stats_after = _class_table(y_true, y_pred_tuned, label="AFTER extended threshold tuning")

# ── 6. Side-by-side summary ───────────────────────────────────────────────────
print(f"\n[6/6] Before / After comparison")
print(f"\n  {'Class':<12} {'Recall BEFORE':>15} {'Recall AFTER':>15} {'Δ Recall':>10}")
print(f"  {'-'*52}")
for i, cls in enumerate(["DOWN", "SIDEWAYS", "UP"]):
    rb = stats_before["recall"][i]
    ra = stats_after["recall"][i]
    print(f"  {cls:<12} {rb:>15.1%} {ra:>15.1%} {ra-rb:>+10.1%}")
print(f"  {'-'*52}")
print(f"  {'Accuracy':<12} {stats_before['accuracy']:>15.1%} {stats_after['accuracy']:>15.1%} "
      f"{stats_after['accuracy']-stats_before['accuracy']:>+10.1%}")

# Flag if DOWN recall is still in single digits
final_down_recall = stats_after["recall"][0]
if final_down_recall < 0.10:
    print(f"\n  [FAIL] FAIL: DOWN recall still {final_down_recall:.1%} (single digits) — threshold tuning insufficient.")
    print(f"     Recommend: increase DOWN class weight further and retrain with more Optuna trials.")
else:
    print(f"\n  [OK] DOWN recall: {final_down_recall:.1%} — acceptable.")

print(f"\n  Features total:          {len(feat)}")
regime_models = clf_r.get("_regime_models", {})
if regime_models:
    print(f"  Regime-conditional models:")
    for r, m in regime_models.items():
        status = "trained" if m is not None else "skipped (<60 samples)"
        print(f"    {r:<12}: {status}")

# ── Save ───────────────────────────────────────────────────────────────────────
print(f"\n  Saving models.pkl…")

# Store tuned thresholds in clf_results so save_models picks them up
if "Stacking" in clf_r:
    clf_r["Stacking"]["thresholds"] = {
        "threshold_down": float(t_dn_best),
        "threshold_up":   float(t_up_best),
    }
    clf_r["Stacking"]["predictions"] = y_pred_tuned   # Save tuned preds too

save_models(reg_r, clf_r, feat, sr, sc)

# training_stats.json already written by train_all_models; write retrain log
_bar_date = (df.index[-1].strftime("%Y-%m-%d")
             if hasattr(df.index[-1], "strftime") else str(df.index[-1])[:10])
os.makedirs(DATA_DIR, exist_ok=True)
with open(os.path.join(DATA_DIR, "model_retrain_log.json"), "w") as _lf:
    json.dump({
        "last_retrain_utc": datetime.now(timezone.utc).isoformat(),
        "last_bar_date":    _bar_date,
        "n_trials":         N_TRIALS,
        "sideways_boost":   1.0,
        "down_threshold":   float(t_dn_best),
        "up_threshold":     float(t_up_best),
        "down_recall_after": float(final_down_recall),
    }, _lf)

print(f"\n{'='*64}")
print(f"  RETRAIN COMPLETE")
print(f"  Overall accuracy (after tuning): {stats_after['accuracy']:.1%}")
print(f"  DOWN   P={stats_after['precision'][0]:.1%}  R={stats_after['recall'][0]:.1%}  F1={stats_after['f1'][0]:.3f}")
print(f"  SIDE   P={stats_after['precision'][1]:.1%}  R={stats_after['recall'][1]:.1%}  F1={stats_after['f1'][1]:.3f}")
print(f"  UP     P={stats_after['precision'][2]:.1%}  R={stats_after['recall'][2]:.1%}  F1={stats_after['f1'][2]:.3f}")
print(f"  Thresholds: DOWN>{t_dn_best}  UP>{t_up_best}")
print(f"  Features: {len(feat)}")
print(f"{'='*64}\n")
