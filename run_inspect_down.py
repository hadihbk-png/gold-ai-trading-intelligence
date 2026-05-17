"""
Inspect stacking meta-learner DOWN suppression on OOF data.
Uses the already-saved models.pkl — no retraining.
"""
import sys, os, warnings, pickle
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np

SEP  = "=" * 68
SEP2 = "-" * 68

# ── Load saved model ──────────────────────────────────────────────────────────
with open("models/models.pkl", "rb") as f:
    d = pickle.load(f)

stack_clf   = d["stack_clf"]
clf_results = d["clf_results"]

oof_y     = getattr(stack_clf, "_oof_y",     None)
oof_proba = getattr(stack_clf, "_oof_proba", None)

if oof_y is None or oof_proba is None:
    print("No OOF data found in stack_clf — cannot inspect.")
    sys.exit(1)

meta_probas = stack_clf._meta_clf.predict_proba(oof_proba)   # (n_oof, 3)
meta_preds  = stack_clf._meta_clf.predict(oof_proba)

n_oof = len(oof_y)
print(f"\n{SEP}\n  OOF DATA OVERVIEW\n{SEP}")
print(f"  OOF rows     : {n_oof}")
for cls, label in [(0,"DOWN"),(1,"SIDEWAYS"),(2,"UP")]:
    n = (oof_y == cls).sum()
    print(f"  {label:<10}: {n:4d} bars ({n/n_oof*100:.1f}%)")

# ── DOWN suppression analysis ──────────────────────────────────────────────────
print(f"\n{SEP}\n  DOWN CLASS — PROBABILITY DISTRIBUTION ON TRUE DOWN BARS\n{SEP}")

down_mask  = (oof_y == 0)
n_down     = down_mask.sum()
p_down_on_down = meta_probas[down_mask, 0]     # P(DOWN) when truth = DOWN
p_side_on_down = meta_probas[down_mask, 1]
p_up_on_down   = meta_probas[down_mask, 2]

print(f"  True DOWN bars in OOF: {n_down}")
print()
print(f"  P(DOWN)    on true-DOWN bars:")
print(f"    Mean     : {p_down_on_down.mean():.4f}")
print(f"    Median   : {np.median(p_down_on_down):.4f}")
print(f"    Max      : {p_down_on_down.max():.4f}")
print(f"    Min      : {p_down_on_down.min():.4f}")
print(f"    Std      : {p_down_on_down.std():.4f}")
print(f"    > 0.40   : {(p_down_on_down > 0.40).sum()} bars ({(p_down_on_down > 0.40).mean()*100:.1f}%)")
print(f"    > 0.33   : {(p_down_on_down > 0.33).sum()} bars ({(p_down_on_down > 0.33).mean()*100:.1f}%)")
print(f"    > 0.25   : {(p_down_on_down > 0.25).sum()} bars ({(p_down_on_down > 0.25).mean()*100:.1f}%)")
print()
print(f"  P(SIDEWAYS) on true-DOWN bars (competing class):")
print(f"    Mean     : {p_side_on_down.mean():.4f}")
print(f"    Median   : {np.median(p_side_on_down):.4f}")
print()
print(f"  P(UP)      on true-DOWN bars (competing class):")
print(f"    Mean     : {p_up_on_down.mean():.4f}")
print(f"    Median   : {np.median(p_up_on_down):.4f}")

# What does the meta predict on true-DOWN bars?
pred_on_down = meta_preds[down_mask]
print(f"\n  Meta-clf prediction on true-DOWN bars:")
for cls, label in [(0,"DOWN"),(1,"SIDEWAYS"),(2,"UP")]:
    n = (pred_on_down == cls).sum()
    print(f"    Predicted {label:<10}: {n:3d} ({n/n_down*100:.1f}%)")

# ── Calibration buckets: P(DOWN) vs actual DOWN fraction ──────────────────────
print(f"\n{SEP2}\n  CALIBRATION: P(DOWN) predicted vs actual DOWN fraction\n{SEP2}")
bins = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 1.01]
p_down_all   = meta_probas[:, 0]
y_down_bin   = (oof_y == 0).astype(int)
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_down_all >= lo) & (p_down_all < hi)
    n    = mask.sum()
    if n == 0:
        continue
    actual = y_down_bin[mask].mean()
    pred   = p_down_all[mask].mean()
    bar    = "#" * int(actual * 30)
    ideal  = "#" * int(pred * 30)
    bias   = "over" if actual < pred else "under"
    print(f"  [{lo:.2f}–{hi:.2f})  n={n:4d}  pred={pred:.3f}  actual={actual:.3f}  "
          f"{'|'+bar:<33}  {bias}-confident")

# ── L1 DOWN probabilities feeding the meta ────────────────────────────────────
print(f"\n{SEP}\n  L1 RAW DOWN PROBABILITIES INTO META (true-DOWN OOF bars)\n{SEP}")
# oof_proba layout: [XGB_D, XGB_S, XGB_U, LGB_D, LGB_S, LGB_U, CB_D, CB_S, CB_U]
model_names = ["XGBoost", "LightGBM", "CatBoost"]
for i, name in enumerate(model_names):
    p_d = oof_proba[down_mask, i * 3]      # col 0 of each model = DOWN proba
    print(f"  {name:<12} P(DOWN) on true-DOWN:  mean={p_d.mean():.4f}  "
          f"median={np.median(p_d):.4f}  max={p_d.max():.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  SUMMARY\n{SEP2}")
mean_p_down = p_down_on_down.mean()
mean_p_side = p_side_on_down.mean()
frac_above_thr = (p_down_on_down > 0.40).mean()

if mean_p_down < 0.30:
    print(f"  DIAGNOSIS: Meta-learner systematically suppresses DOWN.")
    print(f"  Mean P(DOWN) on true-DOWN bars = {mean_p_down:.3f} (should be >0.40).")
    print(f"  Competing class dominates: P(SIDEWAYS) mean = {mean_p_side:.3f}.")
    print(f"  Only {frac_above_thr*100:.1f}% of true-DOWN OOF bars exceed DOWN threshold 0.40.")
    print(f"  Root causes to check:")
    print(f"    1. SMOTE on OOF training folds shifted DOWN distribution in L1 features.")
    print(f"    2. Meta XGB learned SIDEWAYS bias from SMOTE-augmented OOF probabilities.")
    print(f"    3. Isotonic calibration compressed DOWN probabilities upward toward SIDEWAYS.")
elif mean_p_down < 0.40:
    print(f"  DIAGNOSIS: Partial DOWN suppression. Mean={mean_p_down:.3f}.")
    print(f"  Lowering DOWN threshold to 0.30 may recover some DOWN signals.")
else:
    print(f"  DIAGNOSIS: DOWN probabilities look reasonable (mean={mean_p_down:.3f}).")
    print(f"  Suppression may be at threshold level, not meta-learner.")

print(f"\n{SEP}\n  DONE\n{SEP}\n")
