"""
Step 4F retrain script — APEX Metals AI v6.0
Retrains Gold model with new macro + seasonality + regime features.
Uses fast-retrain mode (pre-tuned hyperparams, skips Optuna) for speed.
Run from project root: python run_retrain_v6.py
"""
import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import download_data, get_train_test_split
from src.features import add_features, classify_regime
from src.macro_loader import download_fred, add_macro_features
from src.train import train_all_models, save_models, load_models, compute_training_stats
from src.config import DATA_DIR, MODELS_DIR, FRED_API_KEY

print("\n" + "="*60)
print("  APEX Metals AI v6.0 — Gold Model Retrain")
print("="*60)

# ── Load existing hyperparams ──────────────────────────────────────────────────
print("\n[1/5] Loading existing hyperparams from models.pkl…")
_, old_clf, _, _, _ = load_models()
if old_clf is None or "_hyperparams" not in old_clf:
    print("  WARNING: No saved hyperparams found. Will run with N_TRIALS=5.")
    pretrained_hp = None
else:
    pretrained_hp = old_clf["_hyperparams"]
    print(f"  Loaded hyperparams. LGB clf estimators: "
          f"{pretrained_hp['lgb_clf'].get('n_estimators', '?')}")

# ── Download & feature engineering ────────────────────────────────────────────
print("\n[2/5] Downloading market data and building features…")
t0 = time.perf_counter()

try:
    raw = download_data(force_refresh=True)
except Exception as e:
    print(f"  Force refresh failed ({e}), using cached data.")
    raw = download_data(force_refresh=False)

df = add_features(raw)

# Macro features (FRED — optional)
try:
    if FRED_API_KEY:
        macro_df = download_fred(force_refresh=False)
        df = add_macro_features(df, macro_df)
    else:
        df = add_macro_features(df, pd.DataFrame())
except Exception as e:
    print(f"  FRED macro unavailable ({e}), continuing without.")

# Regime classification (new in v6)
df = classify_regime(df, window=20)
regime_counts = df["regime"].value_counts()

print(f"  Data: {len(df):,} bars | Features will be computed at train time")
print(f"  Regime distribution: {dict(regime_counts)}")
print(f"  Data download: {time.perf_counter()-t0:.1f}s")

# ── Train/test split ──────────────────────────────────────────────────────────
print("\n[3/5] Splitting train/test (80/20 time-ordered)…")
train_df, test_df = get_train_test_split(df)
print(f"  Train: {len(train_df):,} bars | Test: {len(test_df):,} bars")

# ── Training ───────────────────────────────────────────────────────────────────
print("\n[4/5] Training models…")
t_train = time.perf_counter()
logs = []

def _log(msg):
    logs.append(msg)
    print(f"  {msg}")

if pretrained_hp is not None:
    reg_r, clf_r, feat, sr, sc = train_all_models(
        train_df, test_df,
        n_trials=5,
        pretrained_hyperparams=pretrained_hp,
        fast_retrain=True,
        progress_callback=_log,
    )
else:
    from src.config import N_TRIALS
    reg_r, clf_r, feat, sr, sc = train_all_models(
        train_df, test_df,
        n_trials=5,
        progress_callback=_log,
    )

elapsed = time.perf_counter() - t_train
print(f"\n  Training complete in {elapsed:.0f}s")

# ── Evaluate ───────────────────────────────────────────────────────────────────
print("\n[5/5] Evaluating model accuracy…")
stk = clf_r.get("Stacking", {})
y_pred = stk.get("predictions")
y_true = stk.get("y_test")

if y_pred is not None and y_true is not None:
    y_pred = np.array(y_pred).ravel().astype(int)
    y_true = np.array(y_true).ravel().astype(int)

    overall_acc = accuracy_score(y_true, y_pred)

    label_names = {0: "DOWN", 1: "SIDEWAYS", 2: "UP"}
    class_counts = {label_names[c]: int(np.sum(y_true == c)) for c in [0, 1, 2]}

    per_class = {}
    for c in [0, 1, 2]:
        mask = y_true == c
        if mask.sum() > 0:
            per_class[label_names[c]] = float(accuracy_score(y_true[mask], y_pred[mask]))
        else:
            per_class[label_names[c]] = 0.0

    print(f"\n  {'='*50}")
    print(f"  OVERALL ACCURACY:        {overall_acc:.1%}")
    print(f"  {'='*50}")
    for cls in ["DOWN", "SIDEWAYS", "UP"]:
        n = class_counts.get(cls, 0)
        acc = per_class.get(cls, 0)
        print(f"  {cls:<10} accuracy:    {acc:.1%}  (n={n})")

    # Feature count
    print(f"\n  Features total:          {len(feat)}")

    # Regime model sample counts
    regime_models = clf_r.get("_regime_models", {})
    if regime_models:
        print(f"\n  Regime-conditional models:")
        for r, m in regime_models.items():
            status = "trained" if m is not None else "skipped (<60 samples)"
            print(f"    {r:<12}: {status}")

    # Training stats
    ts = clf_r.get("_training_stats", {})
    print(f"\n  Training stats computed: {len(ts)} features")

else:
    print("  ERROR: No predictions found in stacking results.")

# ── Save ───────────────────────────────────────────────────────────────────────
print("\n  Saving models.pkl…")
save_models(reg_r, clf_r, feat, sr, sc)

# Save retrain log
import json as _json
_log_path = os.path.join(DATA_DIR, "model_retrain_log.json")
os.makedirs(DATA_DIR, exist_ok=True)
from datetime import datetime, timezone
_bar_date = (df.index[-1].strftime("%Y-%m-%d")
             if hasattr(df.index[-1], "strftime") else str(df.index[-1])[:10])
with open(_log_path, "w") as _lf:
    _json.dump({"last_retrain_utc": datetime.now(timezone.utc).isoformat(),
                "last_bar_date": _bar_date}, _lf)

print(f"\n{'='*60}")
print("  RETRAIN COMPLETE")
print(f"  Overall accuracy: {overall_acc:.1%}")
print(f"  DOWN: {per_class.get('DOWN', 0):.1%}  SIDEWAYS: {per_class.get('SIDEWAYS', 0):.1%}  UP: {per_class.get('UP', 0):.1%}")
print(f"  Features: {len(feat)}")
print(f"{'='*60}\n")
