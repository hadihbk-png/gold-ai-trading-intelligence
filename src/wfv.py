"""
Walk-Forward Validation — LightGBM only, fixed hyperparams for speed.

Train: 252 bars (~1 year), Test: 21 bars (~1 month), Step: 21 bars.
Requires at least 2 folds to return meaningful statistics.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

_LGB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=15,
    min_child_samples=10,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)


def run_wfv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Target_Direction",
    train_window: int = 252,
    test_window: int = 21,
    step: int = 21,
) -> dict:
    """
    Rolling walk-forward validation using LightGBM.

    Returns a dict with:
      folds          — list of per-fold dicts (fold, start, end, accuracy, n_predictions)
      mean_accuracy  — mean accuracy across folds (%)
      std_accuracy   — standard deviation across folds (%)
      min_accuracy   — worst fold accuracy (%)
      max_accuracy   — best fold accuracy (%)
      n_folds        — number of folds completed
      beat_50pct     — number of folds where accuracy > 50%
    On failure returns {"error": "<reason>", "folds": [...]}.
    """
    valid_cols = [c for c in feature_cols if c in df.columns]
    data = df[valid_cols + [target_col]].dropna()
    n = len(data)

    folds: list[dict] = []
    start = 0

    while start + train_window + test_window <= n:
        train_sl = data.iloc[start : start + train_window]
        test_sl  = data.iloc[start + train_window : start + train_window + test_window]

        X_tr, y_tr = train_sl[valid_cols].values, train_sl[target_col].values
        X_te, y_te = test_sl[valid_cols].values,  test_sl[target_col].values

        try:
            mdl = LGBMClassifier(**_LGB_PARAMS)
            mdl.fit(X_tr, y_tr)
            preds = mdl.predict(X_te)
            acc = float((preds == y_te).mean() * 100)
        except Exception:
            acc = float("nan")

        folds.append({
            "fold":          len(folds) + 1,
            "start":         test_sl.index[0],
            "end":           test_sl.index[-1],
            "accuracy":      acc,
            "n_predictions": len(y_te),
        })
        start += step

    if len(folds) < 2:
        return {"error": f"Only {len(folds)} fold(s) — need at least 2 (try a longer history).", "folds": folds}

    accs = [f["accuracy"] for f in folds if not np.isnan(f["accuracy"])]
    if not accs:
        return {"error": "All folds failed to produce predictions.", "folds": folds}

    return {
        "folds":         folds,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy":  float(np.std(accs)),
        "min_accuracy":  float(np.min(accs)),
        "max_accuracy":  float(np.max(accs)),
        "n_folds":       len(folds),
        "beat_50pct":    int(sum(a > 50 for a in accs)),
    }
