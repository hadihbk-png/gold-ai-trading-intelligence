"""
Feature drift monitoring.

Two complementary metrics:
  PSI (Population Stability Index)
    < 0.10  → Stable
    0.10-0.20 → Moderate drift
    > 0.20  → Significant drift (consider retraining)

  KS test (Kolmogorov–Smirnov)
    Two-sample test of whether train-time and recent distributions differ.

Feature importance rank correlation:
  Spearman ρ between baseline feature importance ranks and those computed
  on the recent window.  ρ < 0.70 suggests the model may be learning
  different patterns than it did at training time.
"""

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore
from src.config import DRIFT_WINDOW_DAYS, DRIFT_PSI_THRESHOLD, DRIFT_MOD_THRESHOLD


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between two 1-D distributions."""
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual,   bins=breakpoints)

    exp_pct = np.where(exp_counts == 0, 1e-4, exp_counts / len(expected))
    act_pct = np.where(act_counts == 0, 1e-4, act_counts / len(actual))

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _drift_label(psi: float) -> str:
    if psi > DRIFT_PSI_THRESHOLD:
        return "High"
    if psi > DRIFT_MOD_THRESHOLD:
        return "Moderate"
    return "Stable"


def monitor_feature_drift(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_end_date,
    window_days: int = DRIFT_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Compare feature distributions between training period and recent window.

    Returns a DataFrame with PSI, KS statistic, p-value, and drift label
    for every feature, sorted by PSI (highest drift first).
    """
    train_mask  = df.index <= pd.Timestamp(train_end_date)
    train_data  = df.loc[train_mask, feature_cols].dropna()
    recent_data = df[feature_cols].tail(window_days).dropna()

    rows = []
    for col in feature_cols:
        if col not in train_data.columns:
            continue
        try:
            exp = train_data[col].values
            act = recent_data[col].values
            if len(exp) < 5 or len(act) < 5:
                continue

            psi_val = _psi(exp, act)
            ks_stat, ks_pval = stats.ks_2samp(exp, act)

            rows.append({
                "Feature":    col,
                "PSI":        round(psi_val, 4),
                "KS Stat":    round(float(ks_stat), 4),
                "KS p-value": round(float(ks_pval), 4),
                "Drift":      _drift_label(psi_val),
                "Train Mean": round(float(np.nanmean(exp)), 4),
                "Recent Mean": round(float(np.nanmean(act)), 4),
            })
        except Exception:
            pass

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("PSI", ascending=False).reset_index(drop=True)


def compare_importance_ranks(
    baseline_imp: pd.Series,
    current_imp:  pd.Series,
) -> dict | None:
    """
    Spearman rank correlation between baseline feature importance and a
    current-window estimate.  Returns None if insufficient overlap.
    """
    if baseline_imp is None or current_imp is None:
        return None

    common = list(set(baseline_imp.index) & set(current_imp.index))
    if len(common) < 10:
        return None

    baseline_ranks = baseline_imp[common].rank(ascending=False)
    current_ranks  = current_imp[common].rank(ascending=False)

    rho, pval = stats.spearmanr(baseline_ranks, current_ranks)
    return {
        "spearman_rho": round(float(rho), 4),
        "p_value":      round(float(pval), 4),
        "n_features":   len(common),
        "status":       "Stable" if rho > 0.70 else "Drifted",
    }


def drift_summary(drift_df: pd.DataFrame) -> dict:
    """Aggregate drift stats over all features."""
    if drift_df.empty:
        return {}
    counts = drift_df["Drift"].value_counts().to_dict()
    return {
        "pct_high":     counts.get("High", 0) / len(drift_df) * 100,
        "pct_moderate": counts.get("Moderate", 0) / len(drift_df) * 100,
        "pct_stable":   counts.get("Stable", 0) / len(drift_df) * 100,
        "top_drifted":  drift_df.head(5)["Feature"].tolist(),
        "mean_psi":     round(drift_df["PSI"].mean(), 4),
    }
