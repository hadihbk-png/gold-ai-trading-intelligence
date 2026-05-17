"""
Probability calibration analysis: Brier score, reliability curves,
and confidence-threshold optimization for directional trading.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

_CLASS_LABELS = {0: "DOWN", 1: "SIDEWAYS", 2: "UP"}


def compute_brier_scores(y_true: np.ndarray, probas: np.ndarray) -> dict:
    """
    Per-class Brier score (lower = better calibrated).
    Includes mean across all classes.
    """
    scores = {}
    n_classes = probas.shape[1] if probas.ndim > 1 else 1
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        p = probas[:, c] if probas.ndim > 1 else probas
        scores[_CLASS_LABELS.get(c, str(c))] = round(float(brier_score_loss(y_bin, p)), 4)
    scores["Mean"] = round(float(np.mean(list(scores.values()))), 4)
    return scores


def compute_reliability_curves(
    y_true: np.ndarray,
    probas: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Per-class reliability (calibration) curves.
    Returns dict: class_label → {fraction_pos, mean_pred} lists.
    """
    curves = {}
    n_classes = probas.shape[1] if probas.ndim > 1 else 1
    for c in range(n_classes):
        label = _CLASS_LABELS.get(c, str(c))
        y_bin = (y_true == c).astype(int)
        p = probas[:, c] if probas.ndim > 1 else probas
        try:
            frac, mean_pred = calibration_curve(
                y_bin, p, n_bins=n_bins, strategy="uniform",
            )
            curves[label] = {
                "fraction_pos": frac.tolist(),
                "mean_pred":    mean_pred.tolist(),
            }
        except Exception:
            curves[label] = {"fraction_pos": [], "mean_pred": []}
    return curves


def optimize_threshold(
    y_true: np.ndarray,
    probas: np.ndarray,
    metric: str = "profit_proxy",
    thresholds_up: np.ndarray | None = None,
    thresholds_down: np.ndarray | None = None,
    min_trade_freq: float = 0.05,
) -> dict:
    """
    Find (threshold_up, threshold_down) that maximises the chosen metric
    on the provided data set (use OOF probas to avoid over-fitting).

    Separate grids for UP and DOWN thresholds — DOWN grid uses 0.35–0.45
    to filter weak DOWN calls while retaining high-confidence shorts.
    UP grid uses a wider range (0.33–0.65) where UP signals emerge at
    higher confidence.

    metric:
      'profit_proxy' — (default) balances directional accuracy, DOWN recall,
                       and trade frequency; designed to select thresholds that
                       produce profitable behaviour not just accuracy
      'accuracy'     — overall classification accuracy
      'directional'  — directional accuracy weighted by trade frequency
      'f1'           — macro-F1

    min_trade_freq: minimum fraction of bars that must be actionable.
                    Hard floor prevents collapse to tiny handful of calls.

    Returns dict with threshold_up, threshold_down, metric, best_score.
    """
    from sklearn.metrics import recall_score as _recall

    if thresholds_up is None:
        # UP signals emerge at higher confidence — wider grid
        thresholds_up = np.linspace(0.33, 0.65, 33)

    if thresholds_down is None:
        # DOWN probabilities are compressed by the meta-learner but calibration
        # shows P(DOWN)>0.30 → 67%+ actual DOWN rate, meaning the model is
        # under-confident.  Raising the floor to 0.35 filters out the weakest
        # spurious DOWN calls while keeping genuinely high-confidence shorts.
        thresholds_down = np.linspace(0.35, 0.45, 11)

    best_score       = -np.inf
    best_thresh_up   = 0.40
    best_thresh_down = 0.33
    min_trades = max(10, int(min_trade_freq * len(y_true)))

    p_down = probas[:, 0]
    p_up   = probas[:, 2]

    for t_up in thresholds_up:
        for t_down in thresholds_down:
            preds = np.full(len(y_true), 1, dtype=int)
            up_mask   = p_up   > t_up
            down_mask = p_down > t_down
            both      = up_mask & down_mask
            preds[up_mask   & ~both] = 2
            preds[down_mask & ~both] = 0
            preds[both] = np.where(p_up[both] >= p_down[both], 2, 0)

            act = preds != 1
            if act.sum() < min_trades:
                continue

            if metric == "profit_proxy":
                # Primary: directional accuracy on actionable bars
                dir_acc = float(np.mean(preds[act] == y_true[act]))
                # Secondary: DOWN recall (counteracts the systematic DOWN suppression
                # identified via calibration analysis — P(DOWN)>0.30 maps to 67%+
                # actual DOWN rate, so the optimizer must reward capturing DOWN)
                down_rec = float(_recall(
                    y_true, preds, labels=[0], average="macro", zero_division=0,
                ))
                # Tertiary: trade frequency (prevents collapse to 1–2 calls)
                freq = float(act.sum()) / len(y_true)
                score = 0.50 * dir_acc + 0.30 * down_rec + 0.20 * freq

            elif metric == "accuracy":
                score = float(np.mean(preds == y_true))

            elif metric == "directional":
                dir_acc = float(np.mean(preds[act] == y_true[act]))
                freq    = float(act.sum()) / len(y_true)
                score   = 0.70 * dir_acc + 0.30 * freq

            else:  # f1
                try:
                    from sklearn.metrics import f1_score
                    score = float(f1_score(
                        y_true, preds, average="macro", zero_division=0,
                    ))
                except Exception:
                    continue

            if score > best_score:
                best_score       = score
                best_thresh_up   = float(t_up)
                best_thresh_down = float(t_down)

    return {
        "threshold_up":   best_thresh_up,
        "threshold_down": best_thresh_down,
        "metric":         metric,
        "best_score":     round(best_score, 4),
    }


def apply_threshold(
    probas: np.ndarray,
    threshold_up:   float = 0.50,
    threshold_down: float = 0.50,
) -> np.ndarray:
    """Apply custom per-class confidence thresholds to a probability matrix."""
    preds  = np.full(len(probas), 1, dtype=int)
    p_up   = probas[:, 2]
    p_down = probas[:, 0]
    up_m   = p_up > threshold_up
    dn_m   = p_down > threshold_down
    both   = up_m & dn_m
    preds[up_m & ~both]   = 2
    preds[dn_m & ~both]   = 0
    preds[both] = np.where(p_up[both] >= p_down[both], 2, 0)
    return preds
