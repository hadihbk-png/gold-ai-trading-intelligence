"""
Model training orchestration.

Flow:
  1. Optuna hyperparameter search for each L1 model (XGB, LGB, CB).
  2. Build OOF stacking ensemble (src/ensemble.py) — no data leakage.
  3. Evaluate both individual models and the stacking ensemble on the test set.
  4. Save full results including baseline feature importance.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from src.config import MODELS_DIR, N_TRIALS, RANDOM_STATE
from src.features import get_feature_columns
from src.ensemble import (
    StackingRegressor, StackingClassifier,
    make_xgb_reg_factory, make_lgb_reg_factory, make_cb_reg_factory,
    make_xgb_clf_factory, make_lgb_clf_factory, make_cb_clf_factory,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ── Metrics ───────────────────────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))


def _get_importance(model, name: str, cols: list) -> pd.Series:
    try:
        imp = (model.get_feature_importance()
               if name == "CatBoost" else model.feature_importances_)
        return pd.Series(imp, index=cols).sort_values(ascending=False)
    except Exception:
        return pd.Series(dtype=float)


def _get_proba(model, X: np.ndarray) -> np.ndarray | None:
    try:
        return model.predict_proba(X) if hasattr(model, "predict_proba") else None
    except Exception:
        return None


def _val_split(X: np.ndarray, y: np.ndarray, ratio: float = 0.8):
    n = int(len(X) * ratio)
    return X[:n], y[:n], X[n:], y[n:]


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def _tune_xgb_reg(X, y, n_trials):
    def obj(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 400),
                 max_depth=trial.suggest_int("max_depth", 3, 8),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                 random_state=RANDOM_STATE, verbosity=0)
        Xtr, ytr, Xv, yv = _val_split(X, y)
        m = xgb.XGBRegressor(**p)
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        return float(np.sqrt(mean_squared_error(yv, m.predict(Xv))))
    s = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    s.optimize(obj, n_trials=n_trials, timeout=180, show_progress_bar=False)
    return s.best_params


def _tune_xgb_clf(X, y, n_trials, sample_weight=None):
    def obj(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 50, 150),
                 max_depth=trial.suggest_int("max_depth", 3, 8),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                 random_state=RANDOM_STATE, verbosity=0)
        Xtr, ytr, Xv, yv = _val_split(X, y)
        n_tr = len(Xtr)
        sw_tr = sample_weight[:n_tr] if sample_weight is not None else None
        m = xgb.XGBClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=sw_tr, eval_set=[(Xv, yv)],
              early_stopping_rounds=20, verbose=False)
        return -accuracy_score(yv, m.predict(Xv))
    s = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    s.optimize(obj, n_trials=n_trials, timeout=240, show_progress_bar=False)
    return s.best_params


def _tune_lgb_reg(X, y, n_trials):
    def obj(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 400),
                 num_leaves=trial.suggest_int("num_leaves", 20, 80),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                 n_jobs=2, random_state=RANDOM_STATE, verbose=-1)
        Xtr, ytr, Xv, yv = _val_split(X, y)
        m = lgb.LGBMRegressor(**p)
        m.fit(Xtr, ytr)
        return float(np.sqrt(mean_squared_error(yv, m.predict(Xv))))
    s = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    s.optimize(obj, n_trials=n_trials, timeout=150, show_progress_bar=False)
    return s.best_params


def _tune_lgb_clf(X, y, n_trials):
    def obj(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 400),
                 num_leaves=trial.suggest_int("num_leaves", 20, 80),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                 class_weight="balanced",
                 n_jobs=2, random_state=RANDOM_STATE, verbose=-1)
        Xtr, ytr, Xv, yv = _val_split(X, y)
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr)
        return -accuracy_score(yv, m.predict(Xv))
    s = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    s.optimize(obj, n_trials=n_trials, timeout=240, show_progress_bar=False)
    return s.best_params


def _tune_cb_reg(X, y, n_trials):
    def obj(trial):
        p = dict(iterations=trial.suggest_int("iterations", 50, 300),
                 depth=trial.suggest_int("depth", 3, 8),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 10, log=True),
                 random_seed=RANDOM_STATE, verbose=0, thread_count=2)
        Xtr, ytr, Xv, yv = _val_split(X, y)
        m = cb.CatBoostRegressor(**p)
        m.fit(Xtr, ytr)
        return float(np.sqrt(mean_squared_error(yv, m.predict(Xv))))
    s = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    s.optimize(obj, n_trials=n_trials, timeout=180, show_progress_bar=False)
    return s.best_params


def _tune_cb_clf(X, y, n_trials, class_weights: list | None = None):
    def obj(trial):
        p = dict(iterations=trial.suggest_int("iterations", 50, 100),
                 depth=trial.suggest_int("depth", 3, 6),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 10, log=True),
                 random_seed=RANDOM_STATE, verbose=0,
                 thread_count=2,
                 loss_function="MultiClass")
        if class_weights is not None:
            p["class_weights"] = class_weights
        Xtr, ytr, Xv, yv = _val_split(X, y)
        m = cb.CatBoostClassifier(**p)
        m.fit(Xtr, ytr, eval_set=(Xv, yv), early_stopping_rounds=20)
        return -accuracy_score(yv, m.predict(Xv).ravel())
    s = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    s.optimize(obj, n_trials=n_trials, timeout=300, show_progress_bar=False)
    return s.best_params


# ── Feature importance for individual models ───────────────────────────────────

def _eval_individual_regressors(
    Xtr, ytr, Xte, yte, test_dates, feature_cols,
    xgb_p, lgb_p, cb_p,
) -> dict:
    """Train individual L1 regressors on full training set, evaluate on test."""
    results = {}
    configs = [
        ("XGBoost",  xgb.XGBRegressor(**{**xgb_p, "random_state": RANDOM_STATE, "verbosity": 0})),
        ("LightGBM", lgb.LGBMRegressor(**{**lgb_p, "random_state": RANDOM_STATE, "verbose": -1, "n_jobs": 1, "deterministic": True})),
        ("CatBoost", cb.CatBoostRegressor(**{**cb_p, "random_seed": RANDOM_STATE, "verbose": 0, "thread_count": 1})),
    ]
    for name, model in configs:
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        results[name] = {
            "model": model,
            "predictions": preds,
            "y_test": yte,
            "test_dates": test_dates,
            "metrics": {
                "RMSE": float(np.sqrt(mean_squared_error(yte, preds))),
                "MAE":  float(mean_absolute_error(yte, preds)),
                "MAPE": _mape(yte, preds),
                "Directional_Accuracy": directional_accuracy(yte, preds),
            },
            "feature_importance": _get_importance(model, name, feature_cols),
        }
    return results


def _eval_individual_classifiers(
    Xtr, ytr, Xte, yte, test_dates, feature_cols,
    xgb_p, lgb_p, cb_p,
    sample_weight=None,
    class_weights_list=None,
) -> dict:
    """Train individual L1 classifiers with class balancing + SMOTE for diagnostics."""
    from src.ensemble import _smote_resample
    Xtr_s, ytr_s = _smote_resample(Xtr, ytr, random_state=RANDOM_STATE)
    # If SMOTE fired, class_weight/sample_weight already handled by resampling
    sw = None if (Xtr_s.shape[0] != Xtr.shape[0]) else sample_weight
    cw_list = None if (Xtr_s.shape[0] != Xtr.shape[0]) else class_weights_list

    results = {}
    configs = [
        ("XGBoost",  xgb.XGBClassifier(**{**xgb_p, "random_state": RANDOM_STATE, "verbosity": 0})),
        ("LightGBM", lgb.LGBMClassifier(**{**lgb_p, "random_state": RANDOM_STATE, "verbose": -1,
                                           "n_jobs": 1, "deterministic": True,
                                           **({"class_weight": "balanced"} if sw is not None else {})})),
        ("CatBoost", cb.CatBoostClassifier(
            **{**cb_p, "random_seed": RANDOM_STATE, "verbose": 0, "loss_function": "MultiClass",
               "thread_count": 1,
               **({"class_weights": cw_list} if cw_list else {})})),
    ]
    for name, model in configs:
        if name == "XGBoost" and sw is not None:
            model.fit(Xtr_s, ytr_s, sample_weight=sw)
        else:
            model.fit(Xtr_s, ytr_s)
        preds = np.array(model.predict(Xte)).ravel().astype(int)
        results[name] = {
            "model": model,
            "predictions": preds,
            "probabilities": _get_proba(model, Xte),
            "y_test": yte,
            "test_dates": test_dates,
            "metrics": {"Accuracy": float(accuracy_score(yte, preds))},
            "feature_importance": _get_importance(model, name, feature_cols),
        }
    return results


# ── Public training API ────────────────────────────────────────────────────────

def train_all_models(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    n_trials: int = N_TRIALS,
    progress_callback=None,
    pretrained_hyperparams: dict | None = None,
):
    """
    Full training pipeline including Optuna tuning and stacking ensemble.

    Parameters
    ----------
    pretrained_hyperparams : optional dict with keys
        xgb_reg, lgb_reg, cb_reg, xgb_clf, lgb_clf, cb_clf.
        When provided, all six Optuna studies are skipped and the supplied
        params are used directly.  Used in rolling walk-forward validation
        to reuse hyperparameters tuned on the first window, keeping
        subsequent windows fast without changing any model logic.

    Returns
    -------
    reg_results, clf_results, feature_cols, stack_reg, stack_clf
    Tuned hyperparameters are stored in clf_results["_hyperparams"] for
    optional reuse by the caller.
    """
    np.random.seed(RANDOM_STATE)

    feature_cols = get_feature_columns(train_df)

    def log(msg):
        if progress_callback:
            progress_callback(msg)

    # ── Prepare arrays ─────────────────────────────────────────────────────────
    # Regression uses Target_Close (secondary — informational)
    # Classification uses Target_Direction (±1%, primary directional target)
    clf_target = "Target_Direction" if "Target_Direction" in train_df.columns else "Target_Signal"

    tr_reg = train_df[feature_cols + ["Target_Close"]].dropna()
    te_reg = test_df[feature_cols + ["Target_Close"]].dropna()
    tr_clf = train_df[feature_cols + [clf_target]].dropna()
    te_clf = test_df[feature_cols + [clf_target]].dropna()

    Xtr_r, ytr_r = tr_reg[feature_cols].values, tr_reg["Target_Close"].values
    Xte_r, yte_r = te_reg[feature_cols].values, te_reg["Target_Close"].values
    Xtr_c, ytr_c = tr_clf[feature_cols].values, tr_clf[clf_target].values.astype(int)
    Xte_c, yte_c = te_clf[feature_cols].values, te_clf[clf_target].values.astype(int)
    dates_r = te_reg.index
    dates_c = te_clf.index

    log(f"Using classification target: {clf_target} "
        f"(class dist: DOWN={np.mean(ytr_c==0):.1%} "
        f"SIDE={np.mean(ytr_c==1):.1%} UP={np.mean(ytr_c==2):.1%})")

    # ── Class balancing ───────────────────────────────────────────────────────
    classes   = np.unique(ytr_c)
    weights   = compute_class_weight("balanced", classes=classes, y=ytr_c)
    cw_dict   = dict(zip(classes.astype(int), weights))
    sw_train  = np.array([cw_dict[y] for y in ytr_c])
    cb_cw_list = [cw_dict.get(c, 1.0) for c in range(3)]  # [w_DOWN, w_SIDE, w_UP]

    # ── Optuna hyperparameter search (or reuse pretrained params) ─────────────
    if pretrained_hyperparams is not None:
        log("Using pre-tuned hyperparameters — Optuna skipped.")
        xgb_reg_p = pretrained_hyperparams.get("xgb_reg", {})
        lgb_reg_p = pretrained_hyperparams.get("lgb_reg", {})
        cb_reg_p  = pretrained_hyperparams.get("cb_reg",  {})
        xgb_clf_p = pretrained_hyperparams.get("xgb_clf", {})
        lgb_clf_p = pretrained_hyperparams.get("lgb_clf", {})
        cb_clf_p  = pretrained_hyperparams.get("cb_clf",  {})
    else:
        log("Optuna tuning – XGBoost regression…")
        xgb_reg_p  = _tune_xgb_reg(Xtr_r, ytr_r, n_trials)
        log("Optuna tuning – LightGBM regression…")
        lgb_reg_p  = _tune_lgb_reg(Xtr_r, ytr_r, n_trials)
        log("Optuna tuning – CatBoost regression…")
        cb_reg_p   = _tune_cb_reg(Xtr_r, ytr_r, n_trials)

        log("Optuna tuning – XGBoost classification (class-balanced)…")
        xgb_clf_p  = _tune_xgb_clf(Xtr_c, ytr_c, n_trials, sample_weight=sw_train)
        log("Optuna tuning – LightGBM classification (class-balanced)…")
        lgb_clf_p  = _tune_lgb_clf(Xtr_c, ytr_c, n_trials)
        log("Optuna tuning – CatBoost classification (class-balanced)…")
        cb_clf_p   = _tune_cb_clf(Xtr_c, ytr_c, n_trials, class_weights=cb_cw_list)

    # ── Individual model diagnostics ──────────────────────────────────────────
    log("Training individual regression models…")
    reg_results = _eval_individual_regressors(
        Xtr_r, ytr_r, Xte_r, yte_r, dates_r, feature_cols,
        xgb_reg_p, lgb_reg_p, cb_reg_p,
    )

    log("Training individual classification models (class-balanced)…")
    clf_results = _eval_individual_classifiers(
        Xtr_c, ytr_c, Xte_c, yte_c, dates_c, feature_cols,
        xgb_clf_p, lgb_clf_p, cb_clf_p,
        sample_weight=sw_train,
        class_weights_list=cb_cw_list,
    )

    # ── Stacking ensembles ────────────────────────────────────────────────────
    log("Building regression stacking ensemble (OOF)…")
    stack_reg = StackingRegressor()
    stack_reg.fit(
        Xtr_r, ytr_r,
        model_factories={
            "XGBoost":  make_xgb_reg_factory(xgb_reg_p),
            "LightGBM": make_lgb_reg_factory(lgb_reg_p),
            "CatBoost": make_cb_reg_factory(cb_reg_p),
        },
        progress_cb=log,
    )
    stack_preds = stack_reg.predict(Xte_r)
    reg_results["Stacking"] = {
        "predictions": stack_preds,
        "y_test": yte_r,
        "test_dates": dates_r,
        "metrics": {
            "RMSE": float(np.sqrt(mean_squared_error(yte_r, stack_preds))),
            "MAE":  float(mean_absolute_error(yte_r, stack_preds)),
            "MAPE": _mape(yte_r, stack_preds),
            "Directional_Accuracy": directional_accuracy(yte_r, stack_preds),
        },
    }

    log("Building calibrated classification stacking ensemble (OOF, class-balanced)…")
    stack_clf = StackingClassifier()
    stack_clf.fit(
        Xtr_c, ytr_c,
        model_factories={
            "XGBoost":  make_xgb_clf_factory(xgb_clf_p),
            "LightGBM": make_lgb_clf_factory(lgb_clf_p),
            "CatBoost": make_cb_clf_factory(cb_clf_p, class_weights=cb_cw_list),
        },
        progress_cb=log,
        sample_weight=sw_train,
    )
    stack_clf_preds = stack_clf.predict(Xte_c)
    stack_clf_proba = stack_clf.predict_proba(Xte_c)

    # ── Threshold optimisation on OOF probabilities ───────────────────────────
    log("Optimising confidence thresholds on OOF data…")
    thresholds = {"threshold_up": 0.40, "threshold_down": 0.33}
    try:
        from src.calibration import optimize_threshold, compute_brier_scores
        oof_y = getattr(stack_clf, "_oof_y", None)
        oof_p = getattr(stack_clf, "_oof_proba", None)
        if oof_y is not None and oof_p is not None and len(oof_y) > 20:
            oof_meta_p = stack_clf._meta_clf.predict_proba(oof_p)
            # profit_proxy metric: 50% dir-acc + 30% DOWN-recall + 20% trade-freq
            # Separate grids: DOWN explores 0.25–0.42 (calibration shows this
            # range has 67–100% actual DOWN rate); UP uses wider 0.33–0.65 range
            thresholds = optimize_threshold(
                oof_y, oof_meta_p,
                metric="profit_proxy",
            )
            log(f"Optimal thresholds: UP>{thresholds['threshold_up']:.3f} "
                f"DOWN>{thresholds['threshold_down']:.3f} "
                f"(profit-proxy score={thresholds['best_score']:.4f})")
    except Exception as exc:
        log(f"Threshold optimisation skipped: {exc}")

    clf_results["Stacking"] = {
        "model": stack_clf,
        "predictions": stack_clf_preds,
        "probabilities": stack_clf_proba,
        "y_test": yte_c,
        "test_dates": dates_c,
        "metrics": {"Accuracy": float(accuracy_score(yte_c, stack_clf_preds))},
        "thresholds": thresholds,
        "clf_target": clf_target,
    }

    # Store tuned hyperparameters so rolling-validation callers can reuse them
    clf_results["_hyperparams"] = {
        "xgb_reg": xgb_reg_p, "lgb_reg": lgb_reg_p, "cb_reg": cb_reg_p,
        "xgb_clf": xgb_clf_p, "lgb_clf": lgb_clf_p, "cb_clf": cb_clf_p,
    }

    log("Training complete.")
    return reg_results, clf_results, feature_cols, stack_reg, stack_clf


# ── Backwards-compat wrappers (called by older app code paths) ────────────────

def train_regression_models(train_df, test_df, n_trials=N_TRIALS, progress_callback=None):
    feature_cols = get_feature_columns(train_df)
    tr = train_df[feature_cols + ["Target_Close"]].dropna()
    te = test_df[feature_cols + ["Target_Close"]].dropna()
    Xtr, ytr = tr[feature_cols].values, tr["Target_Close"].values
    Xte, yte = te[feature_cols].values, te["Target_Close"].values
    p1 = _tune_xgb_reg(Xtr, ytr, n_trials)
    p2 = _tune_lgb_reg(Xtr, ytr, n_trials)
    p3 = _tune_cb_reg(Xtr, ytr, n_trials)
    return _eval_individual_regressors(
        Xtr, ytr, Xte, yte, te.index, feature_cols, p1, p2, p3,
    ), feature_cols


def train_classification_models(train_df, test_df, n_trials=N_TRIALS, progress_callback=None):
    feature_cols = get_feature_columns(train_df)
    tgt = "Target_Direction" if "Target_Direction" in train_df.columns else "Target_Signal"
    tr = train_df[feature_cols + [tgt]].dropna()
    te = test_df[feature_cols + [tgt]].dropna()
    Xtr, ytr = tr[feature_cols].values, tr[tgt].values.astype(int)
    Xte, yte = te[feature_cols].values, te[tgt].values.astype(int)
    sw = np.array([{c: w for c, w in zip(
        *[np.unique(ytr), compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)]
    )}[y] for y in ytr])
    p1 = _tune_xgb_clf(Xtr, ytr, n_trials, sample_weight=sw)
    p2 = _tune_lgb_clf(Xtr, ytr, n_trials)
    p3 = _tune_cb_clf(Xtr, ytr, n_trials)
    return _eval_individual_classifiers(
        Xtr, ytr, Xte, yte, te.index, feature_cols, p1, p2, p3, sample_weight=sw,
    ), feature_cols


# ── Persistence ────────────────────────────────────────────────────────────────

def save_models(reg_results, clf_results, feature_cols,
                stack_reg=None, stack_clf=None):
    os.makedirs(MODELS_DIR, exist_ok=True)
    payload = {
        "reg_results":  reg_results,
        "clf_results":  clf_results,
        "feature_cols": feature_cols,
        "stack_reg":    stack_reg,
        "stack_clf":    stack_clf,
    }
    with open(os.path.join(MODELS_DIR, "models.pkl"), "wb") as f:
        pickle.dump(payload, f)


def load_models():
    path = os.path.join(MODELS_DIR, "models.pkl")
    if not os.path.exists(path):
        # Legacy: try old separate files
        old_paths = [
            os.path.join(MODELS_DIR, "reg.pkl"),
            os.path.join(MODELS_DIR, "clf.pkl"),
            os.path.join(MODELS_DIR, "features.pkl"),
        ]
        if all(os.path.exists(p) for p in old_paths):
            with open(old_paths[0], "rb") as f: reg = pickle.load(f)
            with open(old_paths[1], "rb") as f: clf = pickle.load(f)
            with open(old_paths[2], "rb") as f: feat = pickle.load(f)
            return reg, clf, feat, None, None
        return None, None, None, None, None

    with open(path, "rb") as f:
        d = pickle.load(f)
    return (d["reg_results"], d["clf_results"], d["feature_cols"],
            d.get("stack_reg"), d.get("stack_clf"))
