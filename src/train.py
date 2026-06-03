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

from src.config import (
    MODELS_DIR, N_TRIALS, RANDOM_STATE,
    STACKING_CV_FOLDS, CALIBRATION_CV_FOLDS,
)
from src.features import get_feature_columns
from src.ensemble import (
    StackingRegressor, StackingClassifier,
    make_xgb_reg_factory, make_lgb_reg_factory, make_cb_reg_factory,
    make_xgb_clf_factory, make_lgb_clf_factory, make_cb_clf_factory,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

import json
from datetime import datetime as _dt


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
    cw_dict=None,
) -> dict:
    """Train individual L1 classifiers with class balancing + SMOTE for diagnostics."""
    from src.ensemble import _smote_resample
    Xtr_s, ytr_s = _smote_resample(Xtr, ytr, random_state=RANDOM_STATE)

    # Always apply class weights — recompute sample_weight on SMOTE output when SMOTE fires
    smote_fired = Xtr_s.shape[0] != Xtr.shape[0]
    if smote_fired and cw_dict is not None:
        sw = np.array([cw_dict.get(int(y), 1.0) for y in ytr_s])
    else:
        sw = sample_weight
    cw_list = class_weights_list  # always pass class weights to CatBoost

    results = {}
    configs = [
        ("XGBoost",  xgb.XGBClassifier(**{**xgb_p, "random_state": RANDOM_STATE, "verbosity": 0})),
        ("LightGBM", lgb.LGBMClassifier(**{**lgb_p, "random_state": RANDOM_STATE, "verbose": -1,
                                           "n_jobs": 1, "deterministic": True,
                                           "class_weight": "balanced"})),
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
    fast_retrain: bool = False,
    sideways_weight_boost: float = 1.30,
):
    """
    Full training pipeline including Optuna tuning and stacking ensemble.

    Parameters
    ----------
    pretrained_hyperparams : optional dict with keys
        xgb_reg, lgb_reg, cb_reg, xgb_clf, lgb_clf, cb_clf.
        When provided, all six Optuna studies are skipped and the supplied
        params are used directly.
    fast_retrain : bool
        When True (used with pretrained_hyperparams), applies four speed
        optimisations for the auto-retrain path on Streamlit Cloud:
          1. Caps LightGBM n_estimators at 100 (Optuna may tune to 200+).
          2. Reduces stacking OOF folds from 5 → 3.
          3. Reduces calibration CV folds from 3 → 2.
          4. Feature engineering results are cached upstream by Streamlit;
             no re-computation occurs on unchanged data.
        The Optuna full-Train path is completely unaffected.

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

    # Configurable SIDEWAYS penalty boost (default 1.30 = 30% extra weight).
    # Pass sideways_weight_boost=1.0 to disable when DOWN recall is the priority.
    if 1 in cw_dict and sideways_weight_boost != 1.0:
        cw_dict[1] = cw_dict[1] * sideways_weight_boost

    sw_train  = np.array([cw_dict[y] for y in ytr_c])
    cb_cw_list = [float(cw_dict.get(c, 1.0)) for c in range(3)]  # [w_DOWN, w_SIDE, w_UP]

    # ── Optuna hyperparameter search (or reuse pretrained params) ─────────────
    if pretrained_hyperparams is not None:
        log("Using pre-tuned hyperparameters — Optuna skipped.")
        xgb_reg_p = pretrained_hyperparams.get("xgb_reg", {})
        lgb_reg_p = pretrained_hyperparams.get("lgb_reg", {})
        cb_reg_p  = pretrained_hyperparams.get("cb_reg",  {})
        xgb_clf_p = pretrained_hyperparams.get("xgb_clf", {})
        lgb_clf_p = pretrained_hyperparams.get("lgb_clf", {})
        cb_clf_p  = pretrained_hyperparams.get("cb_clf",  {})

        # ── Fast-retrain optimisations (Optuna path left unchanged) ───────────
        if fast_retrain:
            # 1. Cap LightGBM n_estimators at 100 — Optuna may have set 200+.
            #    Equivalent effect to early stopping: fewer trees = faster fit
            #    with minimal accuracy loss since the model is already warm-started
            #    from tuned hyperparams. XGB and CB are typically already low.
            _FAST_MAX_EST = 100
            lgb_reg_p = {**lgb_reg_p,
                         "n_estimators": min(lgb_reg_p.get("n_estimators", 200), _FAST_MAX_EST)}
            lgb_clf_p = {**lgb_clf_p,
                         "n_estimators": min(lgb_clf_p.get("n_estimators", 200), _FAST_MAX_EST)}
            log(f"Fast-retrain: LGB estimators capped at {_FAST_MAX_EST} "
                f"(reg {lgb_reg_p['n_estimators']}, clf {lgb_clf_p['n_estimators']})")
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
        cw_dict=cw_dict,
    )

    # ── Stacking ensembles ────────────────────────────────────────────────────
    # 2. Reduce OOF folds 5→3 and 3. calibration CV folds 3→2 for fast retrain
    _stk_folds = 3 if fast_retrain else STACKING_CV_FOLDS
    _cal_folds  = 2 if fast_retrain else CALIBRATION_CV_FOLDS
    if fast_retrain:
        log(f"Fast-retrain: stacking OOF folds={_stk_folds}, cal_cv={_cal_folds}")

    log("Building regression stacking ensemble (OOF)…")
    stack_reg = StackingRegressor(cv_folds=_stk_folds)
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
    stack_clf = StackingClassifier(cv_folds=_stk_folds, cal_cv=_cal_folds)
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
    thresholds = {"threshold_up": 0.35, "threshold_down": 0.25}
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

    # ── Phase 4A: LSTM base model integration ─────────────────────────────────
    # Train LSTM after tree models, generate OOF probabilities, and augment the
    # L2 meta-learner input from 9 (3×tree×3-class) to 12 (+LSTM×3-class).
    # All failures are caught — tree-only ensemble always remains as fallback.
    try:
        log("Training LSTM base model (Phase 4A)…")
        from src.lstm_model import train_lstm, LSTMPredictor
        from sklearn.model_selection import TimeSeriesSplit as _TSS

        _lstm_obj, _lstm_scaler, _lstm_oof, _lstm_oof_mask = train_lstm(
            Xtr_c, ytr_c,
            fast=fast_retrain,
            cv_folds=_stk_folds,
            progress_cb=log,
        )

        # Rebuild the same oof_mask used inside StackingClassifier.fit() to
        # align tree OOF rows (already filtered) with LSTM OOF rows.
        _tscv_align = _TSS(n_splits=_stk_folds)
        _tree_mask  = np.zeros(len(Xtr_c), dtype=bool)
        for _, _vi in _tscv_align.split(Xtr_c):
            _tree_mask[_vi] = True
        _valid_idx = np.where(_tree_mask)[0]  # original row indices with valid OOF

        # LSTM OOF for those same indices
        _lstm_valid = _lstm_oof[_valid_idx]                          # (n_valid, 3)
        _both_ok    = ~np.isnan(_lstm_valid).any(axis=1)             # rows valid in both

        if _both_ok.sum() > 30:
            # Augmented feature matrix: tree OOF (9 cols) + LSTM OOF (3 cols) = 12
            _tree_oof  = stack_clf._oof_proba                        # (n_valid, 9)
            _aug_oof   = np.hstack([_tree_oof[_both_ok], _lstm_valid[_both_ok]])
            _aug_y     = stack_clf._oof_y[_both_ok]

            _aug_meta = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                random_state=RANDOM_STATE, verbosity=0,
            )
            _aug_meta.fit(_aug_oof, _aug_y)

            # Persist LSTM model and scaler to separate files
            _lstm_path   = os.path.join(MODELS_DIR, "lstm_model.keras")
            _scaler_path = os.path.join(MODELS_DIR, "lstm_scaler.pkl")
            _lstm_obj.save(_lstm_path, _scaler_path)

            # Attach augmented meta-clf to stack_clf (XGB is pickle-safe)
            stack_clf._augmented_meta_clf = _aug_meta

            # Store lazy-loading predictor in clf_results for inference
            _lstm_pred = LSTMPredictor(_lstm_path, _scaler_path)
            clf_results["Stacking"]["lstm_predictor"] = _lstm_pred

            log(f"LSTM integrated: augmented meta-clf trained on {_both_ok.sum()} OOF samples.")
        else:
            log(f"LSTM OOF insufficient ({_both_ok.sum()} valid rows) — skipping augmentation.")

    except Exception as _lstm_exc:
        log(f"LSTM training skipped (tree ensemble unchanged): {_lstm_exc}")

    # ── Regime-conditional models (4D) ────────────────────────────────────────
    _regime_models: dict = {}
    try:
        from src.features import classify_regime as _classify_regime
        log("Classifying regimes and training regime-conditional models…")
        _clf_target = clf_results["Stacking"]["clf_target"]
        _train_with_regime = _classify_regime(train_df, window=20) if "regime" not in train_df.columns else train_df
        _regime_models = train_regime_models(_train_with_regime, feature_cols, _clf_target)
        _r_counts = {r: (int(((_train_with_regime["regime"] == r).sum())) if "regime" in _train_with_regime.columns else 0)
                     for r in ["high_vol", "trending", "neutral"]}
        log(f"Regime counts — " + " · ".join(f"{r}={n}" for r, n in _r_counts.items()))
    except Exception as _re_exc:
        log(f"Regime models skipped: {_re_exc}")

    # ── Training stats for drift detection (4E) ────────────────────────────────
    _training_stats: dict = {}
    try:
        _training_stats = compute_training_stats(train_df, feature_cols)
        _stats_path = os.path.join(MODELS_DIR, "training_stats.json")
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(_stats_path, "w") as _sf:
            json.dump(_training_stats, _sf)
        log(f"Training stats saved ({len(_training_stats)} features).")
    except Exception as _ts_exc:
        log(f"Training stats skipped: {_ts_exc}")

    clf_results["_regime_models"]   = _regime_models
    clf_results["_training_stats"]  = _training_stats

    log("Training complete.")
    return reg_results, clf_results, feature_cols, stack_reg, stack_clf


# ── Regime-conditional models ──────────────────────────────────────────────────

def train_regime_models(df: pd.DataFrame, feature_cols: list, target_col: str) -> dict:
    """
    Train three LightGBM models — one per market regime (high_vol, trending, neutral).
    Returns dict {regime: model_or_None}.
    """
    regime_models = {}
    for regime in ["high_vol", "trending", "neutral"]:
        try:
            mask = df["regime"] == regime
            n = int(mask.sum())
            if n < 60:
                print(f"Regime '{regime}': only {n} samples — skipped (min 60 required)")
                regime_models[regime] = None
                continue
            X_r = df.loc[mask, feature_cols].values
            y_r = df.loc[mask, target_col].values.astype(int)
            model = lgb.LGBMClassifier(
                class_weight="balanced",
                n_estimators=300,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=1,
            )
            model.fit(X_r, y_r)
            regime_models[regime] = model
            print(f"Regime '{regime}': trained on {n} samples")
        except Exception as exc:
            print(f"Regime '{regime}': training failed ({exc})")
            regime_models[regime] = None
    return regime_models


# ── Training stats for drift detection ────────────────────────────────────────

def compute_training_stats(df: pd.DataFrame, feature_cols: list) -> dict:
    """Compute mean/std/min/max per feature for live drift detection."""
    stats = {}
    try:
        sub = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        for col in feature_cols:
            s = sub[col].dropna()
            if len(s) == 0:
                continue
            stats[col] = {
                "mean": float(s.mean()),
                "std":  float(s.std()),
                "min":  float(s.min()),
                "max":  float(s.max()),
            }
    except Exception:
        pass
    return stats


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
        "reg_results":   reg_results,
        "clf_results":   clf_results,
        "feature_cols":  feature_cols,
        "stack_reg":     stack_reg,
        "stack_clf":     stack_clf,
        "regime_models": clf_results.get("_regime_models", {}),
        "thresholds":    (clf_results.get("Stacking") or {}).get("thresholds", {}),
        "trained_at":    _dt.now().isoformat(),
        "training_stats": clf_results.get("_training_stats", {}),
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


# ── Silver / Platinum model training ──────────────────────────────────────────

def _get_default_metal_hyperparams() -> dict:
    """Fixed reasonable hyperparams for Silver/Platinum (skips Optuna for speed)."""
    return {
        "xgb_reg": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8,
                    "reg_alpha": 0.1, "reg_lambda": 1.0},
        "lgb_reg": {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8,
                    "reg_alpha": 0.1, "reg_lambda": 1.0},
        "cb_reg":  {"iterations": 100, "depth": 4, "learning_rate": 0.05,
                    "l2_leaf_reg": 1.0},
        "xgb_clf": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8},
        "lgb_clf": {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8},
        "cb_clf":  {"iterations": 100, "depth": 4, "learning_rate": 0.05,
                    "l2_leaf_reg": 1.0},
    }


def train_metal_model(ticker: str, metal_name: str,
                      progress_callback=None) -> dict | None:
    """
    Train a signal model for Silver or Platinum using the same pipeline as Gold
    but with default hyperparams (no Optuna) for speed (~5 min per metal).

    Returns a bundle dict or None on failure. Graceful degradation throughout —
    never crashes; any step failure returns None with a log message.
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    log(f"Fetching {metal_name} training data (5y + macro sidecars)…")
    try:
        from src.data_loader import download_metal_ohlcv
        raw = download_metal_ohlcv(ticker, years=5)
    except Exception as exc:
        log(f"Data fetch failed for {metal_name}: {exc}")
        return None

    if raw.empty or len(raw) < 200:
        log(f"Insufficient data for {metal_name} ({len(raw)} bars — need ≥200)")
        return None

    log(f"Building features for {metal_name} ({len(raw)} bars)…")
    try:
        from src.features import add_features, classify_regime
        df = add_features(raw)
        df = classify_regime(df)
    except Exception as exc:
        log(f"Feature engineering failed for {metal_name}: {exc}")
        return None

    clf_target  = ("Target_Direction" if "Target_Direction" in df.columns
                   else "Target_Signal")
    feat_cols   = get_feature_columns(df)
    keep        = feat_cols + [clf_target, "Target_Close"]
    available   = df[[c for c in keep if c in df.columns]].dropna()

    if len(available) < 200:
        log(f"Insufficient usable rows for {metal_name} "
            f"({len(available)} after dropna — need ≥200)")
        return None

    # Temporal 80/20 split (no shuffle — time-series)
    split    = int(len(available) * 0.80)
    train_df = available.iloc[:split]
    test_df  = available.iloc[split:]
    n_train, n_test = len(train_df), len(test_df)
    log(f"{metal_name}: {n_train} train rows, {n_test} test rows, "
        f"{len(feat_cols)} features")

    if n_train < 100 or n_test < 20:
        log(f"Split too small for {metal_name} "
            f"(train={n_train}, test={n_test})")
        return None

    _regime_counts = {}
    if "regime" in df.columns:
        for _r in ["high_vol", "trending", "neutral"]:
            _regime_counts[_r] = int(
                (df["regime"].iloc[:split] == _r).sum())
        log("Regime counts — "
            + " · ".join(f"{r}={n}" for r, n in _regime_counts.items()))

    log(f"Training {metal_name} ensemble (default hyperparams, fast mode)…")
    try:
        reg_r, clf_r, feat, sr, sc = train_all_models(
            train_df.copy(), test_df.copy(),
            n_trials=10,
            pretrained_hyperparams=_get_default_metal_hyperparams(),
            fast_retrain=True,
            sideways_weight_boost=1.0,
            progress_callback=log,
        )
    except Exception as exc:
        log(f"Ensemble training failed for {metal_name}: {exc}")
        return None

    # Evaluate
    stk    = clf_r.get("Stacking", {})
    y_true = np.array(stk.get("y_test",       []))
    y_pred = np.array(stk.get("predictions",  []))
    overall = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    per_cls: dict = {}
    for cls, lbl in [(0, "DOWN"), (1, "SIDEWAYS"), (2, "UP")]:
        mask = y_true == cls
        per_cls[lbl] = (float(np.mean(y_pred[mask] == cls))
                        if mask.sum() > 0 else 0.0)

    log(f"{metal_name} accuracy: {overall:.1%} overall | "
        + " | ".join(f"{l}: {p:.1%}" for l, p in per_cls.items()))

    return {
        "ticker":        ticker,
        "metal_name":    metal_name,
        "reg_results":   reg_r,
        "clf_results":   clf_r,
        "feature_cols":  feat,
        "stack_reg":     sr,
        "stack_clf":     sc,
        "thresholds":    {"threshold_down": 0.25, "threshold_up": 0.38},
        "trained_at":    _dt.now().isoformat(),
        "n_train":       n_train,
        "n_test":        n_test,
        "overall_acc":   overall,
        "per_class_acc": per_cls,
    }


def save_metal_models(payload: dict) -> None:
    """Save {silver: bundle, platinum: bundle} dict to models/metals_models.pkl."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "metals_models.pkl"), "wb") as f:
        pickle.dump(payload, f)


def load_metal_models() -> dict:
    """Load metals_models.pkl. Returns {} if not found or corrupted."""
    path = os.path.join(MODELS_DIR, "metals_models.pkl")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}
