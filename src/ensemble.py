"""
Level-2 stacking ensemble (no data leakage via TimeSeriesSplit OOF).

Regression stack:
  L1 : XGBoost · LightGBM · CatBoost  (Optuna-tuned, retrained on full set)
  L2 : Ridge regression on OOF predictions

Classification stack:
  L1 : CalibratedClassifierCV(XGB) · Cal(LGB) · Cal(CB)
  L2 : XGBClassifier meta-learner on stacked OOF probabilities

Both ensembles expose a scikit-learn–style .predict() / .predict_proba() API.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

try:
    from imblearn.over_sampling import SMOTE as _SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False


def _smote_resample(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Apply SMOTE to (X, y) and return resampled arrays.
    Falls back silently if imblearn is unavailable or any class has < 2 samples.
    MUST only ever be called on training data — never on val or test splits.
    """
    if not _SMOTE_AVAILABLE:
        return X, y
    counts = np.bincount(y.astype(int))
    # Need at least k_neighbors+1 samples in every class for SMOTE
    min_count = int(counts[counts > 0].min())
    k = min(5, min_count - 1)
    if k < 1:
        return X, y
    try:
        sm = _SMOTE(random_state=random_state, k_neighbors=k)
        return sm.fit_resample(X, y)
    except Exception:
        return X, y

from src.config import (
    STACKING_CV_FOLDS, META_RIDGE_ALPHA,
    CALIBRATION_METHOD, CALIBRATION_CV_FOLDS,
    RANDOM_STATE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Regression stack
# ═══════════════════════════════════════════════════════════════════════════════

class StackingRegressor:
    """
    Two-level stacking regressor.

    Call fit_on_training(X_train, y_train, model_factories) to:
      1. Generate time-series OOF predictions from L1 models.
      2. Train Ridge meta-model on OOF predictions.
      3. Retrain L1 models on full training set.

    Then call predict(X_test) for final predictions.
    """

    def __init__(self, alpha: float = META_RIDGE_ALPHA, cv_folds: int = STACKING_CV_FOLDS):
        self.alpha    = alpha
        self.cv_folds = cv_folds
        self._l1_models: list = []
        self._l1_names:  list = []
        self._meta: Ridge | None  = None
        self._scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_factories: dict,      # name → callable() returning unfitted estimator
        progress_cb=None,
    ) -> "StackingRegressor":
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        n_models = len(model_factories)
        oof = np.zeros((len(X_train), n_models))
        oof_mask = np.zeros(len(X_train), dtype=bool)

        # ── OOF pass ──────────────────────────────────────────────────────────
        for fold_i, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
            Xval = X_train[val_idx]
            oof_mask[val_idx] = True

            for col_i, (name, factory) in enumerate(model_factories.items()):
                if progress_cb:
                    progress_cb(f"  Stacking regression fold {fold_i+1}/{self.cv_folds} – {name}")
                m = factory()
                m.fit(Xtr, ytr)
                oof[val_idx, col_i] = m.predict(Xval)

        # ── Train Ridge meta-model on OOF rows ────────────────────────────────
        oof_valid = self._scaler.fit_transform(oof[oof_mask])
        self._meta = Ridge(alpha=self.alpha)
        self._meta.fit(oof_valid, y_train[oof_mask])

        # ── Retrain L1 on full training set ───────────────────────────────────
        self._l1_models, self._l1_names = [], []
        for name, factory in model_factories.items():
            if progress_cb:
                progress_cb(f"  Retraining L1 {name} on full training set")
            m = factory()
            m.fit(X_train, y_train)
            self._l1_models.append(m)
            self._l1_names.append(name)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._meta is None:
            raise RuntimeError("Call fit first.")
        l1_preds = np.column_stack([m.predict(X) for m in self._l1_models])
        return self._meta.predict(self._scaler.transform(l1_preds))

    @property
    def meta_coefficients(self) -> pd.Series:
        if self._meta is None:
            return pd.Series()
        return pd.Series(self._meta.coef_, index=self._l1_names)


# ═══════════════════════════════════════════════════════════════════════════════
# Classification stack with probability calibration
# ═══════════════════════════════════════════════════════════════════════════════

class StackingClassifier:
    """
    Two-level calibrated stacking classifier.

    L1: XGB, LGB, CB each wrapped in CalibratedClassifierCV
        → output: well-calibrated 3-class probability vectors.
    L2: XGBClassifier trained on stacked OOF probabilities (9 features).
    """

    def __init__(self, cv_folds: int = STACKING_CV_FOLDS,
                 cal_method: str = CALIBRATION_METHOD,
                 cal_cv: int = CALIBRATION_CV_FOLDS):
        self.cv_folds   = cv_folds
        self.cal_method = cal_method
        self.cal_cv     = cal_cv
        self._cal_l1:   list = []
        self._cal_names: list = []
        self._meta_clf  = None
        self._n_classes = 3

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_factories: dict,
        progress_cb=None,
        sample_weight: np.ndarray | None = None,
    ) -> "StackingClassifier":
        tscv_oof = TimeSeriesSplit(n_splits=self.cv_folds)
        n_models = len(model_factories)
        oof_proba = np.zeros((len(X_train), n_models * self._n_classes))
        oof_mask  = np.zeros(len(X_train), dtype=bool)

        # ── OOF probability pass ──────────────────────────────────────────────
        for fold_i, (tr_idx, val_idx) in enumerate(tscv_oof.split(X_train)):
            Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
            Xval = X_train[val_idx]
            sw_tr = sample_weight[tr_idx] if sample_weight is not None else None
            oof_mask[val_idx] = True

            # SMOTE only on the training fold — val_idx rows are never touched
            Xtr_s, ytr_s = _smote_resample(Xtr, ytr, random_state=RANDOM_STATE)
            # When SMOTE succeeds, sample_weight is superseded by balanced resampling
            sw_fit = None if (Xtr_s.shape[0] != Xtr.shape[0]) else sw_tr

            for col_i, (name, factory) in enumerate(model_factories.items()):
                if progress_cb:
                    progress_cb(
                        f"  Stacking classification fold {fold_i+1}/{self.cv_folds} – {name}"
                    )
                tscv_cal = TimeSeriesSplit(n_splits=self.cal_cv)
                base = factory()
                cal  = CalibratedClassifierCV(
                    estimator=base,
                    method=self.cal_method,
                    cv=tscv_cal,
                )
                if sw_fit is not None:
                    try:
                        cal.fit(Xtr_s, ytr_s, sample_weight=sw_fit)
                    except TypeError:
                        cal.fit(Xtr_s, ytr_s)
                else:
                    cal.fit(Xtr_s, ytr_s)
                start = col_i * self._n_classes
                end   = start + self._n_classes
                proba = cal.predict_proba(Xval)
                if proba.shape[1] < self._n_classes:
                    # Expand if some classes missing in fold
                    full = np.zeros((len(Xval), self._n_classes))
                    for j, cls in enumerate(cal.classes_):
                        full[:, int(cls)] = proba[:, j]
                    proba = full
                oof_proba[val_idx, start:end] = proba

        # ── Train XGB meta-classifier on OOF probabilities ────────────────────
        valid_proba = oof_proba[oof_mask]
        valid_y     = y_train[oof_mask]
        # Store OOF data for post-training threshold optimisation
        self._oof_proba = valid_proba
        self._oof_y     = valid_y
        self._meta_clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE, verbosity=0,
        )
        self._meta_clf.fit(valid_proba, valid_y)

        # ── Retrain calibrated L1 on full training set (SMOTE on full train) ────
        X_train_s, y_train_s = _smote_resample(X_train, y_train, random_state=RANDOM_STATE)
        sw_full = None if (X_train_s.shape[0] != X_train.shape[0]) else sample_weight

        self._cal_l1, self._cal_names = [], []
        for name, factory in model_factories.items():
            if progress_cb:
                progress_cb(f"  Calibrating L1 {name} on full training set")
            tscv_cal = TimeSeriesSplit(n_splits=self.cal_cv)
            base = factory()
            cal  = CalibratedClassifierCV(
                estimator=base, method=self.cal_method, cv=tscv_cal,
            )
            if sw_full is not None:
                try:
                    cal.fit(X_train_s, y_train_s, sample_weight=sw_full)
                except TypeError:
                    cal.fit(X_train_s, y_train_s)
            else:
                cal.fit(X_train_s, y_train_s)
            self._cal_l1.append(cal)
            self._cal_names.append(name)

        return self

    def _l1_proba(self, X: np.ndarray) -> np.ndarray:
        parts = []
        for cal in self._cal_l1:
            p = cal.predict_proba(X)
            if p.shape[1] < self._n_classes:
                full = np.zeros((len(X), self._n_classes))
                for j, cls in enumerate(cal.classes_):
                    full[:, int(cls)] = p[:, j]
                p = full
            parts.append(p)
        return np.hstack(parts)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(self._meta_clf.predict(self._l1_proba(X))).ravel().astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns calibrated probability of each class from the meta-clf."""
        return self._meta_clf.predict_proba(self._l1_proba(X))

    def soft_vote_proba(self, X: np.ndarray) -> np.ndarray:
        """Average of calibrated L1 probabilities (soft voting)."""
        parts = []
        for cal in self._cal_l1:
            p = cal.predict_proba(X)
            if p.shape[1] < self._n_classes:
                full = np.zeros((len(X), self._n_classes))
                for j, cls in enumerate(cal.classes_):
                    full[:, int(cls)] = p[:, j]
                p = full
            parts.append(p)
        return np.mean(parts, axis=0)


# ── Model factory helpers ──────────────────────────────────────────────────────

def make_xgb_reg_factory(params: dict):
    def factory():
        p = {**params, "random_state": RANDOM_STATE, "verbosity": 0}
        return xgb.XGBRegressor(**p)
    return factory


def make_lgb_reg_factory(params: dict):
    def factory():
        p = {**params, "random_state": RANDOM_STATE, "verbose": -1,
             "n_jobs": 1, "deterministic": True}
        return lgb.LGBMRegressor(**p)
    return factory


def make_cb_reg_factory(params: dict):
    def factory():
        p = {**params, "random_seed": RANDOM_STATE, "verbose": 0, "thread_count": 1}
        return cb.CatBoostRegressor(**p)
    return factory


def make_xgb_clf_factory(params: dict):
    def factory():
        p = {**params, "random_state": RANDOM_STATE, "verbosity": 0}
        return xgb.XGBClassifier(**p)
    return factory


def make_lgb_clf_factory(params: dict):
    def factory():
        p = {**params, "random_state": RANDOM_STATE, "verbose": -1,
             "class_weight": "balanced", "n_jobs": 1, "deterministic": True}
        return lgb.LGBMClassifier(**p)
    return factory


def make_cb_clf_factory(params: dict, class_weights: list | None = None):
    def factory():
        p = {**params, "random_seed": RANDOM_STATE, "verbose": 0,
             "loss_function": "MultiClass", "thread_count": 1}
        if class_weights is not None:
            p["class_weights"] = class_weights
        return cb.CatBoostClassifier(**p)
    return factory
