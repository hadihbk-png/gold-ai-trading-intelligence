"""
Profile each stage of the fast retrain path — before and after optimisation.
Run from the project root: python profile_retrain.py
"""
import os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import download_data, get_train_test_split
from src.features import add_features
from src.macro_loader import add_macro_features
from src.features import get_feature_columns
from src.train import load_models, _eval_individual_regressors, _eval_individual_classifiers
from src.ensemble import (
    StackingRegressor, StackingClassifier,
    make_xgb_reg_factory, make_lgb_reg_factory, make_cb_reg_factory,
    make_xgb_clf_factory, make_lgb_clf_factory, make_cb_clf_factory,
)
from src.config import STACKING_CV_FOLDS, CALIBRATION_CV_FOLDS, INITIAL_CAPITAL
from sklearn.utils.class_weight import compute_class_weight


def _time_pipeline(label: str, hp: dict, stk_folds: int, cal_folds: int,
                   Xtr_r, ytr_r, Xte_r, yte_r, dates_r, feature_cols,
                   Xtr_c, ytr_c, Xte_c, yte_c, dates_c,
                   sw_train, cb_cw_list):
    """Time each stage for a given (folds, cal_cv) configuration."""
    _timings: list[tuple[str, float]] = []

    class T:
        def __init__(self, lbl): self.lbl = lbl
        def __enter__(self): self._t0 = time.perf_counter(); return self
        def __exit__(self, *_):
            e = time.perf_counter() - self._t0
            _timings.append((self.lbl, e))
            print(f"    {self.lbl:<55s} {e:6.2f}s")

    print(f"\n--- {label} ---")
    t0 = time.perf_counter()

    with T(f"Individual regressors (3 fits)"):
        _eval_individual_regressors(
            Xtr_r, ytr_r, Xte_r, yte_r, dates_r, feature_cols,
            hp["xgb_reg"], hp["lgb_reg"], hp["cb_reg"],
        )

    with T(f"Individual classifiers (3 fits + SMOTE)"):
        _eval_individual_classifiers(
            Xtr_c, ytr_c, Xte_c, yte_c, dates_c, feature_cols,
            hp["xgb_clf"], hp["lgb_clf"], hp["cb_clf"],
            sample_weight=sw_train, class_weights_list=cb_cw_list,
        )

    with T(f"Stacking regression OOF+retrain ({stk_folds} folds × 3 models)"):
        sr = StackingRegressor(cv_folds=stk_folds)
        sr.fit(Xtr_r, ytr_r, model_factories={
            "XGBoost":  make_xgb_reg_factory(hp["xgb_reg"]),
            "LightGBM": make_lgb_reg_factory(hp["lgb_reg"]),
            "CatBoost": make_cb_reg_factory(hp["cb_reg"]),
        })

    with T(f"Stacking clf OOF+retrain ({stk_folds} folds × 3 models × cal_cv={cal_folds})"):
        sc = StackingClassifier(cv_folds=stk_folds, cal_cv=cal_folds)
        sc.fit(Xtr_c, ytr_c, model_factories={
            "XGBoost":  make_xgb_clf_factory(hp["xgb_clf"]),
            "LightGBM": make_lgb_clf_factory(hp["lgb_clf"]),
            "CatBoost": make_cb_clf_factory(hp["cb_clf"], class_weights=cb_cw_list),
        }, sample_weight=sw_train)

    with T("Threshold optimisation"):
        try:
            from src.calibration import optimize_threshold
            oof_y = getattr(sc, "_oof_y", None)
            oof_p = getattr(sc, "_oof_proba", None)
            if oof_y is not None and oof_p is not None and len(oof_y) > 20:
                optimize_threshold(oof_y, sc._meta_clf.predict_proba(oof_p), metric="profit_proxy")
        except Exception:
            pass

    total = time.perf_counter() - t0
    print(f"  {'TOTAL':<55s} {total:6.2f}s")
    return _timings, total


# ── Setup ──────────────────────────────────────────────────────────────────────
print("\n=== Fast-Retrain Profiler ===\n")
print("Loading data and models...")
raw = download_data()
df  = add_features(raw)
df  = add_macro_features(df, None)
train_df, test_df = get_train_test_split(df)
_, clf_r, _, _, _ = load_models()

if clf_r is None or "_hyperparams" not in clf_r:
    print("No saved hyperparams — run full Train first."); sys.exit(1)

hp_orig = clf_r["_hyperparams"]

# Fast-retrain hyperparams: cap LGB n_estimators
_FAST_MAX_EST = 100
hp_fast = {
    "xgb_reg": hp_orig["xgb_reg"],
    "lgb_reg": {**hp_orig["lgb_reg"], "n_estimators": min(hp_orig["lgb_reg"].get("n_estimators", 200), _FAST_MAX_EST)},
    "cb_reg":  hp_orig["cb_reg"],
    "xgb_clf": hp_orig["xgb_clf"],
    "lgb_clf": {**hp_orig["lgb_clf"], "n_estimators": min(hp_orig["lgb_clf"].get("n_estimators", 200), _FAST_MAX_EST)},
    "cb_clf":  hp_orig["cb_clf"],
}

print(f"LGB clf estimators: {hp_orig['lgb_clf'].get('n_estimators')} -> {hp_fast['lgb_clf']['n_estimators']}")
print(f"LGB reg estimators: {hp_orig['lgb_reg'].get('n_estimators')} -> {hp_fast['lgb_reg']['n_estimators']}")

# Prepare arrays
feature_cols = get_feature_columns(train_df)
clf_target   = "Target_Direction" if "Target_Direction" in train_df.columns else "Target_Signal"
tr_clf = train_df[feature_cols + [clf_target]].dropna()
te_clf = test_df[feature_cols + [clf_target]].dropna()
tr_reg = train_df[feature_cols + ["Target_Close"]].dropna()
te_reg = test_df[feature_cols + ["Target_Close"]].dropna()
Xtr_c, ytr_c = tr_clf[feature_cols].values, tr_clf[clf_target].values.astype(int)
Xte_c, yte_c = te_clf[feature_cols].values,  te_clf[clf_target].values.astype(int)
Xtr_r, ytr_r = tr_reg[feature_cols].values, tr_reg["Target_Close"].values
Xte_r, yte_r = te_reg[feature_cols].values,  te_reg["Target_Close"].values
classes  = np.unique(ytr_c)
weights  = compute_class_weight("balanced", classes=classes, y=ytr_c)
cw_dict  = dict(zip(classes.astype(int), weights))
sw_train = np.array([cw_dict[y] for y in ytr_c])
cb_cw_list = [cw_dict.get(c, 1.0) for c in range(3)]

print(f"Train rows: {len(Xtr_c):,}  Test rows: {len(Xte_c):,}  Features: {len(feature_cols)}\n")

# ── Run original path ─────────────────────────────────────────────────────────
_, t_orig = _time_pipeline(
    f"ORIGINAL (folds={STACKING_CV_FOLDS}, cal_cv={CALIBRATION_CV_FOLDS}, LGB est={hp_orig['lgb_clf'].get('n_estimators')})",
    hp_orig, STACKING_CV_FOLDS, CALIBRATION_CV_FOLDS,
    Xtr_r, ytr_r, Xte_r, yte_r, te_reg.index, feature_cols,
    Xtr_c, ytr_c, Xte_c, yte_c, te_clf.index,
    sw_train, cb_cw_list,
)

# ── Run optimised path ────────────────────────────────────────────────────────
_, t_fast = _time_pipeline(
    f"OPTIMISED (folds=3, cal_cv=2, LGB est={hp_fast['lgb_clf']['n_estimators']})",
    hp_fast, 3, 2,
    Xtr_r, ytr_r, Xte_r, yte_r, te_reg.index, feature_cols,
    Xtr_c, ytr_c, Xte_c, yte_c, te_clf.index,
    sw_train, cb_cw_list,
)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"  Original total:  {t_orig:6.1f}s")
print(f"  Optimised total: {t_fast:6.1f}s")
print(f"  Speedup:         {t_orig/t_fast:6.1f}×  ({(1-t_fast/t_orig)*100:.0f}% faster)")
print(f"  Target:          < 50s")
print(f"  Result:          {'PASS ✓' if t_fast < 50 else 'FAIL — needs more optimisation'}")
print("="*55)
