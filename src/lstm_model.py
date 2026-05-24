"""
LSTM base model — Phase 4A Enhancement 2.

Two-layer stacked LSTM producing 3-class (DOWN/SIDEWAYS/UP) probability output.
Integrated as a 4th base model in the stacking ensemble: LSTM OOF probabilities
augment the 9-column L1 feature matrix fed to the L2 meta-learner (12 inputs total).

Gracefully degrades to tree-only ensemble if TensorFlow is unavailable or training fails.
"""
import os
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ── TensorFlow availability ────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel("ERROR")
    _TF_AVAILABLE = True
except Exception:
    pass

SEQ_LEN = 20  # sequence length in trading days


# ── Model builder ──────────────────────────────────────────────────────────────

def _build_model(n_features: int, units1: int, units2: int):
    """Two-layer stacked LSTM → Dense(3, softmax)."""
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow not installed")
    inp = keras.Input(shape=(SEQ_LEN, n_features), name="seq_in")
    x = keras.layers.LSTM(units1, return_sequences=True, name="lstm1")(inp)
    x = keras.layers.Dropout(0.2, name="drop1")(x)
    x = keras.layers.LSTM(units2, return_sequences=False, name="lstm2")(x)
    x = keras.layers.Dropout(0.2, name="drop2")(x)
    out = keras.layers.Dense(3, activation="softmax", name="direction")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Sequence preparation ───────────────────────────────────────────────────────

def prepare_sequences(X: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Convert 2D array (n, n_feat) → 3D (n-seq_len+1, seq_len, n_feat).
    Returns empty array when n < seq_len.
    """
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
    return np.stack(
        [X[i : i + seq_len] for i in range(n - seq_len + 1)]
    ).astype(np.float32)


def _oof_sequences(
    X_tr: np.ndarray,
    X_val: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """
    Build validation-set sequences using the last (seq_len-1) training rows as
    context, so each val sample has a full seq_len window.
    Returns shape (len(X_val), seq_len, n_feat).
    """
    ctx_len = seq_len - 1
    if len(X_tr) >= ctx_len:
        ctx = X_tr[-ctx_len:]
    else:
        pad = np.zeros((ctx_len - len(X_tr), X_tr.shape[1]), dtype=np.float32)
        ctx = np.vstack([pad, X_tr])
    extended = np.vstack([ctx, X_val]).astype(np.float32)
    return np.stack(
        [extended[i : i + seq_len] for i in range(len(X_val))]
    ).astype(np.float32)


# ── Training entry point ───────────────────────────────────────────────────────

def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fast: bool = False,
    cv_folds: int = 5,
    progress_cb=None,
) -> tuple:
    """
    Train LSTM with OOF generation using the same TimeSeriesSplit as tree models.

    Parameters
    ----------
    X_train     : 2D training features (n_samples, n_features)
    y_train     : integer class labels (0=DOWN, 1=SIDEWAYS, 2=UP)
    fast        : reduced architecture (32 units, 30 epochs, patience=5)
    cv_folds    : TimeSeriesSplit folds — must match the tree stacking fold count
    progress_cb : optional logging callback

    Returns
    -------
    lstm_wrapper : _LSTMWrapper — call .save(model_path, scaler_path) to persist
    scaler       : MinMaxScaler fitted on X_train
    oof_proba    : ndarray (n_train, 3) — NaN for indices without OOF coverage
    oof_mask     : boolean mask — True where oof_proba is valid
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow not available — LSTM training skipped")

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import TimeSeriesSplit

    tf.random.set_seed(42)
    log = progress_cb or (lambda *_: None)
    log("  LSTM: scaling features for sequence model…")

    scaler = MinMaxScaler()
    X_sc   = scaler.fit_transform(X_train).astype(np.float32)
    n_feat = X_sc.shape[1]

    units1     = 32 if fast else 64
    units2     = 32 if fast else 32
    max_epochs = 30 if fast else 100
    patience   = 5  if fast else 10

    tscv     = TimeSeriesSplit(n_splits=cv_folds)
    oof_prob = np.full((len(X_train), 3), np.nan, dtype=np.float32)
    oof_mask = np.zeros(len(X_train), dtype=bool)

    for fold_i, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        Xtr_s  = X_sc[tr_idx]
        Xval_s = X_sc[val_idx]
        ytr_f  = y_train[tr_idx]

        if len(Xtr_s) < SEQ_LEN + 5:
            log(f"  LSTM fold {fold_i+1}/{cv_folds}: too few rows — skipping")
            continue

        X_seq_tr  = prepare_sequences(Xtr_s, SEQ_LEN)
        y_seq_tr  = ytr_f[SEQ_LEN - 1 :]
        X_seq_val = _oof_sequences(Xtr_s, Xval_s, SEQ_LEN)

        m = _build_model(n_feat, units1, units2)
        cb_es = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0
        )
        try:
            m.fit(
                X_seq_tr, y_seq_tr,
                epochs=max_epochs, batch_size=32,
                validation_split=0.15,
                callbacks=[cb_es], verbose=0,
            )
            oof_prob[val_idx] = m.predict(X_seq_val, verbose=0)
            oof_mask[val_idx] = True
            log(f"  LSTM fold {fold_i+1}/{cv_folds} complete")
        except Exception as exc:
            log(f"  LSTM fold {fold_i+1}/{cv_folds} failed: {exc}")
        finally:
            del m
            keras.backend.clear_session()

    # ── Final LSTM on full training set ────────────────────────────────────────
    log("  LSTM: training final model on full dataset…")
    X_seq_full = prepare_sequences(X_sc, SEQ_LEN)
    y_seq_full = y_train[SEQ_LEN - 1 :]

    final_m = _build_model(n_feat, units1, units2)
    cb_es_f = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0
    )
    final_m.fit(
        X_seq_full, y_seq_full,
        epochs=max_epochs, batch_size=32,
        validation_split=0.15,
        callbacks=[cb_es_f], verbose=0,
    )

    wrapper = _LSTMWrapper(final_m, scaler, SEQ_LEN)
    log("  LSTM: training complete.")
    return wrapper, scaler, oof_prob, oof_mask


# ── Model wrapper (not pickle-safe — use LSTMPredictor for persistence) ────────

class _LSTMWrapper:
    """Holds a live Keras model + scaler. NOT pickle-safe; save to files first."""

    def __init__(self, model, scaler, seq_len: int = SEQ_LEN):
        self._model   = model
        self._scaler  = scaler
        self._seq_len = seq_len

    def save(self, model_path: str, scaler_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        self._model.save(model_path)
        with open(scaler_path, "wb") as fh:
            pickle.dump(self._scaler, fh)


# ── Pickle-safe lazy-loading predictor ────────────────────────────────────────

class LSTMPredictor:
    """
    Pickle-safe LSTM predictor — stores file paths, loads Keras model on demand.
    Safe to store in models.pkl; reloads the .keras file on first inference call.
    Returns None gracefully when TF is unavailable or model files are missing.
    """

    def __init__(self, model_path: str, scaler_path: str, seq_len: int = SEQ_LEN):
        self.model_path  = model_path
        self.scaler_path = scaler_path
        self.seq_len     = seq_len
        self._model  = None
        self._scaler = None

    # Pickle round-trip preserves only the file paths
    def __getstate__(self):
        return {
            "model_path":  self.model_path,
            "scaler_path": self.scaler_path,
            "seq_len":     self.seq_len,
        }

    def __setstate__(self, state):
        self.model_path  = state["model_path"]
        self.scaler_path = state["scaler_path"]
        self.seq_len     = state["seq_len"]
        self._model  = None
        self._scaler = None

    def _load(self) -> bool:
        if self._model is not None and self._scaler is not None:
            return True
        if not _TF_AVAILABLE:
            return False
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            return False
        try:
            from tensorflow import keras as _k
            self._model = _k.models.load_model(self.model_path)
            with open(self.scaler_path, "rb") as fh:
                self._scaler = pickle.load(fh)
            return True
        except Exception:
            return False

    @property
    def available(self) -> bool:
        """True if TF is installed and model files exist."""
        return self._load()

    def predict_proba_from_recent(self, X_recent: np.ndarray) -> "np.ndarray | None":
        """
        Predict using the most recent seq_len rows of X_recent.

        Parameters
        ----------
        X_recent : 2D array (n, n_features) with at least 1 row of raw features.
                   The predictor scales internally using the stored MinMaxScaler.

        Returns
        -------
        proba : (1, 3) probability array, or None on any failure.
        """
        if not self._load():
            return None
        try:
            n = len(X_recent)
            if n < self.seq_len:
                pad = np.zeros(
                    (self.seq_len - n, X_recent.shape[1]), dtype=np.float32
                )
                X_recent = np.vstack([pad, X_recent])
            seq  = X_recent[-self.seq_len :].astype(np.float32)
            X_sc = self._scaler.transform(seq).astype(np.float32)
            X_in = X_sc[np.newaxis, ...]  # (1, seq_len, n_features)
            return self._model.predict(X_in, verbose=0)  # (1, 3)
        except Exception:
            return None
