"""LSTM-based sequence model for sepsis prediction.

Unlike the per-hour snapshot approach in train.py, this module treats each
patient's ICU stay as a TIME SERIES.  A sliding window of past hours is fed
to an LSTM so the model can learn temporal evolution — rising lactate,
deteriorating vitals — directly from the raw imputed values.

Pipeline:
    1. Split by hospital (A=train, B=val).
    2. Normalize features (fit scaler on A only).
    3. Build sliding-window sequences per patient.
    4. Train a 2-layer LSTM with class-weighted BCE loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    ALL_FEATURE_COLS,
    DEMOGRAPHIC_COLS,
    LABEL_COL,
    RANDOM_STATE,
    TIME_COL,
)
from src.train import split_by_hospital

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SEQ_LENGTH = 12
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
FC_HIDDEN_SIZE = 32

LSTM_FEATURE_COLS: list[str] = ALL_FEATURE_COLS + DEMOGRAPHIC_COLS


# ── Device selection ──────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    """Return MPS device on Apple Silicon when available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Sequence preparation ─────────────────────────────────────────────────────

def prepare_sequences(
    df: pd.DataFrame,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    feature_cols: list[str] | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], StandardScaler]:
    """Convert flat per-hour data into sliding-window sequences.

    For each patient with at least ``seq_length`` hours of data, every
    contiguous window of ``seq_length`` rows becomes one sample.  The
    label is the ``SepsisLabel`` at the **last** hour of each window.

    Parameters
    ----------
    df : pd.DataFrame
        Imputed hourly data with ``patient_id``, ``ICULOS``, and
        ``SepsisLabel`` columns.
    seq_length : int
        Number of hours in each lookback window.
    feature_cols : list[str] | None
        Columns to use as LSTM inputs.  Defaults to vitals + labs +
        demographics (no rolling/trend/missingness features).
    scaler : StandardScaler | None
        Pre-fit scaler for transforming features.  When ``None`` a new
        scaler is fit on *df*.  Pass the training scaler when preparing
        validation data to prevent leakage.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str], StandardScaler]
        ``(X, y, patient_ids, scaler)`` where:
        - X has shape ``(n_samples, seq_length, n_features)``
        - y has shape ``(n_samples,)``
        - patient_ids has length ``n_samples`` (one id per window)
        - scaler is the fitted StandardScaler (pass to val set)

    Raises
    ------
    KeyError
        If required columns are missing from *df*.
    ValueError
        If no valid sequences can be built.
    """
    if feature_cols is None:
        feature_cols = LSTM_FEATURE_COLS

    _validate_columns(df, feature_cols)

    if seq_length < 1:
        raise ValueError(f"seq_length must be >= 1, got {seq_length}")

    # ── Normalize ─────────────────────────────────────────────────────
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[feature_cols].values)

    scaled_values = scaler.transform(df[feature_cols].values)
    labels = df[LABEL_COL].values

    # Build a lookup from row position to scaled features / label.
    # We iterate per patient, so we need to know which rows belong
    # to each patient and their ICULOS ordering.
    df_indexed = df[["patient_id", TIME_COL]].copy()
    df_indexed["_row_idx"] = np.arange(len(df_indexed))

    X_windows: list[np.ndarray] = []
    y_windows: list[float] = []
    pid_windows: list[str] = []

    for patient_id, group in df_indexed.groupby("patient_id"):
        group_sorted = group.sort_values(TIME_COL)
        row_indices = group_sorted["_row_idx"].values

        n_hours = len(row_indices)
        if n_hours < seq_length:
            continue

        patient_features = scaled_values[row_indices]
        patient_labels = labels[row_indices]

        for start in range(n_hours - seq_length + 1):
            end = start + seq_length
            X_windows.append(patient_features[start:end])
            y_windows.append(patient_labels[end - 1])
            pid_windows.append(patient_id)

    if not X_windows:
        raise ValueError(
            f"No valid sequences found. All patients have fewer than "
            f"{seq_length} hours of data."
        )

    X = np.stack(X_windows, axis=0).astype(np.float32)
    y = np.array(y_windows, dtype=np.float32)

    return X, y, pid_windows, scaler


def _validate_columns(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Raise KeyError if required columns are missing."""
    required = set(feature_cols) | {"patient_id", TIME_COL, LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {sorted(missing)}")


# ── LSTM model ────────────────────────────────────────────────────────────────

class SepsisLSTM(nn.Module):
    """Two-layer LSTM for binary sepsis prediction.

    Architecture:
        LSTM(n_features -> 64, 2 layers, dropout=0.3)
        -> FC(64 -> 32) -> ReLU
        -> FC(32 -> 1) -> Sigmoid
    """

    def __init__(self, n_features: int, seq_length: int = DEFAULT_SEQ_LENGTH):
        super().__init__()
        self.seq_length = seq_length

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=LSTM_DROPOUT,
        )
        self.fc1 = nn.Linear(LSTM_HIDDEN_SIZE, FC_HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(FC_HIDDEN_SIZE, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_length, n_features)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` — predicted probability of sepsis.
        """
        # lstm_out: (batch, seq_length, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Use only the last time step's hidden state.
        last_hidden = lstm_out[:, -1, :]

        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)


def build_lstm_model(
    n_features: int,
    seq_length: int = DEFAULT_SEQ_LENGTH,
) -> SepsisLSTM:
    """Construct an untrained SepsisLSTM and move it to the best device.

    Parameters
    ----------
    n_features : int
        Number of input features per time step.
    seq_length : int
        Length of the input sequence (for documentation; does not
        constrain the forward pass).

    Returns
    -------
    SepsisLSTM
        Model on MPS (Apple Silicon) or CPU.
    """
    device = _get_device()
    model = SepsisLSTM(n_features=n_features, seq_length=seq_length)
    return model.to(device)


# ── Training ──────────────────────────────────────────────────────────────────

def _compute_pos_weight(y: np.ndarray) -> float:
    """Compute positive-class weight: count(neg) / count(pos)."""
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0:
        raise ValueError("No positive samples in training labels.")
    return n_neg / n_pos


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 0.001,
) -> tuple[SepsisLSTM, dict]:
    """Train the LSTM model with class-weighted binary cross-entropy.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training sequences and labels.
    X_val, y_val : np.ndarray
        Validation sequences and labels.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for DataLoader.
    lr : float
        Learning rate for Adam optimizer.

    Returns
    -------
    tuple[SepsisLSTM, dict]
        ``(model, history)`` where history contains lists keyed by
        ``"train_loss"``, ``"val_loss"``, and ``"val_auroc"``.
    """
    device = _get_device()
    n_features = X_train.shape[2]
    seq_length = X_train.shape[1]

    print(f"Device: {device}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Sequence shape: ({seq_length}, {n_features})")

    # ── Model ─────────────────────────────────────────────────────────
    model = build_lstm_model(n_features=n_features, seq_length=seq_length)

    # ── Loss with class imbalance weighting ───────────────────────────
    # The model outputs sigmoid probabilities, so we use BCELoss (not
    # BCEWithLogitsLoss).  reduction="none" lets us apply per-sample
    # pos_weight manually to handle the ~13:1 class imbalance.
    pos_weight_value = _compute_pos_weight(y_train)
    criterion = nn.BCELoss(reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── DataLoaders ───────────────────────────────────────────────────
    torch.manual_seed(RANDOM_STATE)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
    )

    # ── Training loop ─────────────────────────────────────────────────
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_auroc": [],
    }

    print(f"\nTraining for {epochs} epochs (pos_weight={pos_weight_value:.2f})")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        # — Train —
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch).squeeze(1)
            per_sample_loss = criterion(preds, y_batch)

            # Apply pos_weight: multiply loss for positive samples
            sample_weights = torch.where(
                y_batch == 1.0,
                torch.tensor(pos_weight_value, device=device),
                torch.tensor(1.0, device=device),
            )
            weighted_loss = (per_sample_loss * sample_weights).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            train_loss_sum += weighted_loss.item() * len(y_batch)
            train_samples += len(y_batch)

        epoch_train_loss = train_loss_sum / train_samples

        # — Validate —
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        val_preds_list: list[np.ndarray] = []
        val_labels_list: list[np.ndarray] = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                preds = model(X_batch).squeeze(1)
                per_sample_loss = criterion(preds, y_batch)

                sample_weights = torch.where(
                    y_batch == 1.0,
                    torch.tensor(pos_weight_value, device=device),
                    torch.tensor(1.0, device=device),
                )
                weighted_loss = (per_sample_loss * sample_weights).mean()

                val_loss_sum += weighted_loss.item() * len(y_batch)
                val_samples += len(y_batch)

                val_preds_list.append(preds.cpu().numpy())
                val_labels_list.append(y_batch.cpu().numpy())

        epoch_val_loss = val_loss_sum / val_samples
        val_preds_all = np.concatenate(val_preds_list)
        val_labels_all = np.concatenate(val_labels_list)
        epoch_val_auroc = roc_auc_score(val_labels_all, val_preds_all)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_auroc"].append(epoch_val_auroc)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val AUROC: {epoch_val_auroc:.4f}"
        )

    print("-" * 65)
    best_auroc = max(history["val_auroc"])
    best_epoch = history["val_auroc"].index(best_auroc) + 1
    print(f"Best Val AUROC: {best_auroc:.4f} (epoch {best_epoch})")

    return model, history


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_lstm(
    model: SepsisLSTM,
    X: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Generate sepsis probabilities from a trained LSTM.

    Parameters
    ----------
    model : SepsisLSTM
        Trained model (already on the correct device).
    X : np.ndarray
        Sequences with shape ``(n_samples, seq_length, n_features)``.
    batch_size : int
        Inference batch size.

    Returns
    -------
    np.ndarray
        Predicted probabilities, shape ``(n_samples,)``.
    """
    device = next(model.parameters()).device
    model.eval()

    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).squeeze(1)
            predictions.append(preds.cpu().numpy())

    return np.concatenate(predictions)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def train_lstm_pipeline(
    df: pd.DataFrame,
    seq_length: int = DEFAULT_SEQ_LENGTH,
) -> dict:
    """Run the complete LSTM training pipeline.

    Steps:
        1. Split by hospital (A=train, B=val).
        2. Fit StandardScaler on Hospital A features only.
        3. Build sliding-window sequences for both splits.
        4. Train the LSTM.

    Parameters
    ----------
    df : pd.DataFrame
        Imputed data with ``patient_id``, ``hospital``, ``ICULOS``,
        ``SepsisLabel``, and all feature columns.
    seq_length : int
        Sliding-window length in hours.

    Returns
    -------
    dict
        Keys: ``"model"``, ``"history"``, ``"X_train"``, ``"y_train"``,
        ``"X_val"``, ``"y_val"``, ``"val_patient_ids"``, ``"scaler"``,
        ``"seq_length"``.
    """
    print("=" * 60)
    print("SEPSIS PREDICTION — LSTM TRAINING PIPELINE")
    print("=" * 60)

    # 1. Hospital-based split
    print("\n[1/4] Splitting data by hospital ...")
    df_a, df_b = split_by_hospital(df)
    print(f"  Hospital A: {len(df_a):,} rows")
    print(f"  Hospital B: {len(df_b):,} rows")

    # 2. Build sequences (scaler fit on A only)
    print(f"\n[2/4] Building sequences (window={seq_length} hours) ...")
    X_train, y_train, train_pids, scaler = prepare_sequences(
        df_a, seq_length=seq_length,
    )
    X_val, y_val, val_pids, _ = prepare_sequences(
        df_b, seq_length=seq_length, scaler=scaler,
    )
    print(f"  Training sequences: {X_train.shape}")
    print(f"  Validation sequences: {X_val.shape}")

    n_pos_train = int(np.sum(y_train == 1))
    n_pos_val = int(np.sum(y_val == 1))
    print(f"  Train positive rate: {n_pos_train / len(y_train):.3f}")
    print(f"  Val positive rate:   {n_pos_val / len(y_val):.3f}")

    # 3. Train LSTM
    print("\n[3/4] Training LSTM ...")
    model, history = train_lstm(X_train, y_train, X_val, y_val)

    # 4. Summary
    print("\n[4/4] Pipeline complete.")
    print("=" * 60)

    return {
        "model": model,
        "history": history,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "val_patient_ids": val_pids,
        "scaler": scaler,
        "seq_length": seq_length,
    }
