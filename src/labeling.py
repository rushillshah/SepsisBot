"""Label/row filtering utilities for training-data variants.

This module is intentionally separate from imputation/features so the choice
of *which rows to train on* is decoupled from how features are computed.

The flagship utility is :func:`censor_post_onset`, which strips rows that
arrive after a patient has clinically transitioned to sepsis. It exists so
we can train a variant of the model that only sees the pre-onset trajectory
(the actual early-warning window) instead of being rewarded for recognising
already-septic states.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import EARLY_LABEL_EXTRA_HOURS


def censor_post_onset(
    X: pd.DataFrame,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    patient_ids: np.ndarray,
    iculos: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Drop post-onset rows from septic patients; keep non-septic patients intact.

    PhysioNet's ``SepsisLabel`` flips to 1 at ``t_onset - 6h`` (i.e. 6 hours
    before clinical onset) and stays 1 until discharge. Training on the full
    positive window mixes the early-warning signal with hours where the
    patient is already septic (easy positives), inflating headline metrics.

    This function censors the post-onset hours so the model is only trained
    and evaluated on the pre-onset trajectory.

    For each septic patient:

    * ``first_positive_iculos = min(iculos | y_eval == 1)`` is the first hour
      the label flips on — equal to ``t_onset - EARLY_LABEL_EXTRA_HOURS``.
    * ``t_onset = first_positive_iculos + EARLY_LABEL_EXTRA_HOURS``.
    * Rows where ``iculos < t_onset`` are kept (this includes the entire
      6-hour early-warning window where the label is already 1).
    * Rows where ``iculos >= t_onset`` are dropped.

    Parameters
    ----------
    X
        Feature matrix, one row per (patient, hour). Index is preserved.
    y_train
        Training labels (typically ``early_label``), aligned row-wise with X.
    y_eval
        Evaluation labels (``SepsisLabel``), aligned row-wise with X. Used to
        locate ``t_onset`` for each septic patient.
    patient_ids
        Patient ID per row.
    iculos
        Hours-in-ICU per row.

    Returns
    -------
    X, y_train, y_eval, patient_ids, iculos
        Filtered copies with post-onset rows removed.
    info : dict
        Diagnostics: ``n_rows_censored``, ``n_patients_with_censored_rows``.
    """
    if len(X) != len(y_train) or len(X) != len(y_eval):
        raise ValueError("X, y_train, y_eval must have the same length")
    if len(X) != len(patient_ids) or len(X) != len(iculos):
        raise ValueError("patient_ids, iculos must align with X")

    work = pd.DataFrame({
        "_pid": patient_ids,
        "_iculos": iculos,
        "_y_eval": y_eval,
    })

    positives = work[work["_y_eval"] == 1]
    first_positive_by_pid = positives.groupby("_pid")["_iculos"].min()
    onset_by_pid = first_positive_by_pid + EARLY_LABEL_EXTRA_HOURS

    onset_per_row = work["_pid"].map(onset_by_pid)
    is_septic = onset_per_row.notna()
    keep_mask = (~is_septic) | (work["_iculos"] < onset_per_row)
    keep_mask_np = keep_mask.to_numpy()

    n_rows_censored = int((~keep_mask_np).sum())
    affected_pids = work.loc[~keep_mask_np, "_pid"].unique()
    n_patients_with_censored_rows = int(len(affected_pids))

    info = {
        "n_rows_censored": n_rows_censored,
        "n_patients_with_censored_rows": n_patients_with_censored_rows,
        "n_rows_remaining": int(keep_mask_np.sum()),
    }

    return (
        X.loc[keep_mask_np].copy(),
        np.asarray(y_train)[keep_mask_np],
        np.asarray(y_eval)[keep_mask_np],
        np.asarray(patient_ids)[keep_mask_np],
        np.asarray(iculos)[keep_mask_np],
        info,
    )
