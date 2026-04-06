"""Threshold analysis and patient-level alert aggregation.

Inference-only tools — no retraining needed. Operates on existing
model predictions to analyze precision-recall tradeoffs and
patient-level alerting behavior.
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import LABEL_COL, TIME_COL


def precision_recall_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute precision, recall, F1, and alert rate at multiple thresholds.

    All metrics are hour-level.
    """
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50]

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        alert_rate = (tp + fp) / len(y_true) if len(y_true) > 0 else 0.0

        rows.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "alert_rate": alert_rate,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return pd.DataFrame(rows)


def patient_level_at_thresholds(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute patient-level sensitivity, specificity, precision at multiple thresholds.

    A patient is "predicted sepsis" if max(prob across all their hours) >= threshold.
    A patient "actually has sepsis" if max(SepsisLabel) == 1.
    """
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50]

    work = df[["patient_id", LABEL_COL]].copy()
    work["prob"] = np.asarray(y_prob)

    patient_actual = work.groupby("patient_id")[LABEL_COL].max()
    patient_max_prob = work.groupby("patient_id")["prob"].max()

    rows = []
    for t in thresholds:
        predicted = patient_max_prob >= t
        tp = int(((patient_actual == 1) & predicted).sum())
        fn = int(((patient_actual == 1) & ~predicted).sum())
        fp = int(((patient_actual == 0) & predicted).sum())
        tn = int(((patient_actual == 0) & ~predicted).sum())

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

        rows.append({
            "threshold": t,
            "patient_sensitivity": sens,
            "patient_specificity": spec,
            "patient_precision": prec,
            "patient_f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "total_flagged": tp + fp,
            "total_patients": len(patient_actual),
        })

    return pd.DataFrame(rows)


def consecutive_hour_alerts(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.025,
    min_consecutive: int = 3,
) -> pd.DataFrame:
    """Alert only if X consecutive hours above threshold.

    Returns patient-level summary: did the patient trigger a sustained alert?
    """
    work = df[["patient_id", TIME_COL, LABEL_COL]].copy()
    work["prob"] = np.asarray(y_prob)
    work["alert"] = (work["prob"] >= threshold).astype(int)

    rows = []
    for pid, group in work.groupby("patient_id"):
        group = group.sort_values(TIME_COL)
        actual_sepsis = group[LABEL_COL].max() == 1

        # Find consecutive alert runs
        alerts = group["alert"].values
        max_consecutive = 0
        current_run = 0
        first_sustained_hour = None

        for i, a in enumerate(alerts):
            if a == 1:
                current_run += 1
                if current_run >= min_consecutive and first_sustained_hour is None:
                    first_sustained_hour = float(group[TIME_COL].iloc[i - min_consecutive + 1])
                max_consecutive = max(max_consecutive, current_run)
            else:
                current_run = 0

        sustained_alert = max_consecutive >= min_consecutive

        onset_hour = None
        if actual_sepsis:
            onset_hour = float(group.loc[group[LABEL_COL] == 1, TIME_COL].iloc[0])

        early_warning = None
        if onset_hour is not None and first_sustained_hour is not None:
            early_warning = onset_hour - first_sustained_hour

        rows.append({
            "patient_id": pid,
            "actual_sepsis": actual_sepsis,
            "sustained_alert": sustained_alert,
            "max_consecutive_hours": max_consecutive,
            "first_sustained_hour": first_sustained_hour,
            "onset_hour": onset_hour,
            "early_warning_hours": early_warning,
        })

    return pd.DataFrame(rows)


def plot_threshold_tradeoff(
    patient_df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Plot patient-level sensitivity vs specificity vs precision across thresholds."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(patient_df["threshold"], patient_df["patient_sensitivity"],
            "o-", color="#d62728", linewidth=2, label="Sensitivity (catch rate)")
    ax.plot(patient_df["threshold"], patient_df["patient_specificity"],
            "s-", color="#1f77b4", linewidth=2, label="Specificity (1 - false alarm rate)")
    ax.plot(patient_df["threshold"], patient_df["patient_precision"],
            "^-", color="#2ca02c", linewidth=2, label="Precision (alert accuracy)")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Patient-Level Metrics vs Threshold")
    ax.legend()
    ax.set_xlim(0, max(patient_df["threshold"]) * 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
