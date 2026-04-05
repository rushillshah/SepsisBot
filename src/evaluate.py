"""Model evaluation for the sepsis prediction pipeline.

Provides classification metrics (AUROC, PR-AUC, sensitivity, specificity),
threshold optimization via Youden's J statistic, ROC curve visualization,
feature importance plotting, and patient-level prediction analysis.

All functions operate on immutable inputs — no DataFrames or arrays are
modified in place.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import LABEL_COL, TARGET_AUROC, TIME_COL


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute classification metrics at a given probability threshold.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    threshold : float, optional
        Decision threshold for converting probabilities to binary
        predictions.  Defaults to ``0.5``.

    Returns
    -------
    dict
        Keys: ``auroc``, ``pr_auc``, ``sensitivity``, ``specificity``,
        ``f1``, ``precision``.

    Raises
    ------
    ValueError
        If *y_true* and *y_prob* have different lengths, or if
        *y_true* contains fewer than two distinct classes.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_prob has {len(y_prob)}."
        )
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "y_true must contain at least two distinct classes "
            "to compute AUROC."
        )

    y_pred = (y_prob >= threshold).astype(int)

    auroc = roc_auc_score(y_true, y_prob)

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_score = auc(recall_vals, precision_vals)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auroc": float(auroc),
        "pr_auc": float(pr_auc_score),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    }


# ── Threshold Optimization ───────────────────────────────────────────────────


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Find the threshold that maximizes Youden's J statistic.

    Youden's J = sensitivity + specificity - 1.  This balances true
    positive rate and true negative rate, producing a threshold that
    is generally more clinically useful than the default 0.5 for
    imbalanced sepsis data.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        The probability threshold that maximizes Youden's J.

    Raises
    ------
    ValueError
        If *y_true* contains fewer than two distinct classes.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "y_true must contain at least two distinct classes "
            "to compute a ROC curve."
        )

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr + (1 - fpr) - 1  # sensitivity + specificity - 1
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


# ── Plotting ─────────────────────────────────────────────────────────────────


_PLOT_STYLE = "seaborn-v0_8-whitegrid"


def _apply_plot_style() -> None:
    """Set the shared matplotlib style for all evaluation plots."""
    plt.style.use(_PLOT_STYLE)


def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    """Save figure to *save_path* if provided, otherwise display it."""
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_roc_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: str | None = None,
) -> None:
    """Plot ROC curves for one or more models.

    Parameters
    ----------
    results : dict
        Mapping of model name to ``(y_true, y_prob)`` tuples.
    save_path : str or None, optional
        File path to save the figure.  If ``None``, the plot is
        displayed interactively.

    Raises
    ------
    ValueError
        If *results* is empty.
    """
    if not results:
        raise ValueError("results dict must contain at least one model.")

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, (y_true, y_prob) in results.items():
        fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_prob))
        auroc = roc_auc_score(np.asarray(y_true), np.asarray(y_prob))
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUROC = {auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUROC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    _save_or_show(fig, save_path)


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
    save_path: str | None = None,
) -> None:
    """Plot the top N feature importances from an XGBoost model.

    Parameters
    ----------
    model
        A fitted model with a ``feature_importances_`` attribute
        (e.g. an XGBoost classifier).
    feature_names : list[str]
        Feature names corresponding to the model's input columns.
    top_n : int, optional
        Number of top features to display.  Defaults to ``20``.
    save_path : str or None, optional
        File path to save the figure.  If ``None``, the plot is
        displayed interactively.

    Raises
    ------
    AttributeError
        If *model* does not have ``feature_importances_``.
    ValueError
        If *feature_names* length does not match the importance array.
    """
    importances = np.asarray(model.feature_importances_)

    if len(feature_names) != len(importances):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"feature_importances_ length ({len(importances)})."
        )

    sorted_indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in sorted_indices]
    top_values = importances[sorted_indices]

    # Reverse so the most important feature is at the top of the bar chart.
    top_names = top_names[::-1]
    top_values = top_values[::-1]

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    ax.barh(range(len(top_names)), top_values, align="center")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {len(top_names)} Feature Importances")

    _save_or_show(fig, save_path)


# ── Patient-Level Analysis ───────────────────────────────────────────────────


def patient_level_analysis(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Summarize predictions at the patient level.

    For each patient, determines whether sepsis was actually present,
    whether the model predicted it, and computes early-warning timing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``patient_id``, ``ICULOS``, and ``SepsisLabel``
        columns.  Must have the same number of rows as *y_prob*.
    y_prob : array-like
        Predicted probabilities for the positive class, aligned with
        the rows of *df*.
    threshold : float
        Decision threshold for converting probabilities to alerts.

    Returns
    -------
    pd.DataFrame
        One row per patient with columns: ``patient_id``,
        ``sepsis_actual``, ``sepsis_predicted``, ``first_alert_hour``,
        ``sepsis_onset_hour``, ``early_warning_hours``.

    Raises
    ------
    KeyError
        If *df* is missing required columns.
    ValueError
        If *df* row count does not match *y_prob* length.
    """
    required_cols = {"patient_id", TIME_COL, LABEL_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    y_prob = np.asarray(y_prob)
    if len(df) != len(y_prob):
        raise ValueError(
            f"DataFrame has {len(df)} rows but y_prob has "
            f"{len(y_prob)} elements."
        )

    # Work on a copy to avoid mutating the caller's DataFrame.
    work = df[["patient_id", TIME_COL, LABEL_COL]].copy()
    work["y_prob"] = y_prob
    work["predicted_positive"] = (y_prob >= threshold).astype(int)

    rows: list[dict] = []
    for patient_id, group in work.groupby("patient_id"):
        sepsis_actual = int(group[LABEL_COL].max()) == 1

        alerts = group.loc[group["predicted_positive"] == 1]
        sepsis_predicted = len(alerts) > 0
        first_alert_hour = (
            float(alerts[TIME_COL].iloc[0]) if sepsis_predicted else np.nan
        )

        onset_rows = group.loc[group[LABEL_COL] == 1]
        sepsis_onset_hour = (
            float(onset_rows[TIME_COL].iloc[0]) if len(onset_rows) > 0 else np.nan
        )

        early_warning_hours = sepsis_onset_hour - first_alert_hour

        rows.append({
            "patient_id": patient_id,
            "sepsis_actual": sepsis_actual,
            "sepsis_predicted": sepsis_predicted,
            "first_alert_hour": first_alert_hour,
            "sepsis_onset_hour": sepsis_onset_hour,
            "early_warning_hours": early_warning_hours,
        })

    return pd.DataFrame(rows)


# ── Reporting ────────────────────────────────────────────────────────────────


def print_evaluation_report(metrics: dict, model_name: str) -> None:
    """Pretty-print a metrics dictionary to stdout.

    Parameters
    ----------
    metrics : dict
        Output of :func:`compute_metrics`.
    model_name : str
        Human-readable model name for the header.
    """
    width = 40
    print("=" * width)
    print(f"  {model_name} — Evaluation Report")
    print("=" * width)

    for key, value in metrics.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<20s}: {value:.4f}")

    target_met = metrics.get("auroc", 0) >= TARGET_AUROC
    status = "PASS" if target_met else "BELOW TARGET"
    print("-" * width)
    print(f"  Target AUROC ({TARGET_AUROC:.2f}): {status}")
    print("=" * width)


# ── Full Evaluation Pipeline ─────────────────────────────────────────────────


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    df_with_ids: pd.DataFrame | None = None,
) -> dict:
    """Run the complete evaluation pipeline for a trained model.

    Generates probability predictions, computes metrics, finds the
    optimal threshold, prints a report, and optionally performs
    patient-level analysis.

    Parameters
    ----------
    model
        A fitted model with a ``predict_proba`` method.
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Ground-truth binary labels.
    model_name : str
        Human-readable model name for reporting.
    df_with_ids : pd.DataFrame or None, optional
        If provided, must contain ``patient_id``, ``ICULOS``, and
        ``SepsisLabel`` columns aligned with *X* and *y*.
        Enables patient-level analysis.

    Returns
    -------
    dict
        Keys: ``metrics`` (dict), ``optimal_threshold`` (float),
        ``y_prob`` (ndarray).  If *df_with_ids* is provided, also
        includes ``patient_analysis`` (DataFrame).

    Raises
    ------
    AttributeError
        If *model* does not have a ``predict_proba`` method.
    """
    y = np.asarray(y)
    y_prob = model.predict_proba(X)[:, 1]

    optimal_threshold = find_optimal_threshold(y, y_prob)
    metrics = compute_metrics(y, y_prob, threshold=optimal_threshold)

    print_evaluation_report(metrics, model_name)
    print(f"  Optimal threshold (Youden's J): {optimal_threshold:.4f}")

    result = {
        "metrics": metrics,
        "optimal_threshold": optimal_threshold,
        "y_prob": y_prob,
    }

    if df_with_ids is not None:
        patient_df = patient_level_analysis(df_with_ids, y_prob, optimal_threshold)
        result["patient_analysis"] = patient_df

        n_patients = len(patient_df)
        n_actual = int(patient_df["sepsis_actual"].sum())
        n_predicted = int(patient_df["sepsis_predicted"].sum())
        early = patient_df["early_warning_hours"].dropna()
        median_early = float(early.median()) if len(early) > 0 else 0.0

        print(f"\n  Patient-level summary ({n_patients} patients):")
        print(f"    Actual sepsis:    {n_actual}")
        print(f"    Predicted sepsis: {n_predicted}")
        print(f"    Median early warning: {median_early:.1f} hours")

    return result
