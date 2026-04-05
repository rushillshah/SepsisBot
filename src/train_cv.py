"""Patient-level stratified cross-validation for the sepsis prediction pipeline.

Trains both a Logistic Regression baseline and a calibrated XGBoost model
across CV_FOLDS outer folds, where all rows belonging to a single patient are
kept in the same fold.  Feature scaling is fit on the training portion of each
fold and applied to validation.  XGBoost probabilities are calibrated via
Platt scaling after inner hyperparameter search.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    CV_N_ITER,
    LABEL_COL,
    RANDOM_STATE,
    XGBOOST_PARAM_GRID_V2,
)
from src.evaluate import compute_metrics, find_optimal_threshold
from src.features import build_feature_matrix, scale_features


# ── Public API ────────────────────────────────────────────────────────────────


def patient_stratified_split(
    df: pd.DataFrame,
    n_splits: int = CV_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate patient-level stratified K-fold splits.

    All hourly rows belonging to one patient land in exactly one fold.
    The split is stratified at the row level so that each fold contains
    a representative proportion of positive (sepsis) labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``patient_id`` and ``SepsisLabel`` columns.
    n_splits : int, optional
        Number of folds. Defaults to ``CV_FOLDS``.

    Returns
    -------
    list of (train_indices, val_indices)
        Each element contains two 1-D integer arrays of row positions
        (not index labels) into *df*.

    Raises
    ------
    KeyError
        If ``patient_id`` or ``SepsisLabel`` is missing from *df*.
    ValueError
        If there are fewer unique patients than *n_splits*.
    """
    missing = {"patient_id", LABEL_COL} - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    n_patients = df["patient_id"].nunique()
    if n_patients < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} folds with only {n_patients} unique patients."
        )

    groups = df["patient_id"].to_numpy()
    y = df[LABEL_COL].to_numpy()
    # Row positions — StratifiedGroupKFold returns positional indices.
    row_positions = np.arange(len(df))

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_pos, val_pos in splitter.split(row_positions, y, groups=groups):
        folds.append((train_pos, val_pos))

    return folds


def _train_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_num: int,
) -> dict:
    """Train and evaluate both models on a single CV fold.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame (both hospitals combined) before feature engineering.
    train_idx : np.ndarray
        Integer row positions for the training portion.
    val_idx : np.ndarray
        Integer row positions for the validation portion.
    fold_num : int
        1-based fold index used only for console output.

    Returns
    -------
    dict
        Keys: ``lr_metrics``, ``xgb_metrics``, ``xgb_best_params``,
        ``xgb_threshold``, ``lr_threshold``, ``y_val``,
        ``xgb_prob``, ``lr_prob``.
    """
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    X_train_raw, y_train = build_feature_matrix(df_train)
    X_val_raw, y_val = build_feature_matrix(df_val)

    X_train, X_val, _ = scale_features(X_train_raw, X_val_raw)

    # ── Logistic Regression ────────────────────────────────────────────────
    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_val)[:, 1]
    lr_threshold = find_optimal_threshold(y_val.to_numpy(), lr_prob)
    lr_metrics = compute_metrics(y_val.to_numpy(), lr_prob, threshold=lr_threshold)

    # ── XGBoost with inner RandomizedSearchCV ─────────────────────────────
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    base_xgb = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=XGBOOST_PARAM_GRID_V2,
        n_iter=CV_N_ITER,
        scoring="roc_auc",
        cv=inner_cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    xgb_best_params: dict = search.best_params_

    # ── Platt scaling calibration on the already-fitted best estimator ────
    calibrated_xgb = CalibratedClassifierCV(
        estimator=search.best_estimator_,
        method="sigmoid",
        cv="prefit",
    )
    calibrated_xgb.fit(X_train, y_train)

    xgb_prob = calibrated_xgb.predict_proba(X_val)[:, 1]
    xgb_threshold = find_optimal_threshold(y_val.to_numpy(), xgb_prob)
    xgb_metrics = compute_metrics(y_val.to_numpy(), xgb_prob, threshold=xgb_threshold)

    print(
        f"  Fold {fold_num} | "
        f"LR AUROC={lr_metrics['auroc']:.4f}  "
        f"XGB AUROC={xgb_metrics['auroc']:.4f}  "
        f"(inner best={search.best_score_:.4f})"
    )

    return {
        "lr_metrics": lr_metrics,
        "xgb_metrics": xgb_metrics,
        "xgb_best_params": xgb_best_params,
        "xgb_threshold": xgb_threshold,
        "lr_threshold": lr_threshold,
        "y_val": y_val.to_numpy(),
        "xgb_prob": xgb_prob,
        "lr_prob": lr_prob,
    }


def cross_validate_pipeline(df: pd.DataFrame) -> dict:
    """Run the full patient-level stratified CV pipeline.

    Splits *df* into ``CV_FOLDS`` folds at the patient level, trains both
    models on each fold, and aggregates metrics across folds.

    Parameters
    ----------
    df : pd.DataFrame
        Combined (both hospitals) imputed DataFrame with ``patient_id``,
        ``hospital``, and ``SepsisLabel`` columns.

    Returns
    -------
    dict
        Keys: ``avg_xgb_metrics``, ``avg_lr_metrics``, ``fold_results``.
        Averaged metric dicts include both ``mean`` and ``std`` sub-keys
        for every metric.
    """
    print("=" * 60)
    print("SEPSIS PREDICTION — PATIENT-LEVEL STRATIFIED CV")
    print(f"  Folds: {CV_FOLDS} | n_iter: {CV_N_ITER} | random_state: {RANDOM_STATE}")
    print("=" * 60)

    folds = patient_stratified_split(df, n_splits=CV_FOLDS)

    fold_results: list[dict] = []
    for i, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n[Fold {i}/{CV_FOLDS}]  train={len(train_idx):,} rows  val={len(val_idx):,} rows")
        result = _train_fold(df, train_idx, val_idx, fold_num=i)
        fold_results.append(result)

    avg_xgb_metrics = _average_metrics([r["xgb_metrics"] for r in fold_results])
    avg_lr_metrics = _average_metrics([r["lr_metrics"] for r in fold_results])

    _print_summary(avg_xgb_metrics, avg_lr_metrics)

    return {
        "avg_xgb_metrics": avg_xgb_metrics,
        "avg_lr_metrics": avg_lr_metrics,
        "fold_results": fold_results,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _average_metrics(metrics_list: list[dict]) -> dict:
    """Average a list of metric dicts, reporting mean and std per metric.

    Parameters
    ----------
    metrics_list : list[dict]
        One dict per fold, all sharing the same keys.

    Returns
    -------
    dict
        Each key maps to a sub-dict ``{"mean": float, "std": float}``.
    """
    all_keys = list(metrics_list[0].keys())
    averaged: dict = {}
    for key in all_keys:
        values = np.array([m[key] for m in metrics_list], dtype=float)
        averaged[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return averaged


def _print_summary(avg_xgb: dict, avg_lr: dict) -> None:
    """Print the cross-validation summary table to stdout."""
    width = 60
    print("\n" + "=" * width)
    print("  CROSS-VALIDATION SUMMARY")
    print("=" * width)
    print(f"  {'Metric':<20s} {'LR Mean':>10s} {'LR Std':>8s} {'XGB Mean':>10s} {'XGB Std':>8s}")
    print("-" * width)

    for key in avg_lr:
        lr_mean = avg_lr[key]["mean"]
        lr_std = avg_lr[key]["std"]
        xgb_mean = avg_xgb[key]["mean"]
        xgb_std = avg_xgb[key]["std"]
        label = key.replace("_", " ").title()
        print(
            f"  {label:<20s} {lr_mean:>10.4f} {lr_std:>8.4f} {xgb_mean:>10.4f} {xgb_std:>8.4f}"
        )

    # Gini = 2 * AUROC - 1
    lr_gini = 2.0 * avg_lr["auroc"]["mean"] - 1.0
    xgb_gini = 2.0 * avg_xgb["auroc"]["mean"] - 1.0
    print("-" * width)
    print(f"  {'Gini Coefficient':<20s} {lr_gini:>10.4f} {'':>8s} {xgb_gini:>10.4f}")
    print("=" * width)
