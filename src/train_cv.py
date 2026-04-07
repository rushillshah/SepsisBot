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
from sklearn.metrics import roc_auc_score
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

    # ── Oversample sepsis rows in training set ────────────────────────────
    # Duplicate sepsis rows to reach ~1:3 ratio (sepsis:non-sepsis)
    sepsis_mask = y_train == 1
    n_sepsis = int(sepsis_mask.sum())
    n_nonsepsis = int((~sepsis_mask).sum())

    if n_sepsis > 0:
        target_sepsis = n_nonsepsis // 3  # aim for 1:3 ratio
        oversample_factor = max(1, target_sepsis // n_sepsis)

        if oversample_factor > 1:
            X_sepsis = X_train_raw[sepsis_mask]
            y_sepsis = y_train[sepsis_mask]
            # Repeat sepsis rows
            X_oversampled = pd.concat([X_train_raw] + [X_sepsis] * (oversample_factor - 1), ignore_index=True)
            y_oversampled = pd.concat([y_train] + [y_sepsis] * (oversample_factor - 1), ignore_index=True)
            print(f"    Oversampled sepsis: {n_sepsis} -> {n_sepsis * oversample_factor} rows ({oversample_factor}x), non-sepsis: {n_nonsepsis}")
            X_train_raw = X_oversampled
            y_train = y_oversampled

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

    # ── Platt scaling calibration ──────────────────────────────────────────
    calibrated_xgb = CalibratedClassifierCV(
        estimator=search.best_estimator_,
        method="sigmoid",
        cv=3,
    )
    calibrated_xgb.fit(X_train, y_train)

    xgb_prob = calibrated_xgb.predict_proba(X_val)[:, 1]
    xgb_threshold = find_optimal_threshold(y_val.to_numpy(), xgb_prob)
    xgb_metrics = compute_metrics(y_val.to_numpy(), xgb_prob, threshold=xgb_threshold)

    # ── Train-set evaluation (overfit checking) ────────────────────────────
    lr_train_prob = lr.predict_proba(X_train)[:, 1]
    lr_train_auroc = roc_auc_score(y_train, lr_train_prob)

    xgb_train_prob = calibrated_xgb.predict_proba(X_train)[:, 1]
    xgb_train_auroc = roc_auc_score(y_train, xgb_train_prob)

    # ── Patient-level metrics (real, not fabricated) ─────────────────────────
    val_pids = df_val["patient_id"].values
    val_labels = df_val[LABEL_COL].values

    def _patient_level_cm(pids, labels, probs, threshold):
        """Compute real patient-level confusion matrix."""
        import pandas as _pd
        work = _pd.DataFrame({"pid": pids, "label": labels, "prob": probs})
        patient_actual = work.groupby("pid")["label"].max()  # 1 if ever sepsis
        patient_predicted = work.groupby("pid")["prob"].max() >= threshold  # 1 if ever alerted
        tp = int(((patient_actual == 1) & (patient_predicted == True)).sum())
        fn = int(((patient_actual == 1) & (patient_predicted == False)).sum())
        fp = int(((patient_actual == 0) & (patient_predicted == True)).sum())
        tn = int(((patient_actual == 0) & (patient_predicted == False)).sum())
        n_total = len(patient_actual)
        n_sepsis = int((patient_actual == 1).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "total_patients": n_total, "actual_sepsis": n_sepsis,
            "sensitivity": sens, "specificity": spec, "precision": prec,
        }

    xgb_patient_cm = _patient_level_cm(val_pids, val_labels, xgb_prob, xgb_threshold)

    print(
        f"  Fold {fold_num} | "
        f"Hour: XGB AUROC={xgb_metrics['auroc']:.4f}  "
        f"Patient: sens={xgb_patient_cm['sensitivity']:.3f} spec={xgb_patient_cm['specificity']:.3f} prec={xgb_patient_cm['precision']:.3f}"
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
        "lr_train_auroc": float(lr_train_auroc),
        "xgb_train_auroc": float(xgb_train_auroc),
        "xgb_patient_cm": xgb_patient_cm,
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

    overfit_table = []
    for i, r in enumerate(fold_results, 1):
        overfit_table.append({
            "fold": i,
            "lr_train_auroc": r["lr_train_auroc"],
            "lr_val_auroc": r["lr_metrics"]["auroc"],
            "lr_gap": r["lr_train_auroc"] - r["lr_metrics"]["auroc"],
            "xgb_train_auroc": r["xgb_train_auroc"],
            "xgb_val_auroc": r["xgb_metrics"]["auroc"],
            "xgb_gap": r["xgb_train_auroc"] - r["xgb_metrics"]["auroc"],
        })

    # Average patient-level metrics across folds
    avg_patient_cm = {}
    for key in ["sensitivity", "specificity", "precision"]:
        vals = [r["xgb_patient_cm"][key] for r in fold_results]
        avg_patient_cm[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Sum TP/FN/FP/TN across folds for aggregate confusion matrix
    total_cm = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "total_patients": 0, "actual_sepsis": 0}
    for r in fold_results:
        for k in total_cm:
            total_cm[k] += r["xgb_patient_cm"][k]

    _print_summary(avg_xgb_metrics, avg_lr_metrics, overfit_table, avg_patient_cm)

    return {
        "avg_xgb_metrics": avg_xgb_metrics,
        "avg_lr_metrics": avg_lr_metrics,
        "fold_results": fold_results,
        "overfit_table": overfit_table,
        "avg_patient_metrics": avg_patient_cm,
        "total_patient_cm": total_cm,
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


def _print_summary(avg_xgb: dict, avg_lr: dict, overfit_table: list[dict], avg_patient: dict | None = None) -> None:
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

    print("\n  OVERFIT CHECK (Train vs Val AUROC)")
    print(f"  {'Fold':<6s} {'LR Train':>10s} {'LR Val':>10s} {'LR Gap':>10s} {'XGB Train':>10s} {'XGB Val':>10s} {'XGB Gap':>10s}")
    for row in overfit_table:
        print(f"  {row['fold']:<6d} {row['lr_train_auroc']:>10.4f} {row['lr_val_auroc']:>10.4f} {row['lr_gap']:>10.4f} {row['xgb_train_auroc']:>10.4f} {row['xgb_val_auroc']:>10.4f} {row['xgb_gap']:>10.4f}")

    if avg_patient is not None:
        print(f"\n  PATIENT-LEVEL METRICS (XGBoost, averaged across folds)")
        print(f"  Sensitivity: {avg_patient['sensitivity']['mean']:.3f} +/- {avg_patient['sensitivity']['std']:.3f}")
        print(f"  Specificity: {avg_patient['specificity']['mean']:.3f} +/- {avg_patient['specificity']['std']:.3f}")
        print(f"  Precision:   {avg_patient['precision']['mean']:.3f} +/- {avg_patient['precision']['std']:.3f}")
