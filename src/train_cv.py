"""Patient-level stratified cross-validation for the sepsis prediction pipeline.

Features are computed ONCE by the caller and passed in pre-computed.
Fold splitting uses row indices on the pre-computed feature matrix.
XGBoost probabilities are calibrated via Platt scaling.
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
    DEFAULT_THRESHOLD,
    INNER_CV_FOLDS,
    LABEL_COL,
    RANDOM_STATE,
    XGBOOST_PARAM_GRID_V2,
)
from src.evaluate import compute_metrics, find_optimal_threshold
from src.features import scale_features


def patient_stratified_split(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    n_splits: int = CV_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate patient-level stratified K-fold splits.

    All hourly rows belonging to one patient land in exactly one fold.
    """
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE,
    )
    row_positions = np.arange(len(patient_ids))
    folds = []
    for train_pos, val_pos in splitter.split(row_positions, labels, groups=patient_ids):
        folds.append((train_pos, val_pos))
    return folds


def _patient_level_cm(pids, labels, probs, threshold):
    """Compute real patient-level confusion matrix."""
    work = pd.DataFrame({"pid": pids, "label": labels, "prob": probs})
    patient_actual = work.groupby("pid")["label"].max()
    patient_predicted = work.groupby("pid")["prob"].max() >= threshold
    tp = int(((patient_actual == 1) & patient_predicted).sum())
    fn = int(((patient_actual == 1) & ~patient_predicted).sum())
    fp = int(((patient_actual == 0) & patient_predicted).sum())
    tn = int(((patient_actual == 0) & ~patient_predicted).sum())
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


def _train_fold(
    X: pd.DataFrame,
    y_train_labels: np.ndarray,
    y_eval_labels: np.ndarray,
    patient_ids: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_num: int,
) -> dict:
    """Train and evaluate both models on a single CV fold.

    X and labels are pre-computed. This function only slices, scales,
    oversamples, trains, and evaluates — no feature engineering.
    """
    X_train_raw = X.iloc[train_idx]
    X_val_raw = X.iloc[val_idx]
    y_train = y_train_labels[train_idx]
    y_val = y_eval_labels[val_idx]
    val_pids = patient_ids[val_idx]
    val_labels_raw = y_eval_labels[val_idx]

    # ── Oversample sepsis rows (numpy index repetition, no DataFrame concat) ─
    sepsis_idx = np.where(y_train == 1)[0]
    n_sepsis = len(sepsis_idx)
    n_nonsepsis = len(y_train) - n_sepsis

    if n_sepsis > 0:
        target = n_nonsepsis // 3
        oversample_factor = max(1, target // n_sepsis)
        if oversample_factor > 1:
            repeat_idx = np.tile(sepsis_idx, oversample_factor - 1)
            all_idx = np.concatenate([np.arange(len(y_train)), repeat_idx])
            X_train_raw = X_train_raw.iloc[all_idx].reset_index(drop=True)
            y_train = y_train[all_idx]
            print(f"    Oversampled: {n_sepsis} -> {n_sepsis * oversample_factor} ({oversample_factor}x)")

    # ── Scale features ────────────────────────────────────────────────────────
    X_train, X_val, scaler = scale_features(X_train_raw, X_val_raw)

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000,
        solver="lbfgs", random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_val)[:, 1]
    lr_threshold = find_optimal_threshold(y_val, lr_prob)
    lr_metrics = compute_metrics(y_val, lr_prob, threshold=lr_threshold)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    scale_pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

    base_xgb = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )
    inner_cv = StratifiedKFold(
        n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE,
    )
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
    xgb_best_params = search.best_params_

    # ── Platt scaling calibration ─────────────────────────────────────────────
    calibrated_xgb = CalibratedClassifierCV(
        estimator=search.best_estimator_,
        method="sigmoid",
        cv=INNER_CV_FOLDS,
    )
    calibrated_xgb.fit(X_train, y_train)

    xgb_prob = calibrated_xgb.predict_proba(X_val)[:, 1]
    xgb_threshold = find_optimal_threshold(y_val, xgb_prob)
    xgb_metrics = compute_metrics(y_val, xgb_prob, threshold=xgb_threshold)

    # ── Overfit check ─────────────────────────────────────────────────────────
    lr_train_auroc = roc_auc_score(y_train, lr.predict_proba(X_train)[:, 1])
    xgb_train_auroc = roc_auc_score(y_train, calibrated_xgb.predict_proba(X_train)[:, 1])

    # ── Patient-level metrics ─────────────────────────────────────────────────
    xgb_patient_cm = _patient_level_cm(val_pids, val_labels_raw, xgb_prob, xgb_threshold)

    print(
        f"  Fold {fold_num} | "
        f"Hour: XGB AUROC={xgb_metrics['auroc']:.4f}  "
        f"Patient: sens={xgb_patient_cm['sensitivity']:.3f} "
        f"spec={xgb_patient_cm['specificity']:.3f} "
        f"prec={xgb_patient_cm['precision']:.3f}"
    )

    return {
        "lr_metrics": lr_metrics,
        "xgb_metrics": xgb_metrics,
        "xgb_best_params": xgb_best_params,
        "xgb_threshold": xgb_threshold,
        "lr_threshold": lr_threshold,
        "y_val": y_val,
        "xgb_prob": xgb_prob,
        "lr_prob": lr_prob,
        "lr_train_auroc": float(lr_train_auroc),
        "xgb_train_auroc": float(xgb_train_auroc),
        "xgb_patient_cm": xgb_patient_cm,
        "val_patient_ids": val_pids,
        "val_labels": val_labels_raw,
        "scaler": scaler,
    }


def cross_validate_pipeline(
    X: pd.DataFrame,
    y_train_labels: np.ndarray,
    patient_ids: np.ndarray,
    y_eval_labels: np.ndarray,
) -> dict:
    """Run patient-level stratified CV on pre-computed features.

    Parameters
    ----------
    X : pd.DataFrame
        Pre-computed feature matrix (from a single build_feature_matrix call).
    y_train_labels : np.ndarray
        Training labels (early_label if extended window).
    patient_ids : np.ndarray
        Patient ID per row, aligned with X.
    y_eval_labels : np.ndarray
        Original SepsisLabel for evaluation metrics.
    """
    print("=" * 60)
    print("SEPSIS PREDICTION — PATIENT-LEVEL STRATIFIED CV")
    print(f"  Folds: {CV_FOLDS} | n_iter: {CV_N_ITER} | random_state: {RANDOM_STATE}")
    print(f"  Features: {X.shape[1]} (pre-computed, no per-fold recomputation)")
    print("=" * 60)

    folds = patient_stratified_split(patient_ids, y_eval_labels, n_splits=CV_FOLDS)

    fold_results = []
    for i, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n[Fold {i}/{CV_FOLDS}]  train={len(train_idx):,} rows  val={len(val_idx):,} rows")
        result = _train_fold(
            X, y_train_labels, y_eval_labels, patient_ids,
            train_idx, val_idx, fold_num=i,
        )
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

    # Average patient-level metrics
    avg_patient_cm = {}
    for key in ["sensitivity", "specificity", "precision"]:
        vals = [r["xgb_patient_cm"][key] for r in fold_results]
        avg_patient_cm[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    total_cm = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "total_patients": 0, "actual_sepsis": 0}
    for r in fold_results:
        for k in total_cm:
            total_cm[k] += r["xgb_patient_cm"][k]

    # ── Concatenate all fold predictions (honest) ────────────────────────────
    all_patient_ids = np.concatenate([r["val_patient_ids"] for r in fold_results])
    all_labels = np.concatenate([r["val_labels"] for r in fold_results])
    all_xgb_probs = np.concatenate([r["xgb_prob"] for r in fold_results])
    all_lr_probs = np.concatenate([r["lr_prob"] for r in fold_results])

    # Patient-level threshold analysis on concat
    from src.threshold_analysis import patient_level_at_thresholds, plot_threshold_tradeoff
    concat_df = pd.DataFrame({"patient_id": all_patient_ids, LABEL_COL: all_labels})
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
    threshold_table = patient_level_at_thresholds(concat_df, all_xgb_probs, thresholds)

    print("\n  THRESHOLD ANALYSIS (from concatenated CV predictions — honest)")
    print(threshold_table[["threshold", "patient_sensitivity", "patient_specificity", "patient_precision", "total_flagged"]].to_string(index=False))

    from src.config import DATA_PROCESSED
    plot_threshold_tradeoff(threshold_table, save_path=str(DATA_PROCESSED / "threshold_tradeoff.png"))

    _print_summary(avg_xgb_metrics, avg_lr_metrics, overfit_table, avg_patient_cm)

    return {
        "avg_xgb_metrics": avg_xgb_metrics,
        "avg_lr_metrics": avg_lr_metrics,
        "fold_results": fold_results,
        "overfit_table": overfit_table,
        "avg_patient_metrics": avg_patient_cm,
        "total_patient_cm": total_cm,
        "threshold_analysis": threshold_table.to_dict(orient="records"),
        "concat_predictions": {
            "patient_ids": all_patient_ids,
            "labels": all_labels,
            "xgb_probs": all_xgb_probs,
            "lr_probs": all_lr_probs,
        },
    }


def _average_metrics(metrics_list: list[dict]) -> dict:
    all_keys = list(metrics_list[0].keys())
    averaged = {}
    for key in all_keys:
        values = np.array([m[key] for m in metrics_list], dtype=float)
        averaged[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return averaged


def _print_summary(avg_xgb, avg_lr, overfit_table, avg_patient=None):
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
        print(f"  {key:<20s} {lr_mean:>10.4f} {lr_std:>8.4f} {xgb_mean:>10.4f} {xgb_std:>8.4f}")

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
