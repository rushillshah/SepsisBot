"""Run the entire sepsis prediction pipeline end-to-end.

Steps:
    1. Load raw PSV data from both hospitals and save as parquet.
    2. Impute missing values (missingness flags, time-since-measured, ffill).
    3. Patient-level cross-validation (primary evaluation).
    4. Feature importance analysis (IV, SHAP, gain ranking).
    5. Save metrics JSON, plots, and model .pkl files.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_curve

from src.config import DATA_PROCESSED
from src.data_loader import load_all_data, load_processed, save_processed
from src.evaluate import plot_roc_curves
from src.feature_importance import (
    combined_feature_ranking,
    compute_gain_importance,
    compute_information_value,
    compute_shap_values,
)
from src.features import build_feature_matrix
from src.imputation import impute
from src.train_cv import cross_validate_pipeline


def _build_dashboard_json(cv_results: dict, ranking_df, iv_df) -> dict:
    """Assemble the JSON payload consumed by the Streamlit dashboard."""
    avg_xgb = cv_results["avg_xgb_metrics"]
    avg_lr = cv_results["avg_lr_metrics"]

    return {
        # CV metrics (primary)
        "cv_xgb_auroc": avg_xgb["auroc"]["mean"],
        "cv_xgb_auroc_std": avg_xgb["auroc"]["std"],
        "cv_xgb_sensitivity": avg_xgb["sensitivity"]["mean"],
        "cv_xgb_specificity": avg_xgb["specificity"]["mean"],
        "cv_xgb_precision": avg_xgb["precision"]["mean"],
        "cv_xgb_f1": avg_xgb["f1"]["mean"],
        "cv_xgb_pr_auc": avg_xgb["pr_auc"]["mean"],
        "cv_xgb_gini": 2 * avg_xgb["auroc"]["mean"] - 1,
        "cv_lr_auroc": avg_lr["auroc"]["mean"],
        "cv_lr_auroc_std": avg_lr["auroc"]["std"],
        "cv_lr_sensitivity": avg_lr["sensitivity"]["mean"],
        "cv_lr_specificity": avg_lr["specificity"]["mean"],
        "cv_lr_precision": avg_lr["precision"]["mean"],
        "cv_lr_f1": avg_lr["f1"]["mean"],
        "cv_lr_pr_auc": avg_lr["pr_auc"]["mean"],
        "cv_lr_gini": 2 * avg_lr["auroc"]["mean"] - 1,
        # Overfit table
        "cv_overfit_table": cv_results.get("overfit_table", []),
        # Feature importance
        "feature_ranking": ranking_df.head(30).to_dict(orient="records") if ranking_df is not None else [],
        "iv_top20": iv_df.head(20).to_dict(orient="records") if iv_df is not None else [],
        # Per-fold ROC data (use last fold for plotting)
        "fpr": [],
        "tpr": [],
    }


def _print_summary(cv_results: dict) -> None:
    """Print a final pipeline summary to stdout."""
    avg_xgb = cv_results["avg_xgb_metrics"]
    avg_lr = cv_results["avg_lr_metrics"]

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — SUMMARY (Patient-Level CV)")
    print("=" * 60)

    print(f"\n  {'Metric':<20s} {'LR':>10s} {'XGBoost':>10s}")
    print(f"  {'-' * 42}")
    for key in ("auroc", "pr_auc", "sensitivity", "specificity", "precision", "f1"):
        lr_val = avg_lr[key]["mean"]
        xgb_val = avg_xgb[key]["mean"]
        print(f"  {key:<20s} {lr_val:>10.4f} {xgb_val:>10.4f}")

    lr_gini = 2 * avg_lr["auroc"]["mean"] - 1
    xgb_gini = 2 * avg_xgb["auroc"]["mean"] - 1
    print(f"  {'gini':<20s} {lr_gini:>10.4f} {xgb_gini:>10.4f}")

    print("\n  Artifacts saved:")
    print(f"    {DATA_PROCESSED / 'model_metrics.json'}")
    print(f"    {DATA_PROCESSED / 'models/'}")
    print("=" * 60)


def run() -> None:
    """Execute the full sepsis prediction pipeline."""

    # ── 1. Load raw data ────────────────────────────────────────────────────
    print("\n[Step 1/5] Loading raw data ...")
    try:
        raw_df = load_processed("raw_data")
        print(f"  Loaded cached raw data: {len(raw_df):,} rows")
    except FileNotFoundError:
        raw_df = load_all_data()
        save_processed(raw_df, "raw_data")

    # ── 2. Impute missing values ────────────────────────────────────────────
    print("\n[Step 2/5] Running imputation ...")
    imputed_df = impute(raw_df)
    save_processed(imputed_df, "imputed_data")

    # ── 3. Patient-level cross-validation ──────────────────────────────────
    print("\n[Step 3/5] Running patient-level cross-validation ...")
    cv_results = cross_validate_pipeline(imputed_df)

    # Save the best XGBoost model from last fold as .pkl
    last_fold = cv_results["fold_results"][-1]
    models_dir = DATA_PROCESSED / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # The CV pipeline doesn't return fitted models currently —
    # train a final model on ALL data with best params from CV
    print("\n  Training final model on full dataset ...")
    from src.features import scale_features
    X_full, y_full = build_feature_matrix(imputed_df)
    feature_names = list(X_full.columns)

    # Use best params from last fold
    best_params = last_fold.get("xgb_best_params", {})

    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier
    from src.config import RANDOM_STATE

    n_pos = int(np.sum(y_full == 1))
    n_neg = int(np.sum(y_full == 0))
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    # Train LR
    lr_model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000,
        random_state=RANDOM_STATE, solver="lbfgs",
    )
    lr_model.fit(X_full_scaled, y_full)

    # Train XGBoost with best params
    xgb_model = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        **best_params,
    )
    xgb_model.fit(X_full_scaled, y_full)

    # Calibrate
    calibrated_xgb = CalibratedClassifierCV(
        estimator=xgb_model, method="sigmoid", cv=3,
    )
    calibrated_xgb.fit(X_full_scaled, y_full)

    joblib.dump(calibrated_xgb, models_dir / "xgboost_model.pkl")
    joblib.dump(lr_model, models_dir / "logistic_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    print(f"  Saved model .pkl files to {models_dir}")

    # ── 4. Feature Importance Analysis ────────────────────────────────────
    print("\n[Step 4/5] Running feature importance analysis ...")

    # IV on full data
    print("  Computing Information Value ...")
    iv_df = compute_information_value(X_full, y_full)

    # Gain % (use raw xgb, not calibrated)
    print("  Computing XGBoost Gain % ...")
    gain_df = compute_gain_importance(xgb_model, feature_names)

    # SHAP (sample 10K)
    print("  Computing SHAP values (10K sample) ...")
    sample_idx = np.random.RandomState(42).choice(
        len(X_full), size=min(10_000, len(X_full)), replace=False,
    )
    X_sample = X_full.iloc[sample_idx]
    shap_df = compute_shap_values(
        xgb_model, X_sample, feature_names, save_dir=str(DATA_PROCESSED),
    )

    # Combined ranking
    print("  Building combined ranking ...")
    ranking_df = combined_feature_ranking(iv_df, gain_df, shap_df, top_n=30)
    print(ranking_df.head(15).to_string(index=False))

    # ── 5. Save artifacts ──────────────────────────────────────────────────
    print("\n[Step 5/5] Saving evaluation artifacts ...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    dashboard_json = _build_dashboard_json(cv_results, ranking_df, iv_df)
    dashboard_json["xgb_best_params"] = best_params
    dashboard_json["n_features"] = len(feature_names)

    # Add ROC data from last CV fold
    last_y_val = last_fold["y_val"]
    last_xgb_prob = last_fold["xgb_prob"]
    last_lr_prob = last_fold["lr_prob"]
    fpr_xgb, tpr_xgb, _ = roc_curve(last_y_val, last_xgb_prob)
    fpr_lr, tpr_lr, _ = roc_curve(last_y_val, last_lr_prob)
    dashboard_json["fpr"] = fpr_xgb.tolist()
    dashboard_json["tpr"] = tpr_xgb.tolist()
    dashboard_json["lr_fpr"] = fpr_lr.tolist()
    dashboard_json["lr_tpr"] = tpr_lr.tolist()

    metrics_path = DATA_PROCESSED / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(dashboard_json, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    _print_summary(cv_results)


def main() -> None:
    """Entry point with error handling for missing data files."""
    try:
        run()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] Data files not found: {exc}", file=sys.stderr)
        print(
            "Ensure raw PSV files are placed in data/raw/training_setA/ "
            "and data/raw/training_setB/.",
            file=sys.stderr,
        )
        sys.exit(1)
    except ValueError as exc:
        print(f"\n[ERROR] Data validation failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
