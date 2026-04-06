"""Run the entire sepsis prediction pipeline end-to-end.

Steps:
    1. Load raw PSV data from both hospitals and save as parquet.
    2. Impute missing values (missingness flags, time-since-measured, ffill).
    3. Patient-level cross-validation (primary evaluation).
    4. Split imputed data by hospital.
    5. Train logistic regression baseline and tuned XGBoost on Hospital A.
    6. Evaluate both models on Hospital B (validation set).
    7. Feature importance analysis (IV, SHAP, gain ranking).
    8. Save metrics JSON, ROC curve, and feature importance plot for the
       Streamlit dashboard.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from src.config import DATA_PROCESSED
from src.data_loader import load_all_data, load_processed, save_processed
from src.evaluate import (
    evaluate_model,
    plot_feature_importance,
    plot_roc_curves,
)
from src.feature_importance import (
    compute_information_value,
    compute_gain_importance,
    compute_shap_values,
    combined_feature_ranking,
)
from src.features import build_feature_matrix, get_feature_names
from src.imputation import impute
from src.train import split_by_hospital, train_pipeline
from src.train_cv import cross_validate_pipeline
from src.train_lstm import predict_lstm, train_lstm_pipeline


def _build_dashboard_json(
    xgb_results: dict,
    logistic_results: dict,
    xgb_model,
    best_params: dict,
    feature_names: list[str],
    y_val: np.ndarray,
    cv_results: dict,
) -> dict:
    """Assemble the JSON payload consumed by the Streamlit dashboard.

    Includes both models' metrics, ROC curve arrays, top-30 feature importances,
    model parameters, and patient-level analysis.
    """
    xgb_metrics = xgb_results["metrics"]
    lr_metrics = logistic_results["metrics"]
    y_prob = xgb_results["y_prob"]
    y_prob_lr = logistic_results["y_prob"]

    # ROC curve data for both models
    fpr_xgb, tpr_xgb, _ = roc_curve(np.asarray(y_val), np.asarray(y_prob))
    fpr_lr, tpr_lr, _ = roc_curve(np.asarray(y_val), np.asarray(y_prob_lr))

    # Precision-recall curve data for XGBoost
    from sklearn.metrics import precision_recall_curve as pr_curve
    pr_precision, pr_recall, _ = pr_curve(np.asarray(y_val), np.asarray(y_prob))

    # Top 30 feature importances
    importances = np.asarray(xgb_model.feature_importances_)
    top_indices = np.argsort(importances)[::-1][:30]
    feature_importance = {
        feature_names[i]: float(importances[i]) for i in top_indices
    }

    # Gini coefficient = 2 * AUROC - 1
    gini_xgb = 2 * xgb_metrics["auroc"] - 1
    gini_lr = 2 * lr_metrics["auroc"] - 1

    # Patient analysis with NaN -> None for JSON serialization
    patient_analysis = []
    if "patient_analysis" in xgb_results:
        patient_df = xgb_results["patient_analysis"]
        for record in patient_df.to_dict(orient="records"):
            cleaned = {
                k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in record.items()
            }
            patient_analysis.append(cleaned)

    return {
        # XGBoost metrics
        "auroc": xgb_metrics["auroc"],
        "pr_auc": xgb_metrics["pr_auc"],
        "sensitivity": xgb_metrics["sensitivity"],
        "specificity": xgb_metrics["specificity"],
        "f1": xgb_metrics["f1"],
        "precision": xgb_metrics["precision"],
        "recall": xgb_metrics["sensitivity"],  # recall == sensitivity
        "gini": gini_xgb,
        "optimal_threshold": xgb_results["optimal_threshold"],
        # Logistic regression metrics
        "lr_auroc": lr_metrics["auroc"],
        "lr_pr_auc": lr_metrics["pr_auc"],
        "lr_sensitivity": lr_metrics["sensitivity"],
        "lr_specificity": lr_metrics["specificity"],
        "lr_f1": lr_metrics["f1"],
        "lr_precision": lr_metrics["precision"],
        "lr_recall": lr_metrics["sensitivity"],
        "lr_gini": gini_lr,
        "lr_optimal_threshold": logistic_results["optimal_threshold"],
        # ROC curves
        "fpr": fpr_xgb.tolist(),
        "tpr": tpr_xgb.tolist(),
        "lr_fpr": fpr_lr.tolist(),
        "lr_tpr": tpr_lr.tolist(),
        # Precision-recall curve (XGBoost)
        "pr_precision": pr_precision.tolist(),
        "pr_recall": pr_recall.tolist(),
        # Model parameters
        "xgb_best_params": best_params,
        "lr_params": {
            "penalty": "l2",
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
        },
        # Feature info
        "n_features": len(feature_names),
        "feature_importance": feature_importance,
        # Patient analysis
        "patient_analysis": patient_analysis,
        # Cross-validation metrics (XGBoost)
        "cv_xgb_auroc": cv_results["avg_xgb_metrics"]["auroc"]["mean"],
        "cv_xgb_auroc_std": cv_results["avg_xgb_metrics"]["auroc"]["std"],
        "cv_xgb_sensitivity": cv_results["avg_xgb_metrics"]["sensitivity"]["mean"],
        "cv_xgb_specificity": cv_results["avg_xgb_metrics"]["specificity"]["mean"],
        "cv_xgb_precision": cv_results["avg_xgb_metrics"]["precision"]["mean"],
        "cv_xgb_f1": cv_results["avg_xgb_metrics"]["f1"]["mean"],
        "cv_xgb_pr_auc": cv_results["avg_xgb_metrics"]["pr_auc"]["mean"],
        "cv_xgb_gini": 2 * cv_results["avg_xgb_metrics"]["auroc"]["mean"] - 1,
        # Cross-validation metrics (Logistic Regression)
        "cv_lr_auroc": cv_results["avg_lr_metrics"]["auroc"]["mean"],
        "cv_lr_sensitivity": cv_results["avg_lr_metrics"]["sensitivity"]["mean"],
        "cv_lr_specificity": cv_results["avg_lr_metrics"]["specificity"]["mean"],
        "cv_lr_precision": cv_results["avg_lr_metrics"]["precision"]["mean"],
        "cv_lr_f1": cv_results["avg_lr_metrics"]["f1"]["mean"],
        "cv_lr_pr_auc": cv_results["avg_lr_metrics"]["pr_auc"]["mean"],
        "cv_lr_gini": 2 * cv_results["avg_lr_metrics"]["auroc"]["mean"] - 1,
    }


def _print_summary(logistic_results: dict, xgb_results: dict, lstm_results: dict | None = None) -> None:
    """Print a final pipeline summary to stdout."""
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 70)

    lr_m = logistic_results["metrics"]
    xgb_m = xgb_results["metrics"]

    header = f"  {'Metric':<20s} {'Logistic':>10s} {'XGBoost':>10s}"
    divider_len = 44
    if lstm_results is not None:
        lstm_m = lstm_results["metrics"]
        header += f" {'LSTM':>10s}"
        divider_len = 56

    print(f"\n{header}")
    print(f"  {'-' * divider_len}")
    for key in ("auroc", "pr_auc", "sensitivity", "specificity", "precision", "f1"):
        line = f"  {key:<20s} {lr_m.get(key, 0):>10.4f} {xgb_m.get(key, 0):>10.4f}"
        if lstm_results is not None:
            line += f" {lstm_m.get(key, 0):>10.4f}"
        print(line)

    # Gini
    lr_gini = 2 * lr_m["auroc"] - 1
    xgb_gini = 2 * xgb_m["auroc"] - 1
    line = f"  {'gini':<20s} {lr_gini:>10.4f} {xgb_gini:>10.4f}"
    if lstm_results is not None:
        lstm_gini = 2 * lstm_m["auroc"] - 1
        line += f" {lstm_gini:>10.4f}"
    print(line)

    print(f"\n  Optimal threshold (XGBoost): "
          f"{xgb_results['optimal_threshold']:.4f}")
    if lstm_results is not None:
        print(f"  Optimal threshold (LSTM):    "
              f"{lstm_results['optimal_threshold']:.4f}")

    if "patient_analysis" in xgb_results:
        pa = xgb_results["patient_analysis"]
        n_patients = len(pa)
        n_actual = int(pa["sepsis_actual"].sum())
        n_predicted = int(pa["sepsis_predicted"].sum())
        print(f"  Patients evaluated: {n_patients}")
        print(f"  Actual sepsis:      {n_actual}")
        print(f"  Predicted sepsis:   {n_predicted}")

    print("\n  Artifacts saved:")
    print(f"    {DATA_PROCESSED / 'model_metrics.json'}")
    print(f"    {DATA_PROCESSED / 'roc_curves.png'}")
    print(f"    {DATA_PROCESSED / 'feature_importance.png'}")
    print(f"    {DATA_PROCESSED / 'lstm_training.png'}")
    print("=" * 70)


def run() -> None:
    """Execute the full sepsis prediction pipeline."""

    # ── 1. Load raw data ────────────────────────────────────────────────────
    print("\n[Step 1/8] Loading raw data ...")
    try:
        raw_df = load_processed("raw_data")
        print(f"  Loaded cached raw data: {len(raw_df):,} rows")
    except FileNotFoundError:
        raw_df = load_all_data()
        save_processed(raw_df, "raw_data")

    # ── 2. Impute missing values ────────────────────────────────────────────
    print("\n[Step 2/8] Running imputation ...")
    imputed_df = impute(raw_df)
    save_processed(imputed_df, "imputed_data")

    # ── 3. Patient-level cross-validation (primary evaluation) ──────────────
    print("\n[Step 3/8] Running patient-level cross-validation ...")
    cv_results = cross_validate_pipeline(imputed_df)

    # ── 4. Keep Hospital B imputed data for patient-level analysis ──────────
    # train_pipeline will also split internally, but we need the full
    # Hospital B DataFrame (with patient_id, ICULOS, SepsisLabel) to pass
    # into evaluate_model for patient-level analysis.
    print("\n[Step 4/8] Splitting imputed data by hospital ...")
    _, df_b_imputed = split_by_hospital(imputed_df)
    print(f"  Hospital B (validation): {len(df_b_imputed):,} rows")

    # ── 5. Train models ────────────────────────────────────────────────────
    print("\n[Step 5/8] Training models ...")
    pipeline_output = train_pipeline(imputed_df)
    xgb_model = pipeline_output["xgboost"]
    logistic_model = pipeline_output["logistic"]
    X_val = pipeline_output["X_val"]
    y_val = pipeline_output["y_val"]

    # Save model objects as .pkl
    import joblib
    models_dir = DATA_PROCESSED / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, models_dir / "xgboost_model.pkl")
    joblib.dump(logistic_model, models_dir / "logistic_model.pkl")
    print(f"  Saved model .pkl files to {models_dir}")

    # ── 6. Evaluate models on Hospital B ──────────────────────────────────────
    print("\n[Step 6/8] Evaluating models on Hospital B ...")
    logistic_results = evaluate_model(
        logistic_model, X_val, y_val, "Logistic Regression",
    )
    xgb_results = evaluate_model(
        xgb_model, X_val, y_val, "XGBoost", df_with_ids=df_b_imputed,
    )

    # ── 7. Feature Importance Analysis ────────────────────────────────────
    print("\n[Step 7/8] Running feature importance analysis ...")

    # We need an uncalibrated XGBoost for SHAP (TreeExplainer needs raw XGBoost)
    # Use the XGBoost from the cross-hospital pipeline
    xgb_model_raw = pipeline_output["xgboost"]

    # Build feature matrix on full imputed data for IV
    X_full, y_full = build_feature_matrix(imputed_df)
    fi_feature_names = list(X_full.columns)

    # IV
    print("  Computing Information Value ...")
    iv_df = compute_information_value(X_full, y_full)

    # Gain %
    print("  Computing XGBoost Gain % ...")
    gain_df = compute_gain_importance(xgb_model_raw, fi_feature_names)

    # SHAP (sample 10K rows for speed)
    print("  Computing SHAP values (10K sample) ...")
    sample_idx = np.random.RandomState(42).choice(
        len(X_full), size=min(10_000, len(X_full)), replace=False,
    )
    X_sample = X_full.iloc[sample_idx]
    shap_df = compute_shap_values(
        xgb_model_raw, X_sample, fi_feature_names, save_dir=str(DATA_PROCESSED),
    )

    # Combined ranking
    print("  Building combined ranking ...")
    ranking_df = combined_feature_ranking(iv_df, gain_df, shap_df, top_n=30)
    print(ranking_df.head(15).to_string(index=False))

    # ── 8. Save evaluation artifacts ────────────────────────────────────────
    print("\n[Step 8/8] Saving evaluation artifacts ...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Feature names from the validation feature matrix
    feature_names = list(X_val.columns)

    # JSON for the Streamlit dashboard
    best_params = pipeline_output["best_params"]
    dashboard_json = _build_dashboard_json(
        xgb_results, logistic_results, xgb_model, best_params, feature_names, y_val,
        cv_results,
    )

    # Feature importance
    dashboard_json["feature_ranking"] = ranking_df.head(30).to_dict(orient="records")
    dashboard_json["iv_top20"] = iv_df.head(20).to_dict(orient="records")

    # Overfit table
    dashboard_json["cv_overfit_table"] = cv_results.get("overfit_table", [])

    metrics_path = DATA_PROCESSED / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(dashboard_json, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    # ROC curve plot
    roc_data = {
        "Logistic Regression": (y_val, logistic_results["y_prob"]),
        "XGBoost": (y_val, xgb_results["y_prob"]),
    }
    roc_path = str(DATA_PROCESSED / "roc_curves.png")
    plot_roc_curves(roc_data, save_path=roc_path)

    # Feature importance plot
    importance_path = str(DATA_PROCESSED / "feature_importance.png")
    plot_feature_importance(
        xgb_model, feature_names, top_n=30, save_path=importance_path,
    )

    # ── 7. Summary ──────────────────────────────────────────────────────────
    _print_summary(logistic_results, xgb_results)


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
