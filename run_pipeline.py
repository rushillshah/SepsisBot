"""Run the entire sepsis prediction pipeline end-to-end.

Steps:
    1. Load raw PSV data from both hospitals and save as parquet.
    2. Impute missing values (missingness flags, time-since-measured, ffill).
    3. Patient-level cross-validation (primary evaluation).
    4. Save metrics JSON, plots, and model artifacts.
"""

import json
import sys

from sklearn.metrics import roc_curve

from src.config import DATA_PROCESSED
from src.data_loader import load_all_data, load_processed, save_processed
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
        # REAL patient-level confusion matrix (summed across folds)
        "confusion_matrix": cv_results.get("total_patient_cm", {}),
        # Patient-level metrics (averaged across folds)
        "patient_sensitivity": cv_results.get("avg_patient_metrics", {}).get("sensitivity", {}).get("mean", 0),
        "patient_specificity": cv_results.get("avg_patient_metrics", {}).get("specificity", {}).get("mean", 0),
        "patient_precision": cv_results.get("avg_patient_metrics", {}).get("precision", {}).get("mean", 0),
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
    print("\n[Step 1/4] Loading raw data ...")
    try:
        raw_df = load_processed("raw_data")
        print(f"  Loaded cached raw data: {len(raw_df):,} rows")
    except FileNotFoundError:
        raw_df = load_all_data()
        save_processed(raw_df, "raw_data")

    # ── 2. Impute missing values ────────────────────────────────────────────
    print("\n[Step 2/4] Running imputation ...")
    try:
        imputed_df = load_processed("imputed_data")
        print(f"  Loaded cached imputed data: {len(imputed_df):,} rows")
    except FileNotFoundError:
        imputed_df = impute(raw_df)
        save_processed(imputed_df, "imputed_data")

    # ── 2b. Create early label for training ────────────────────────────────
    print("  Creating early_label (extended positive window for training) ...")
    from src.features import create_early_label
    imputed_df = create_early_label(imputed_df)
    n_early = int(imputed_df["early_label"].sum())
    n_orig = int(imputed_df["SepsisLabel"].sum())
    print(f"  SepsisLabel=1 rows: {n_orig:,}, early_label=1 rows: {n_early:,} ({n_early/n_orig:.1f}x)")

    # ── 3. Patient-level cross-validation (ALL metrics come from here) ────
    print("\n[Step 3/4] Running patient-level cross-validation ...")
    cv_results = cross_validate_pipeline(imputed_df)

    last_fold = cv_results["fold_results"][-1]
    best_params = last_fold.get("xgb_best_params", {})

    # Build feature matrix for IV/SHAP (read-only, no training)
    X_full, y_full = build_feature_matrix(imputed_df)
    feature_names = list(X_full.columns)

    # ── 4. Save artifacts ──────────────────────────────────────────────────
    print("\n[Step 4/4] Saving evaluation artifacts ...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Dashboard JSON — all metrics from CV concatenated predictions
    dashboard_json = _build_dashboard_json(cv_results, None, None)
    dashboard_json["xgb_best_params"] = best_params
    dashboard_json["n_features"] = len(feature_names)
    dashboard_json["default_threshold"] = 0.10

    # Per-fold detail for dashboard
    fold_detail = []
    for i, fr in enumerate(cv_results["fold_results"], 1):
        fold_detail.append({
            "fold": i,
            "xgb_auroc": fr["xgb_metrics"]["auroc"],
            "xgb_sensitivity": fr["xgb_metrics"]["sensitivity"],
            "xgb_specificity": fr["xgb_metrics"]["specificity"],
            "xgb_precision": fr["xgb_metrics"]["precision"],
            "xgb_f1": fr["xgb_metrics"]["f1"],
            "xgb_pr_auc": fr["xgb_metrics"]["pr_auc"],
            "xgb_train_auroc": fr["lr_train_auroc"],
            "xgb_best_params": fr["xgb_best_params"],
            "xgb_threshold": fr["xgb_threshold"],
            "lr_auroc": fr["lr_metrics"]["auroc"],
            "lr_sensitivity": fr["lr_metrics"]["sensitivity"],
            "lr_specificity": fr["lr_metrics"]["specificity"],
            "lr_precision": fr["lr_metrics"]["precision"],
            "patient_sensitivity": fr["xgb_patient_cm"]["sensitivity"],
            "patient_specificity": fr["xgb_patient_cm"]["specificity"],
            "patient_precision": fr["xgb_patient_cm"]["precision"],
            "patient_tp": fr["xgb_patient_cm"]["tp"],
            "patient_fp": fr["xgb_patient_cm"]["fp"],
            "patient_fn": fr["xgb_patient_cm"]["fn"],
            "patient_tn": fr["xgb_patient_cm"]["tn"],
        })
    dashboard_json["fold_detail"] = fold_detail

    # Save concatenated CV predictions to parquet
    concat = cv_results.get("concat_predictions", {})
    if "xgb_probs" in concat:
        import pandas as pd
        cv_preds_df = pd.DataFrame({
            "patient_id": concat["patient_ids"],
            "label": concat["labels"],
            "xgb_prob": concat["xgb_probs"],
            "lr_prob": concat["lr_probs"],
        })
        cv_preds_path = DATA_PROCESSED / "cv_predictions.parquet"
        cv_preds_df.to_parquet(cv_preds_path, index=False)
        print(f"  Saved CV predictions to {cv_preds_path} ({len(cv_preds_df):,} rows)")

    # Threshold analysis from honest CV concat (already computed in cross_validate_pipeline)
    dashboard_json["threshold_analysis"] = cv_results.get("threshold_analysis", [])

    # Patient-level metrics at default threshold (0.10)
    ta = cv_results.get("threshold_analysis", [])
    t010 = [r for r in ta if abs(r["threshold"] - 0.10) < 0.001]
    if t010:
        row = t010[0]
        dashboard_json["patient_sensitivity"] = row["patient_sensitivity"]
        dashboard_json["patient_specificity"] = row["patient_specificity"]
        dashboard_json["patient_precision"] = row["patient_precision"]
        dashboard_json["confusion_matrix"] = {
            "tp": row["tp"], "fn": row["fn"], "fp": row["fp"], "tn": row["tn"],
            "total_patients": row["total_patients"],
            "actual_sepsis": row["tp"] + row["fn"],
        }

    # ROC curve from concatenated CV predictions
    concat = cv_results.get("concat_predictions", {})
    if "xgb_probs" in concat:
        all_y = concat["labels"]
        all_xgb_p = concat["xgb_probs"]
        all_lr_p = concat["lr_probs"]
        fpr_xgb, tpr_xgb, _ = roc_curve(all_y, all_xgb_p)
        fpr_lr, tpr_lr, _ = roc_curve(all_y, all_lr_p)
        dashboard_json["fpr"] = fpr_xgb.tolist()
        dashboard_json["tpr"] = tpr_xgb.tolist()
        dashboard_json["lr_fpr"] = fpr_lr.tolist()
        dashboard_json["lr_tpr"] = tpr_lr.tolist()

    # Feature importance from existing feature_analysis files (no recompute)
    import os
    fa_dir = DATA_PROCESSED / "feature_analysis"
    if (fa_dir / "shap_ranking.csv").exists():
        import pandas as pd
        shap_df = pd.read_csv(fa_dir / "shap_ranking.csv")
        gain_path = fa_dir / "gain_ranking.csv"
        gain_df = pd.read_csv(gain_path) if gain_path.exists() else None
        fi = {}
        if gain_df is not None:
            for _, r in gain_df.head(30).iterrows():
                fi[r["feature"]] = float(r["gain_pct"])
        else:
            for _, r in shap_df.head(30).iterrows():
                fi[r["feature"]] = float(r["mean_abs_shap"])
        dashboard_json["feature_importance"] = fi
        dashboard_json["feature_ranking"] = shap_df.head(30).to_dict(orient="records")

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
