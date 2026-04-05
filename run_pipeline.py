"""Run the entire sepsis prediction pipeline end-to-end.

Steps:
    1. Load raw PSV data from both hospitals and save as parquet.
    2. Impute missing values (missingness flags, time-since-measured, ffill).
    3. Train logistic regression baseline and tuned XGBoost on Hospital A.
    4. Evaluate both models on Hospital B (validation set).
    5. Save metrics JSON, ROC curve, and feature importance plot for the
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
from src.features import get_feature_names
from src.imputation import impute
from src.train import split_by_hospital, train_pipeline
from src.train_lstm import predict_lstm, train_lstm_pipeline


def _build_dashboard_json(
    xgb_results: dict,
    logistic_results: dict,
    xgb_model,
    best_params: dict,
    feature_names: list[str],
    y_val: np.ndarray,
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
    print("\n[Step 1/7] Loading raw data ...")
    try:
        raw_df = load_processed("raw_data")
        print(f"  Loaded cached raw data: {len(raw_df):,} rows")
    except FileNotFoundError:
        raw_df = load_all_data()
        save_processed(raw_df, "raw_data")

    # ── 2. Impute missing values ────────────────────────────────────────────
    print("\n[Step 2/7] Running imputation ...")
    imputed_df = impute(raw_df)
    save_processed(imputed_df, "imputed_data")

    # ── 3. Keep Hospital B imputed data for patient-level analysis ──────────
    # train_pipeline will also split internally, but we need the full
    # Hospital B DataFrame (with patient_id, ICULOS, SepsisLabel) to pass
    # into evaluate_model for patient-level analysis.
    print("\n[Step 3/7] Splitting imputed data by hospital ...")
    _, df_b_imputed = split_by_hospital(imputed_df)
    print(f"  Hospital B (validation): {len(df_b_imputed):,} rows")

    # ── 4. Train models ────────────────────────────────────────────────────
    print("\n[Step 4/7] Training models ...")
    pipeline_output = train_pipeline(imputed_df)
    xgb_model = pipeline_output["xgboost"]
    logistic_model = pipeline_output["logistic"]
    X_val = pipeline_output["X_val"]
    y_val = pipeline_output["y_val"]

    # ── 5. Train LSTM ─────────────────────────────────────────────────────────
    print("\n[Step 5/9] Training LSTM sequence model ...")
    lstm_output = train_lstm_pipeline(imputed_df, seq_length=12)
    lstm_model = lstm_output["model"]
    X_val_lstm = lstm_output["X_val"]
    y_val_lstm = lstm_output["y_val"]

    # ── 6. Evaluate all models ─────────────────────────────────────────────
    print("\n[Step 6/9] Evaluating models on Hospital B ...")
    logistic_results = evaluate_model(
        logistic_model, X_val, y_val, "Logistic Regression",
    )
    xgb_results = evaluate_model(
        xgb_model, X_val, y_val, "XGBoost", df_with_ids=df_b_imputed,
    )

    # LSTM evaluation (manual since it's not a sklearn model)
    lstm_y_prob = predict_lstm(lstm_model, X_val_lstm)
    from src.evaluate import compute_metrics, find_optimal_threshold, print_evaluation_report
    lstm_threshold = find_optimal_threshold(y_val_lstm, lstm_y_prob)
    lstm_metrics = compute_metrics(y_val_lstm, lstm_y_prob, threshold=lstm_threshold)
    print_evaluation_report(lstm_metrics, "LSTM (seq=12)")
    print(f"  Optimal threshold (Youden's J): {lstm_threshold:.4f}")
    lstm_results = {
        "metrics": lstm_metrics,
        "optimal_threshold": lstm_threshold,
        "y_prob": lstm_y_prob,
    }

    # ── 7. Save evaluation artifacts ────────────────────────────────────────
    print("\n[Step 7/9] Saving evaluation artifacts ...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Feature names from the validation feature matrix
    feature_names = list(X_val.columns)

    # JSON for the Streamlit dashboard
    best_params = pipeline_output["best_params"]
    dashboard_json = _build_dashboard_json(
        xgb_results, logistic_results, xgb_model, best_params, feature_names, y_val,
    )

    # Add LSTM metrics and ROC curve to dashboard JSON
    lstm_gini = 2 * lstm_metrics["auroc"] - 1
    dashboard_json["lstm_auroc"] = lstm_metrics["auroc"]
    dashboard_json["lstm_pr_auc"] = lstm_metrics["pr_auc"]
    dashboard_json["lstm_sensitivity"] = lstm_metrics["sensitivity"]
    dashboard_json["lstm_specificity"] = lstm_metrics["specificity"]
    dashboard_json["lstm_f1"] = lstm_metrics["f1"]
    dashboard_json["lstm_precision"] = lstm_metrics["precision"]
    dashboard_json["lstm_recall"] = lstm_metrics["sensitivity"]
    dashboard_json["lstm_gini"] = lstm_gini
    dashboard_json["lstm_optimal_threshold"] = lstm_threshold
    dashboard_json["lstm_params"] = {
        "architecture": "LSTM(2 layers) -> FC(64->32->1) -> Sigmoid",
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "seq_length": 12,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "epochs": 20,
        "batch_size": 256,
        "loss": "BCELoss (class-weighted)",
    }
    dashboard_json["lstm_history"] = {
        "val_auroc": lstm_output["history"]["val_auroc"],
    }

    # LSTM ROC curve
    lstm_fpr, lstm_tpr, _ = roc_curve(
        np.asarray(y_val_lstm), np.asarray(lstm_y_prob)
    )
    dashboard_json["lstm_fpr"] = lstm_fpr.tolist()
    dashboard_json["lstm_tpr"] = lstm_tpr.tolist()

    metrics_path = DATA_PROCESSED / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(dashboard_json, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    # ROC curve plot (all 3 models)
    roc_data = {
        "Logistic Regression": (y_val, logistic_results["y_prob"]),
        "XGBoost": (y_val, xgb_results["y_prob"]),
        "LSTM": (y_val_lstm, lstm_y_prob),
    }
    roc_path = str(DATA_PROCESSED / "roc_curves.png")
    plot_roc_curves(roc_data, save_path=roc_path)

    # Feature importance plot
    importance_path = str(DATA_PROCESSED / "feature_importance.png")
    plot_feature_importance(
        xgb_model, feature_names, top_n=30, save_path=importance_path,
    )

    # ── 8. Save LSTM training curve ────────────────────────────────────────
    print("\n[Step 8/9] Saving LSTM training history ...")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(lstm_output["history"]["train_loss"]) + 1)
    ax1.plot(epochs_range, lstm_output["history"]["train_loss"], label="Train")
    ax1.plot(epochs_range, lstm_output["history"]["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("LSTM Training Curve")
    ax1.legend()
    ax2.plot(epochs_range, lstm_output["history"]["val_auroc"], color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUROC")
    ax2.set_title("LSTM Validation AUROC")
    fig.tight_layout()
    fig.savefig(str(DATA_PROCESSED / "lstm_training.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved to {DATA_PROCESSED / 'lstm_training.png'}")

    # ── 9. Summary ──────────────────────────────────────────────────────────
    _print_summary(logistic_results, xgb_results, lstm_results)


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
