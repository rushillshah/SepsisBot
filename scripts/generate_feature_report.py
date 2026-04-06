"""Generate feature analysis report for presentation.

Produces:
- IV ranking CSV
- XGBoost gain ranking CSV
- SHAP ranking CSV + top-50 beeswarm/bar plots
- Combined ranking CSV (top 50)
- Precision/recall comparison (CV folds vs cross-hospital holdout)

Reads from cached data (parquet + model .pkl) — no retraining needed.

Usage: python scripts/generate_feature_report.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PROCESSED, LABEL_COL
from src.data_loader import load_processed
from src.feature_importance import (
    compute_gain_importance,
    compute_information_value,
    combined_feature_ranking,
)
from src.features import build_feature_matrix
from src.imputation import impute

OUTPUT_DIR = DATA_PROCESSED / "feature_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("FEATURE ANALYSIS REPORT")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading data ...")
    raw_df = load_processed("raw_data")
    print(f"  Raw data: {len(raw_df):,} rows")

    print("  Running imputation ...")
    imputed_df = impute(raw_df)

    print("  Building feature matrix ...")
    X_full, y_full = build_feature_matrix(imputed_df)
    feature_names = list(X_full.columns)
    print(f"  Features: {len(feature_names)}")

    # ── Load model ─────────────────────────────────────────────────────────
    print("\n[2/6] Loading XGBoost model ...")
    model_path = DATA_PROCESSED / "models" / "xgboost_model.pkl"
    xgb_model = joblib.load(model_path)
    print(f"  Loaded from {model_path}")

    # ── IV Ranking ─────────────────────────────────────────────────────────
    print("\n[3/6] Computing Information Value ...")
    iv_df = compute_information_value(X_full, y_full)
    iv_df.to_csv(OUTPUT_DIR / "iv_ranking.csv", index=False)
    print(f"  Saved iv_ranking.csv ({len(iv_df)} features)")
    print("  Top 10 by IV:")
    print(iv_df.head(10)[["feature", "iv", "iv_strength"]].to_string(index=False))

    # ── Gain Ranking ───────────────────────────────────────────────────────
    print("\n[4/6] Computing XGBoost Gain % ...")
    gain_df = compute_gain_importance(xgb_model, feature_names)
    gain_df.to_csv(OUTPUT_DIR / "gain_ranking.csv", index=False)
    print(f"  Saved gain_ranking.csv ({len(gain_df)} features)")
    print("  Top 10 by Gain:")
    print(gain_df.head(10)[["feature", "gain_pct"]].to_string(index=False))

    # ── SHAP Values + Plots ────────────────────────────────────────────────
    print("\n[5/6] Computing SHAP values (10K sample) ...")
    sample_idx = np.random.RandomState(42).choice(
        len(X_full), size=min(10_000, len(X_full)), replace=False,
    )
    X_sample = X_full.iloc[sample_idx]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # Handle binary classifier output (list of 2 arrays)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Mean |SHAP| ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df.to_csv(OUTPUT_DIR / "shap_ranking.csv", index=False)
    print(f"  Saved shap_ranking.csv")
    print("  Top 10 by SHAP:")
    print(shap_df.head(10).to_string(index=False))

    # SHAP beeswarm plot — top 50
    print("  Generating SHAP top-50 beeswarm plot ...")
    plt.figure(figsize=(12, 20))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        max_display=50,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_top50_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved shap_top50_beeswarm.png")

    # SHAP bar plot — top 50
    print("  Generating SHAP top-50 bar plot ...")
    plt.figure(figsize=(10, 16))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=50,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_top50_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved shap_top50_bar.png")

    # ── Combined Ranking ───────────────────────────────────────────────────
    ranking_df = combined_feature_ranking(iv_df, gain_df, shap_df, top_n=50)
    ranking_df.to_csv(OUTPUT_DIR / "combined_ranking.csv", index=False)
    print(f"\n  Saved combined_ranking.csv (top 50)")

    # ── Precision / Recall Comparison ──────────────────────────────────────
    print("\n[6/6] Building precision/recall comparison ...")
    metrics_path = DATA_PROCESSED / "model_metrics.json"
    with open(metrics_path) as f:
        m = json.load(f)

    rows = []

    # CV fold metrics (from overfit table — we have auroc per fold, need precision/recall)
    # The model_metrics.json has cv_xgb_* averaged metrics
    rows.append({
        "Set": "CV Mean (XGBoost)",
        "Precision": m.get("cv_xgb_precision", "N/A"),
        "Recall": m.get("cv_xgb_sensitivity", "N/A"),
        "AUROC": m.get("cv_xgb_auroc", "N/A"),
        "F1": m.get("cv_xgb_f1", "N/A"),
    })
    rows.append({
        "Set": "CV Mean (Logistic Reg)",
        "Precision": m.get("cv_lr_precision", "N/A"),
        "Recall": m.get("cv_lr_sensitivity", "N/A"),
        "AUROC": m.get("cv_lr_auroc", "N/A"),
        "F1": m.get("cv_lr_f1", "N/A"),
    })

    # Cross-hospital holdout
    rows.append({
        "Set": "Cross-Hospital Holdout (XGBoost)",
        "Precision": m.get("precision", "N/A"),
        "Recall": m.get("sensitivity", "N/A"),
        "AUROC": m.get("auroc", "N/A"),
        "F1": m.get("f1", "N/A"),
    })
    rows.append({
        "Set": "Cross-Hospital Holdout (Logistic Reg)",
        "Precision": m.get("lr_precision", "N/A"),
        "Recall": m.get("lr_sensitivity", "N/A"),
        "AUROC": m.get("lr_auroc", "N/A"),
        "F1": m.get("lr_f1", "N/A"),
    })

    pr_df = pd.DataFrame(rows)
    pr_df.to_csv(OUTPUT_DIR / "precision_recall_comparison.csv", index=False)
    print(f"  Saved precision_recall_comparison.csv")
    print(pr_df.to_string(index=False))

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("REPORT COMPLETE")
    print("=" * 60)
    print(f"\nAll files saved to: {OUTPUT_DIR}")
    print("Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
