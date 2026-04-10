"""Run CV with top-100 features only, merge results with existing 309-feature metrics."""

import gc
import json

import pandas as pd
from sklearn.metrics import roc_curve

from src.config import DATA_PROCESSED
from src.data_loader import load_processed
from src.features import build_feature_matrix, create_early_label
from src.train_cv import cross_validate_pipeline


def run():
    # Load cached data
    print("[1/3] Loading cached data ...")
    imputed_df = load_processed("imputed_data")
    print(f"  {len(imputed_df):,} rows")

    # Grab metadata before feature engineering
    patient_ids = imputed_df["patient_id"].to_numpy()
    eval_labels = imputed_df["SepsisLabel"].to_numpy()
    iculos = imputed_df["ICULOS"].to_numpy()

    # Determine which top-100 features we need so we can drop the rest early
    print("\n[2/3] Selecting top 100 features by IV ...")
    iv_path = DATA_PROCESSED / "feature_analysis" / "iv_ranking.csv"
    iv_df = pd.read_csv(iv_path)
    top_100_names = iv_df.head(100)["feature"].tolist()

    print("  Creating early_label + building features ...")
    imputed_df = create_early_label(imputed_df)
    X_all, y_early = build_feature_matrix(imputed_df, use_early_label=True)
    del imputed_df
    gc.collect()

    top_100 = [f for f in top_100_names if f in X_all.columns]
    print(f"  Selected {len(top_100)} of {X_all.shape[1]} features")

    # Select only needed columns THEN defragment to avoid 2.4GB consolidation alloc
    X_selected = pd.DataFrame(X_all[top_100].to_numpy(), columns=top_100, index=X_all.index)
    del X_all
    gc.collect()

    # Run CV on top 100 only
    print("\n[3/3] Running CV on top 100 features ...")
    cv_100 = cross_validate_pipeline(
        X_selected, y_early.to_numpy(), patient_ids, eval_labels, iculos=iculos,
    )

    # Merge with existing metrics
    metrics_path = DATA_PROCESSED / "model_metrics.json"
    with open(metrics_path) as f:
        dashboard = json.load(f)

    avg_100 = cv_100["avg_xgb_metrics"]
    avg_100_lr = cv_100["avg_lr_metrics"]
    overfit_100 = cv_100.get("overfit_table", [])
    avg_gap_100 = sum(r["xgb_gap"] for r in overfit_100) / len(overfit_100) if overfit_100 else 0
    overfit_309 = dashboard.get("cv_overfit_table", [])
    avg_gap_309 = sum(r["xgb_gap"] for r in overfit_309) / len(overfit_309) if overfit_309 else 0

    dashboard["model_comparison"] = {
        "full_309": {
            "n_features": 309,
            "xgb_auroc": dashboard["cv_xgb_auroc"],
            "xgb_auroc_std": dashboard["cv_xgb_auroc_std"],
            "lr_auroc": dashboard["cv_lr_auroc"],
            "xgb_gini": dashboard.get("cv_xgb_gini", 2 * dashboard["cv_xgb_auroc"] - 1),
            "xgb_pr_auc": dashboard["cv_xgb_pr_auc"],
            "xgb_sensitivity": dashboard.get("patient_sensitivity", 0),
            "xgb_specificity": dashboard.get("patient_specificity", 0),
            "xgb_precision": dashboard.get("patient_precision", 0),
            "xgb_f1": dashboard["cv_xgb_f1"],
            "overfit_gap": avg_gap_309,
        },
        "top_100": {
            "n_features": len(top_100),
            "xgb_auroc": avg_100["auroc"]["mean"],
            "xgb_auroc_std": avg_100["auroc"]["std"],
            "lr_auroc": avg_100_lr["auroc"]["mean"],
            "xgb_gini": 2 * avg_100["auroc"]["mean"] - 1,
            "xgb_pr_auc": avg_100["pr_auc"]["mean"],
            "xgb_sensitivity": cv_100["avg_patient_metrics"]["sensitivity"]["mean"],
            "xgb_specificity": cv_100["avg_patient_metrics"]["specificity"]["mean"],
            "xgb_precision": cv_100["avg_patient_metrics"]["precision"]["mean"],
            "xgb_f1": avg_100["f1"]["mean"],
            "overfit_gap": avg_gap_100,
        },
        "selected_features": top_100,
    }

    # ROC curve for top-100
    concat_100 = cv_100.get("concat_predictions", {})
    if "xgb_probs" in concat_100:
        fpr_100, tpr_100, _ = roc_curve(concat_100["labels"], concat_100["xgb_probs"])
        dashboard["model_comparison"]["top_100"]["fpr"] = fpr_100.tolist()
        dashboard["model_comparison"]["top_100"]["tpr"] = tpr_100.tolist()

    with open(metrics_path, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"\n  Saved comparison to {metrics_path}")

    # Print comparison
    full = dashboard["model_comparison"]["full_309"]
    slim = dashboard["model_comparison"]["top_100"]
    print(f"\n  {'':>20s} {'309 features':>14s} {'Top 100':>14s}")
    print(f"  {'XGB AUROC':>20s} {full['xgb_auroc']:>14.4f} {slim['xgb_auroc']:>14.4f}")
    print(f"  {'Overfit Gap':>20s} {full['overfit_gap']:>14.4f} {slim['overfit_gap']:>14.4f}")
    print(f"  {'Patient Sens':>20s} {full['xgb_sensitivity']:>14.4f} {slim['xgb_sensitivity']:>14.4f}")
    print(f"  {'Patient Prec':>20s} {full['xgb_precision']:>14.4f} {slim['xgb_precision']:>14.4f}")


if __name__ == "__main__":
    run()
