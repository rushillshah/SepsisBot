"""Train and evaluate the early-warning variant of the sepsis model.

The full-label model (run_pipeline.py) trains on every hour where SepsisLabel=1,
which includes hours after clinical onset where the patient is already septic.
That inflates AUROC by mixing easy already-septic positives with the actual
early-warning window we care about.

This runner trains a sibling variant on a censored dataset: post-onset hours
are dropped for septic patients so the model is only rewarded for catching
the pre-onset transition. Results are merged into ``model_metrics.json`` under
the ``label_variants`` key alongside the full-label run, and a patient-level
intersection table (Venn-style: flagged by both / by full only / by early only)
is precomputed for the dashboard.

Prerequisites: ``run_pipeline.py`` has been run, producing:
  - ``data/processed/imputed_data.parquet``
  - ``data/processed/cv_predictions.parquet`` (full-label CV preds)
  - ``data/processed/model_metrics.json``
"""

from __future__ import annotations

import gc
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.config import DATA_PROCESSED, EARLY_LABEL_EXTRA_HOURS
from src.data_loader import load_processed
from src.features import build_feature_matrix, create_early_label
from src.labeling import censor_post_onset
from src.threshold_analysis import patient_intersection_at_thresholds
from src.train_cv import cross_validate_pipeline


def _exclude_early_onset(
    X: pd.DataFrame,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    patient_ids: np.ndarray,
    iculos: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Drop patients whose SepsisLabel flips on at ICULOS <= 6 (already septic on admission).

    Mirrors the exclusion in run_pipeline.py:131-148, but operates on flat arrays so
    we don't need to keep imputed_df around.
    """
    work = pd.DataFrame({"_pid": patient_ids, "_iculos": iculos, "_y_eval": y_eval})
    onset_hours = (
        work[work["_y_eval"] == 1].groupby("_pid")["_iculos"].min()
    )
    early_onset_pids = set(onset_hours[onset_hours <= 6].index)
    mask = ~pd.Series(patient_ids).isin(early_onset_pids).to_numpy()
    info = {
        "n_patients_excluded": int(len(early_onset_pids)),
        "n_rows_excluded": int((~mask).sum()),
    }
    return (
        X.loc[mask].copy(),
        np.asarray(y_train)[mask],
        np.asarray(y_eval)[mask],
        np.asarray(patient_ids)[mask],
        np.asarray(iculos)[mask],
        info,
    )


def _summarize_full_variant(metrics: dict, total_rows: int, total_patients: int) -> dict:
    """Copy headline numbers for the full-label model out of the existing metrics JSON."""
    return {
        "auroc": metrics.get("cv_xgb_auroc"),
        "auroc_std": metrics.get("cv_xgb_auroc_std"),
        "pr_auc": metrics.get("cv_xgb_pr_auc"),
        "f1": metrics.get("cv_xgb_f1"),
        "patient_sensitivity": metrics.get("patient_sensitivity"),
        "patient_specificity": metrics.get("patient_specificity"),
        "patient_precision": metrics.get("patient_precision"),
        "fpr": metrics.get("fpr", []),
        "tpr": metrics.get("tpr", []),
        "n_rows": total_rows,
        "n_patients": total_patients,
        "label_strategy": (
            "Full label: SepsisLabel=1 from t_onset-6h until discharge (PhysioNet default). "
            "Includes already-septic post-onset hours."
        ),
    }


def _summarize_early_variant(
    cv_results: dict,
    n_rows_after: int,
    n_patients_after: int,
    censor_info: dict,
) -> dict:
    avg_xgb = cv_results["avg_xgb_metrics"]
    concat = cv_results.get("concat_predictions", {})
    fpr_list, tpr_list = [], []
    if "xgb_probs" in concat:
        fpr, tpr, _ = roc_curve(concat["labels"], concat["xgb_probs"])
        fpr_list, tpr_list = fpr.tolist(), tpr.tolist()

    return {
        "auroc": avg_xgb["auroc"]["mean"],
        "auroc_std": avg_xgb["auroc"]["std"],
        "pr_auc": avg_xgb["pr_auc"]["mean"],
        "f1": avg_xgb["f1"]["mean"],
        "patient_sensitivity": cv_results["avg_patient_metrics"]["sensitivity"]["mean"],
        "patient_specificity": cv_results["avg_patient_metrics"]["specificity"]["mean"],
        "patient_precision": cv_results["avg_patient_metrics"]["precision"]["mean"],
        "fpr": fpr_list,
        "tpr": tpr_list,
        "n_rows": int(n_rows_after),
        "n_patients": int(n_patients_after),
        "n_rows_censored": censor_info["n_rows_censored"],
        "n_patients_with_censored_rows": censor_info["n_patients_with_censored_rows"],
        "label_strategy": (
            f"Early-warning only: rows where ICULOS >= t_onset are dropped "
            f"(t_onset = first SepsisLabel=1 + {EARLY_LABEL_EXTRA_HOURS}). "
            f"Model only sees the pre-onset trajectory."
        ),
    }


def _intersection_thresholds() -> list[float]:
    return [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]


def run() -> None:
    print("=" * 60)
    print("EARLY-WARNING VARIANT (post-onset hours censored)")
    print("=" * 60)

    metrics_path = DATA_PROCESSED / "model_metrics.json"
    full_preds_path = DATA_PROCESSED / "cv_predictions.parquet"
    if not metrics_path.exists() or not full_preds_path.exists():
        raise FileNotFoundError(
            "Missing artifacts from full-label run. Execute `python run_pipeline.py` first."
        )

    # ── 1. Load cached imputed data ─────────────────────────────────────────
    print("\n[1/5] Loading cached imputed data ...")
    imputed_df = load_processed("imputed_data")
    print(f"  {len(imputed_df):,} rows")

    # ── 2. Build features ──────────────────────────────────────────────────
    print("\n[2/5] Building features ...")
    imputed_df = create_early_label(imputed_df)
    X_all, y_early = build_feature_matrix(imputed_df, use_early_label=True)

    from src.config import USE_LEADING_INDICATORS_ONLY
    if USE_LEADING_INDICATORS_ONLY:
        from src.features import select_leading_indicators
        keep, drop_reasons = select_leading_indicators(list(X_all.columns))
        reason_counts = pd.Series(drop_reasons).value_counts().to_dict()
        print(f"  Leading-indicator filter: keep {len(keep)}, "
              f"drop {len(drop_reasons)} — {reason_counts}")
        X_all = X_all[keep]

    feature_names = list(X_all.columns)
    patient_ids = imputed_df["patient_id"].to_numpy()
    eval_labels = imputed_df["SepsisLabel"].to_numpy()
    iculos = imputed_df["ICULOS"].to_numpy()
    del imputed_df
    gc.collect()
    print(f"  Features: {len(feature_names)}, Rows: {len(X_all):,}")

    # ── 3. Apply early-onset exclusion + post-onset censoring ──────────────
    print("\n[3/5] Filtering rows ...")
    y_early_np = y_early.to_numpy() if hasattr(y_early, "to_numpy") else np.asarray(y_early)

    (X_all, y_early_np, eval_labels, patient_ids, iculos, exclusion_info) = _exclude_early_onset(
        X_all, y_early_np, eval_labels, patient_ids, iculos
    )
    print(
        f"  Excluded {exclusion_info['n_patients_excluded']} early-onset sepsis patients "
        f"({exclusion_info['n_rows_excluded']:,} rows)"
    )

    (X_all, y_early_np, eval_labels, patient_ids, iculos, censor_info) = censor_post_onset(
        X_all, y_early_np, eval_labels, patient_ids, iculos
    )
    print(
        f"  Censored {censor_info['n_rows_censored']:,} post-onset rows from "
        f"{censor_info['n_patients_with_censored_rows']} septic patients"
    )
    print(f"  Remaining: {len(X_all):,} rows, {len(set(patient_ids)):,} patients")

    # ── 4. Cross-validate ──────────────────────────────────────────────────
    print("\n[4/5] Running patient-level cross-validation ...")
    cv_results = cross_validate_pipeline(
        X_all, y_early_np, patient_ids, eval_labels, iculos=iculos,
    )

    # Save the early-variant predictions
    concat = cv_results.get("concat_predictions", {})
    early_preds_df = pd.DataFrame()
    if "xgb_probs" in concat:
        early_preds_df = pd.DataFrame({
            "patient_id": concat["patient_ids"],
            "label": concat["labels"],
            "iculos": concat["iculos"],
            "xgb_prob": concat["xgb_probs"],
            "lr_prob": concat["lr_probs"],
        })
        early_preds_path = DATA_PROCESSED / "cv_predictions_early.parquet"
        early_preds_df.to_parquet(early_preds_path, index=False)
        print(f"  Saved early-variant predictions to {early_preds_path} ({len(early_preds_df):,} rows)")

    # ── 5. Merge into model_metrics.json ───────────────────────────────────
    print("\n[5/5] Merging metrics into model_metrics.json ...")
    with open(metrics_path) as f:
        dashboard = json.load(f)

    full_preds_df = pd.read_parquet(full_preds_path)

    # Total row/patient counts for the full variant — derive from the full preds parquet.
    full_n_rows = len(full_preds_df)
    full_n_patients = int(full_preds_df["patient_id"].nunique())

    # Intersection: join on patient_id, max prob per patient.
    intersection_table = pd.DataFrame()
    if not early_preds_df.empty:
        full_for_join = full_preds_df.rename(columns={"xgb_prob": "prob"})[["patient_id", "label", "prob"]]
        early_for_join = early_preds_df.rename(columns={"xgb_prob": "prob"})[["patient_id", "label", "prob"]]
        intersection_table = patient_intersection_at_thresholds(
            full_for_join, early_for_join, thresholds=_intersection_thresholds(),
        )

    dashboard["label_variants"] = {
        "full_label": _summarize_full_variant(dashboard, full_n_rows, full_n_patients),
        "early_warning": _summarize_early_variant(
            cv_results, len(X_all), len(set(patient_ids)), censor_info
        ),
        "intersection": {
            "thresholds": _intersection_thresholds(),
            "rows": intersection_table.to_dict(orient="records") if not intersection_table.empty else [],
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"  Updated {metrics_path}")

    # ── Console summary ────────────────────────────────────────────────────
    full = dashboard["label_variants"]["full_label"]
    early = dashboard["label_variants"]["early_warning"]
    print("\n" + "=" * 60)
    print("EARLY-WARNING VARIANT — SUMMARY")
    print("=" * 60)
    print(f"  {'':>22s} {'Full Label':>12s} {'Early-Warning':>14s}")
    print(f"  {'AUROC':>22s} {full['auroc']:>12.4f} {early['auroc']:>14.4f}")
    print(f"  {'PR-AUC':>22s} {full['pr_auc']:>12.4f} {early['pr_auc']:>14.4f}")
    print(f"  {'Patient Sensitivity':>22s} {full['patient_sensitivity']:>12.4f} {early['patient_sensitivity']:>14.4f}")
    print(f"  {'Patient Precision':>22s} {full['patient_precision']:>12.4f} {early['patient_precision']:>14.4f}")
    print(f"  {'Rows':>22s} {full['n_rows']:>12,} {early['n_rows']:>14,}")
    print(f"  {'Patients':>22s} {full['n_patients']:>12,} {early['n_patients']:>14,}")

    if not intersection_table.empty:
        default_t = 0.30
        row = intersection_table[intersection_table["threshold"] == default_t]
        if not row.empty:
            r = row.iloc[0]
            print(f"\n  Intersection at threshold {default_t}:")
            print(f"    Total patients:        {int(r['total_patients']):,}")
            print(f"    Actual sepsis:         {int(r['actual_sepsis']):,}")
            print(f"    Flagged by full:       {int(r['flagged_full']):,}")
            print(f"    Flagged by early:      {int(r['flagged_early']):,}")
            print(f"    Flagged by BOTH:       {int(r['flagged_both']):,}")
            print(f"    Of those, true sepsis: {int(r['tp_intersection']):,} ({r['precision_intersection']:.1%})")
    print("=" * 60)


if __name__ == "__main__":
    run()
