---
name: sepsis-cv-validation
description: Use when changing or reviewing cross-validation, labels, threshold metrics, patient-level metrics, calibration, or training/evaluation separation in the SepsisDataModel project.
---

# Sepsis CV Validation

Use this workflow before trusting model metrics.

Steps:

1. Confirm patient-level grouping: no patient appears in both train and validation.
2. Confirm labels used for training vs evaluation are explicit.
3. Confirm feature selection using labels is fold-local.
4. Confirm scaler/calibration/tuning are fit only on training data.
5. Confirm validation metrics come from held-out predictions.
6. Report AUROC, PR-AUC, sensitivity, specificity, precision, and patient-level threshold table.
7. Save or inspect `model_metrics.json` and `cv_predictions.parquet`.

Red flags:

- final full-data model used for validation metrics
- global IV/SHAP feature selection before CV
- stale `cv_predictions`
- patient-level claims from hour-level metrics
