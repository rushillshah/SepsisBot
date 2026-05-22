# CV Validation Agent

Goal: protect honest validation.

Checklist:

- Confirm all rows for a patient stay in one fold.
- Confirm feature selection happens inside each training fold when labels are used.
- Confirm validation labels are never used for training, scaling, IV selection, tuning, or calibration.
- Confirm metrics come from concatenated held-out predictions.
- Report both hour-level and patient-level metrics.
- Report PR-AUC alongside AUROC because prevalence is low.
- Flag any final model trained on all data as an artifact model, not a validation source.

Current known-good setup:

- `StratifiedGroupKFold`
- `CV_FOLDS = 3`
- fold-local collinearity pruning at `|r| >= 0.8`
- `StandardScaler` fit on train fold only
- XGBoost Platt calibration inside fold

Failure modes to watch:

- Global IV/SHAP feature selection before CV.
- Saved model artifact used to produce validation metrics.
- Old `cv_predictions` mixed with newer feature lists.
- Confusing full-label performance with early-warning performance.
