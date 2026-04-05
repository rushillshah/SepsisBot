# Reduce False Alarm Rate — Design Spec

## Context

The current sepsis prediction model has a **96.8% patient-level false alarm rate** (19,290 of 18,742 non-sepsis patients flagged) and an AUROC of 0.624 on cross-hospital validation. The root causes are identified:

1. **Site-specific confounders** — Unit2 (ward type) and ICULOS are the #1 and #2 most important features, encoding hospital-specific patterns rather than clinical signal
2. **Training on one hospital only** — training on Hospital A and validating on Hospital B maximizes distribution shift
3. **No regularization** — XGBoost with max_depth=7 memorized training data (0.993 CV vs 0.624 validation)
4. **No feature scaling** — logistic regression failed to converge (max_iter=1000)
5. **Uncalibrated probabilities** — optimal threshold at 0.018 instead of ~0.5

The clinical signal IS present (Lactate, Temperature, Creatinine in top 10 features) but is buried under site-specific noise.

**Goal:** Balance sensitivity (~60%) with false alarm rate (~15-20%) by fixing the training pipeline. Target AUROC >= 0.80 on patient-level cross-validation.

## Changes

### 1. Remove Site-Specific Confounders

**Drop from features:**
- `Unit1` — hospital ward type (site-specific)
- `Unit2` — hospital ward type (site-specific)
- `HospAdmTime` — hospital admission time artifact

**Modify:** `src/config.py`
- Create `EXCLUDED_FEATURES = ["Unit1", "Unit2", "HospAdmTime"]`
- Update `DEMOGRAPHIC_COLS` to exclude these, or filter them out in feature matrix assembly

**Modify:** `src/features.py` — `build_feature_matrix()` drops `EXCLUDED_FEATURES` from X alongside patient_id/hospital/SepsisLabel.

### 2. Patient-Level Stratified Cross-Validation on Both Hospitals

**Current:** Train Hospital A, validate Hospital B (cross-site holdout).

**New primary evaluation:** 5-fold patient-level stratified CV on combined data:
- All hours for a given patient go in the same fold (no temporal leakage)
- Stratified by patient-level sepsis outcome (~7% prevalence preserved per fold)
- Both hospitals represented in every fold

**Secondary evaluation:** Keep cross-hospital holdout as a generalization stress test.

**New file:** `src/train_cv.py` with:
- `patient_stratified_split(df, n_splits=5) -> list of (train_idx, val_idx)` — groups by patient_id, stratifies by max(SepsisLabel), returns row-level indices
- `cross_validate_pipeline(df) -> dict` — runs imputation + feature engineering + training + evaluation for each fold, returns averaged metrics + per-fold details
- `train_best_model(df) -> dict` — trains on full data with best hyperparameters from CV, returns final model + evaluation

### 3. Stronger XGBoost Regularization

**New param grid** in `src/config.py`:

```python
XGBOOST_PARAM_GRID_V2 = {
    "max_depth": [3, 4, 5],            # was [3, 5, 7]
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500],
    "min_child_weight": [5, 10, 20],    # NEW — prevents tiny-group splits
    "gamma": [0.1, 0.5, 1.0],          # NEW — minimum loss reduction
    "reg_alpha": [0.1, 1.0],           # NEW — L1 regularization
    "reg_lambda": [1.0, 5.0],          # NEW — L2 regularization
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
}
```

RandomizedSearchCV with n_iter=30 (up from 20) to cover the larger space.

### 4. Feature Scaling

**Add `StandardScaler`** in the feature pipeline:
- Fit on training fold only
- Transform both train and validation folds
- Stored per fold for reproducibility

**Modify:** `src/features.py` — add `scale_features(X_train, X_val) -> (X_train_scaled, X_val_scaled, scaler)` function.

This fixes logistic regression convergence and should improve its AUROC from 0.685 to 0.75+.

### 5. Probability Calibration

**Add Platt scaling** via `sklearn.calibration.CalibratedClassifierCV`:
- Wrap the trained XGBoost model
- Calibrate on a held-out portion of the training fold (or via nested CV)
- Output probabilities should be well-calibrated, making the optimal threshold land near 0.5

**Modify:** `src/train_cv.py` — calibrate after training, before evaluation.

### 6. Evaluation Updates

**Dual reporting:**
- **Primary:** Patient-level 5-fold CV metrics (AUROC, sensitivity, specificity, precision, F1, Gini, PR-AUC) — averaged across folds with std
- **Secondary:** Cross-hospital holdout metrics (same set) — for generalization assessment

**Patient-level evaluation per fold:**
- For each patient in the validation fold, the model's max predicted probability across all their hours is compared to a threshold
- This gives patient-level sensitivity/specificity/false alarm rate

**Update:** `data/processed/model_metrics.json` to include:
- `cv_metrics` — averaged fold metrics
- `cv_fold_details` — per-fold breakdown
- `cross_hospital_metrics` — Hospital A->B holdout
- All existing dashboard fields preserved

**Update:** `app.py` — show CV results as primary, cross-hospital as secondary comparison.

## Files Modified

| File | Change |
|------|--------|
| `src/config.py` | Add EXCLUDED_FEATURES, XGBOOST_PARAM_GRID_V2 |
| `src/features.py` | Drop excluded features in build_feature_matrix, add scale_features |
| `src/train_cv.py` | NEW — patient-level CV, calibration, best model training |
| `run_pipeline.py` | Add CV pipeline step, update JSON builder |
| `app.py` | Show CV metrics as primary, cross-hospital as secondary |

## Verification

1. EXCLUDED_FEATURES are not in the feature matrix X
2. No patient appears in both train and val within any CV fold
3. Each CV fold has ~7% sepsis prevalence
4. XGBoost optimal threshold is between 0.3-0.7 (calibration working)
5. CV AUROC >= 0.80
6. Patient-level false alarm rate < 20% at the operating threshold
7. Patient-level sensitivity >= 50% at the same threshold
8. Cross-hospital AUROC > 0.70 (better than current 0.624)
9. All 40 existing tests still pass
10. Dashboard displays both CV and cross-hospital results
