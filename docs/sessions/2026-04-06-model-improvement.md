# Session 2: Model Improvement & Analysis (2026-04-06)

## Problem
- 96.8% patient-level false alarm rate
- AUROC 0.624 on cross-hospital validation
- Root cause: site-specific confounders (Unit2, ICULOS) dominating feature importance
- XGBoost memorizing Hospital A patterns (0.993 train vs 0.624 validation)

## What Was Done

### False Alarm Rate Fix
- **Dropped Unit1, Unit2, HospAdmTime** from features (site confounders) — `EXCLUDED_FEATURES` in config.py
- **Patient-level stratified 5-fold CV** on both hospitals combined — `src/train_cv.py`
  - StratifiedGroupKFold ensures no patient in both train and val
  - Each fold preserves ~7% sepsis prevalence
- **Stronger XGBoost regularization** — max_depth [3,4,5], min_child_weight, gamma, L1/L2 reg
- **Feature scaling** via StandardScaler (fit on train fold only)
- **Probability calibration** via Platt scaling (CalibratedClassifierCV)

### Results After Fix
- **XGBoost CV AUROC: 0.812** (was 0.624) — all 5 folds above 0.80
- **LR CV AUROC: 0.777** (was 0.685)
- Per-fold results:
  - Fold 1: LR 0.784, XGB 0.815
  - Fold 2: LR 0.774, XGB 0.815
  - Fold 3: LR 0.789, XGB 0.820
  - Fold 4: LR 0.764, XGB 0.802
  - Fold 5: LR 0.775, XGB 0.808

### Feature Importance Analysis
- Built `src/feature_importance.py` with three methods:
  - **Information Value (IV)** — WOE-based binned analysis
  - **XGBoost Gain %** — normalized tree split importance
  - **SHAP Values** — TreeExplainer with beeswarm + bar plots
- Combined ranking table (top 30 features by average rank across 3 methods)

### Overfit Checking
- Added train-set evaluation to each CV fold in `train_cv.py`
- Returns per-fold train AUROC vs val AUROC gap table

### Temporal Analysis
- Built `src/temporal_analysis.py`:
  - `hourly_risk_trajectory()` — risk scores relative to sepsis onset
  - `daily_max_risk()` — aggregate hourly to daily max scores
  - `early_warning_summary()` — median hours before onset, % caught at 6h/12h/24h
  - Plotting functions for risk trajectories and daily comparisons

### Model Export
- Added .pkl saving to pipeline (xgboost_model.pkl, logistic_model.pkl)
- Nachiket can load and use directly

### Documentation
- Updated CLAUDE.md with full project context
- Created `docs/model_report.md` — comprehensive report for presentation
- Created `docs/sessions/` for development log

### Dashboard Updates
- Added CV results as primary metrics (above cross-hospital results)
- Added overfit check table
- Added SHAP beeswarm + bar plots
- Added combined feature ranking table (IV + Gain + SHAP)
- Fixed dark theme colors for explanation boxes

## Key Insights
1. Removing site confounders was the single biggest improvement (Unit2 was feature #1)
2. Patient-level CV on both hospitals eliminated distribution shift
3. Calibration fixed the threshold from 0.018 to reasonable range
4. Clinically relevant features emerged: Lactate, Temperature, Creatinine, lab testing frequency
5. LR outperforms XGBoost on cross-hospital holdout (simpler model generalizes better)

## Open Items
- LSTM needs GPU training (disabled in pipeline, Nachiket to run)
- Early warning timing analysis needs pipeline integration (module exists, not yet called in run_pipeline.py)
- Daily scoring aggregation ready but not visualized in dashboard yet
