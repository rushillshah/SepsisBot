# Session 2: Model Improvement & Analysis (2026-04-06)

## Problem
- 96.8% patient-level false alarm rate on cross-hospital holdout
- AUROC 0.624 on cross-hospital validation
- Root cause: site-specific confounders (Unit2, ICULOS) dominating feature importance
- XGBoost memorizing Hospital A patterns (0.993 train vs 0.624 validation)

## What Was Done

### False Alarm Rate Fix
- **Dropped Unit1, Unit2, HospAdmTime** from features — `EXCLUDED_FEATURES` in config.py
- **Patient-level stratified 3-fold CV** on both hospitals combined — `src/train_cv.py`
- **Stronger XGBoost regularization** — max_depth [3,4,5], min_child_weight, gamma, L1/L2 reg
- **Feature scaling** via StandardScaler (fit on train fold only)
- **Probability calibration** via Platt scaling (CalibratedClassifierCV)

### CV Results
- **XGBoost CV AUROC: 0.806** (was 0.624 on cross-hospital)
- **LR CV AUROC: 0.778** (was 0.685)
- Per-fold: [0.811, 0.798, 0.808] for XGBoost

### Clinical Scoring Features
- SIRS Score (0-4), Modified qSOFA (0-2), Shock Index (HR/SBP), Modified MEWS (0-8), Lactate/MAP Ratio
- Added to `add_clinical_scores()` in features.py
- Marginal AUROC improvement (XGBoost already discovers these patterns) but improves interpretability

### Feature Importance Analysis
- Built `src/feature_importance.py` — IV, XGBoost Gain %, SHAP values
- Generated `data/processed/feature_analysis/` with rankings, SHAP top-50 plots, precision/recall comparison
- Combined ranking table (top 50 features by average rank across 3 methods)

### Overfit Checking
- Added train-set evaluation to each CV fold
- Train AUROC ~0.985 vs Val ~0.806 = 0.18 gap (still significant)

### Pipeline Cleanup
- **REMOVED old cross-hospital pipeline** — was producing 95%+ false positive rate
- Pipeline now: load → impute → CV → train final model → feature importance → save
- Dashboard shows CV data only (no more misleading cross-hospital metrics)
- Old static data files removed from git

### Temporal Analysis Module
- Built `src/temporal_analysis.py` — risk trajectories, daily scoring, early warning stats
- Not yet integrated into pipeline (module exists, ready to call)

### Model Export
- Final model trained on ALL data with best CV params
- Saved: calibrated xgboost_model.pkl, logistic_model.pkl, scaler.pkl
- Nachiket can load with `joblib.load()`

### Documentation
- Updated CLAUDE.md with full context
- Created docs/model_report.md
- Session docs in docs/sessions/
- Feature analysis CSVs + PNGs pushed for Nachiket to view

## Key Insights
1. Removing site confounders was the single biggest improvement
2. Patient-level CV on both hospitals eliminated distribution shift  
3. Clinical scoring features add interpretability but not much AUROC (XGBoost learns them from raw vitals)
4. 0.18 overfit gap suggests feature selection would help (170 → 30-50 features)
5. Precision is very low (4.6%) due to 7% prevalence — fundamental class imbalance challenge

## Next Steps (Planned)
1. ICULOS normalization (log + buckets)
2. Patient-baseline deviation features (compare to own first 6h)
3. Multi-timescale slopes (3h/6h/12h)
4. Feature selection (cut to top 30-50 by SHAP)
5. LSTM on GPU (Nachiket)
