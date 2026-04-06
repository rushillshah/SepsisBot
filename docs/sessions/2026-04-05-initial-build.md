# Session 1: Initial Build (2026-04-05)

## What Was Done

### Research & Design
- Researched open-source sepsis datasets — selected PhysioNet CinC 2019 (40K patients, open access, pre-labeled)
- Designed data pipeline, feature engineering, and model architecture
- Wrote design spec: `docs/superpowers/specs/2026-04-05-sepsis-prediction-poc-design.md`

### Data Pipeline
- Downloaded 40,336 PSV files from PhysioNet (training sets A and B)
- Built `src/data_loader.py` — parses PSV files, saves as parquet
- Built `src/imputation.py` — forward-fill + missingness flags + time-since-measured + median fill
- Built `src/features.py` — rolling windows (6h), trend deltas, feature matrix assembly (148 features)
- Built `src/config.py` — centralized column lists, hyperparams, paths

### Models
- Built `src/train.py` — Logistic Regression + XGBoost with RandomizedSearchCV
- Built `src/evaluate.py` — AUROC, PR-AUC, ROC curves, feature importance, patient-level analysis
- Built `src/train_lstm.py` — 2-layer LSTM with 12-hour sliding window (PyTorch)
- Built `run_pipeline.py` — end-to-end pipeline script

### Dashboard
- Built `app.py` — 7-page Streamlit dashboard with dual clinical/technical audience toggle
- Pages: Executive Summary, The Data, How the Model Works, Model Performance, Patient Deep Dive, What Drives Predictions, Limitations & Next Steps
- Dark theme, interactive Plotly charts

### Testing
- 40 unit tests covering data loading, imputation, and feature engineering

### Initial Results (Cross-Hospital Holdout)
- XGBoost AUROC: 0.624 (overfit — 0.993 on training CV)
- LR AUROC: 0.685
- False alarm rate: 96.8% at patient level
- Root cause: Unit2 (hospital ward type) was #1 feature — site confounder

## Key Decisions
- CinC 2019 over MIMIC-IV (open access, no credentialing)
- Per-hour snapshot approach for XGBoost/LR
- Forward-fill + missingness flags for imputation
- Train Hospital A, validate Hospital B for initial test
