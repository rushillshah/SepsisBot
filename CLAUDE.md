# CLAUDE.md — Sepsis Early Warning System

## What This Project Is

A proof-of-concept predictive model for **early sepsis detection** from ICU patient data. The goal is to demonstrate feasibility with open-source data (PhysioNet CinC 2019) so we can get access to real patient data for a production system.

The core question: given a patient's hourly vitals and periodic lab draws, can we predict sepsis **up to 6 hours before clinical onset**?

## Data Source

**PhysioNet/CinC Challenge 2019** — 40,111 ICU patients, 1.5M hourly observations, 2 hospitals.

- Open access (CC BY 4.0), no credentialing needed
- Download: ~40K individual PSV files from `https://physionet.org/files/challenge-2019/1.0.0/training/`
- `SepsisLabel` column flips to 1 at 6 hours before Sepsis-3 clinical onset
- Data goes into `data/raw/training_setA/` and `data/raw/training_setB/`
- Processed parquet files and model metrics go into `data/processed/` (gitignored)

### Variables We Have (15 of 22 target)

Vitals: HR, SpO2, Temp, SBP, DBP, MAP, Resp, EtCO2
Labs: WBC, Hgb, Hct, Platelets, PTT, Fibrinogen, FiO2, Bilirubin, Lactate, Creatinine, + 14 others

### Variables We're Missing (7)

Urine output, Procalcitonin, CRP, INR, Vasopressor/inotropic use, Ventilator status, Albumin. These would come from MIMIC-IV in a future iteration.

## Architecture

Three models trained and compared:

1. **Logistic Regression** — L2 baseline, class_weight='balanced'. Best cross-hospital AUROC so far (0.685).
2. **XGBoost** — RandomizedSearchCV (20 iter, 5-fold CV). Overfits to training hospital (0.993 CV vs 0.624 validation).
3. **LSTM** — 2-layer LSTM (hidden=64, dropout=0.3) with 12-hour sliding window. Sees temporal evolution directly from raw values.

### Key Design Decisions

- **Per-hour snapshot** for LR/XGBoost: each hourly row = one training example with 148 engineered features (rolling stats, trends, missingness flags)
- **Sequence model** for LSTM: 12-hour sliding window over raw vitals+labs+demographics only (LSTM learns its own temporal patterns)
- **Train on Hospital A, validate on Hospital B** — cross-hospital holdout is the harshest realistic test
- **Imputation: forward-fill + missingness flags** — missingness metadata (was this lab drawn? how long since last draw?) is itself predictive. Flags computed BEFORE forward-fill.
- **Class imbalance (~13:1)** handled via scale_pos_weight (XGBoost), class_weight='balanced' (LR), sample-weighted BCE loss (LSTM)

### Known Issues

1. **XGBoost overfits severely** — Unit2 and ICULOS dominate feature importance (site-specific confounders, not clinical signal)
2. **Cross-hospital distribution shift** — different measurement patterns, populations, and protocols between hospitals
3. **Logistic regression needs feature scaling** — convergence warning at max_iter=1000
4. **Remaining NaN after forward-fill** filled with column medians + final `X.fillna(0.0)` in `build_feature_matrix`

### Improvement Roadmap (in priority order)

1. Train on BOTH hospitals with stratified CV (most impactful)
2. Drop Unit1/Unit2 from features (removes site confounders)
3. Add StandardScaler for logistic regression
4. Reduce XGBoost max_depth to 3-4 (less overfitting)
5. Feature selection (drop low-importance noise features)
6. MIMIC-IV integration for missing variables

## Project Structure

```
src/config.py          — Feature lists, hyperparams, paths (SINGLE SOURCE OF TRUTH for column names)
src/data_loader.py     — PSV parsing, parquet I/O
src/imputation.py      — Forward-fill + missingness flags + median fill
src/features.py        — Rolling windows, trends, feature matrix (for snapshot models)
src/train.py           — Logistic Regression + XGBoost training
src/train_lstm.py      — LSTM sequence model (PyTorch)
src/evaluate.py        — Metrics, ROC plots, patient-level analysis
run_pipeline.py        — End-to-end: load -> impute -> train all 3 models -> evaluate -> save JSON
app.py                 — Streamlit dashboard (7 pages, dual clinical/technical audience toggle)
tests/                 — 40 unit tests (pytest) using synthetic data
```

## Running

```bash
source .venv/bin/activate
python run_pipeline.py      # Full pipeline (load, impute, train, evaluate, save)
streamlit run app.py        # Dashboard at localhost:8501
pytest tests/ -v            # Unit tests
```

Pipeline caches raw data as parquet after first load — subsequent runs skip the 40K file parse.

## Dashboard

7 pages with a sidebar toggle between "Clinical / Non-Technical" and "Data Science / Technical" explanations. Dark theme. Shows: executive summary, data exploration, model explanation, full metrics table (AUC, Gini, Recall, Precision, F1, PR-AUC for all 3 models), ROC curves, PR curves, patient deep dive, feature importance, limitations.

Model metrics are saved to `data/processed/model_metrics.json` — the dashboard reads from this file.

## Conventions

- **Immutability**: all pipeline functions return new DataFrames, never mutate inputs
- **Config is centralized**: column lists, hyperparams, paths all in `src/config.py`
- **Feature engineering order matters**: missingness flags -> time-since-measured -> forward-fill -> median fill -> rolling/trend features
- **LSTM uses raw features only** (no rolling/trend/missingness from features.py — it learns temporal patterns from sequences)
- **Snapshot models use engineered features** (148 total from features.build_feature_matrix)
