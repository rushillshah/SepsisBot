# CLAUDE.md — Sepsis Early Warning System

## On Session Start

Read the session logs in `docs/sessions/` for detailed development history, past decisions, and context behind the current state. Also read `docs/model_report.md` for the full model documentation. These files contain critical context that isn't duplicated here.

## What This Project Is

A proof-of-concept predictive model for **early sepsis detection** from ICU patient data. The goal is to demonstrate feasibility with open-source data (PhysioNet CinC 2019) so we can get access to real patient data for a production system.

The core question: given a patient's hourly vitals and periodic lab draws, can we predict sepsis **up to 6 hours before clinical onset**?

## Current Status (as of 2026-04-07)

- **XGBoost CV AUROC: 0.806** (3-fold patient-level stratified CV on both hospitals) — but this is HOUR-LEVEL
- **True patient-level: Sensitivity 95.7%, Specificity 56.0%, Precision 14.6%**
- Old cross-hospital holdout approach has been REMOVED — CV is the only evaluation
- LSTM model built but not yet trained on GPU (use `src/train_lstm.py`)
- Model .pkl files saved to `data/processed/models/` (calibrated XGBoost + LR + scaler)
- Feature importance: IV, XGBoost Gain %, SHAP values computed (see `data/processed/feature_analysis/`)
- Dashboard live at localhost:8501 — **WARNING: confusion matrix and some metrics are wrong (see Known Issues)**

### KNOWN ISSUES (from Nachiket's review 2026-04-07)
1. **Dashboard confusion matrix is fabricated** — estimated from hour-level rates applied to patient counts, NOT from actual predictions
2. **Metrics labeled "patient-level" are actually hour-level** — AUROC 0.806, sensitivity 73.9% etc are per-hour, not per-patient
3. **Median imputation is harmful** — labs with >50% missingness filled with population median; XGBoost handles NaN natively
4. **ICULOS still a top feature** — possible site confounder despite removing Unit1/Unit2
5. **Threshold miscalibrated** — XGBoost probabilities max at ~0.27, optimal threshold ~0.025, very trigger-happy
6. See `todos/todos.md` for full fix list

### What's Been Done
1. Full data pipeline (load 40K PSV files, impute, feature engineer)
2. Three models: LR, XGBoost, LSTM (LSTM needs GPU training)
3. Removed site confounders (Unit1, Unit2, HospAdmTime) — this was the #1 fix
4. Patient-level stratified 3-fold CV with Platt calibration
5. Stronger XGBoost regularization (max_depth 3-5, L1/L2 reg, min_child_weight, gamma)
6. Feature scaling via StandardScaler
7. Clinical scoring features (SIRS, qSOFA mod, Shock Index, MEWS mod, Lactate/MAP)
8. Feature importance analysis (IV, Gain, SHAP) — saved to `data/processed/feature_analysis/`
9. Overfit checking (train vs val per fold)
10. Temporal analysis module (early warning timing) — `src/temporal_analysis.py`
11. Interactive Streamlit dashboard (dual clinical/technical audience, dark theme)
12. Comprehensive model report (`docs/model_report.md`)
13. Old cross-hospital pipeline removed — was producing 95%+ false positive rate

### What Still Needs Work
1. **Run LSTM on GPU** — `train_lstm_pipeline()` in `src/train_lstm.py`, currently disabled in run_pipeline.py
2. **ICULOS normalization** — replace raw ICULOS with log + buckets (plan written, not implemented)
3. **Patient-baseline deviation features** — compare vitals to patient's own first-6h baseline
4. **Multi-timescale slopes** — 3h/6h/12h rate-of-change (currently only 1h deltas)
5. **Feature selection** — cut from ~170 to top 30-50 by SHAP to reduce overfit gap
6. **MIMIC-IV integration** — adds 7 missing variables

## Data Source

**PhysioNet/CinC Challenge 2019** — 40,111 ICU patients, 1.5M hourly observations, 2 hospitals.

- Open access (CC BY 4.0), no credentialing needed
- Download: ~40K individual PSV files from `https://physionet.org/files/challenge-2019/1.0.0/training/`
- `SepsisLabel` column flips to 1 at 6 hours before Sepsis-3 clinical onset
- Data goes into `data/raw/training_setA/` and `data/raw/training_setB/`
- Processed parquet files and model metrics go into `data/processed/` (gitignored)
- Sepsis prevalence: 7.3% (2,922 / 40,111 patients)

### Variables Available (34 clinical columns)
- **Vitals (8):** HR, SpO2, Temp, SBP, DBP, MAP, Resp, EtCO2
- **Labs (26):** WBC, Hgb, Hct, Platelets, PTT, Fibrinogen, FiO2, Bilirubin (total + direct), Lactate, Creatinine, BUN, Glucose, + 14 others
- **Demographics (5):** Age, Gender, Unit1, Unit2, HospAdmTime — but **Unit1, Unit2, HospAdmTime are EXCLUDED** as site confounders

### Variables Missing (7 — need MIMIC-IV)
Urine output, Procalcitonin, CRP, INR, Vasopressor/inotropic use, Ventilator status, Albumin

## Data Cleaning Pipeline

Order matters — each step depends on the previous:

1. **Missingness flags** — binary `{col}_measured` for each of 26 lab columns (1 if drawn this hour, 0 if not)
2. **Time since measurement** — `{col}_hours_since` counting hours since last draw, -1 sentinel before first draw
3. **Forward-fill** — carry last known lab value forward within each patient
4. **Median fill** — remaining NaN (before any measurement) filled with column medians
5. **Final NaN fill** — `X.fillna(0.0)` in build_feature_matrix catches any stragglers

**Critical insight:** Missingness flags and time-since-measured are computed BEFORE forward-fill so they reflect actual draw times, not imputed values. The missingness pattern itself is clinically informative (sicker patients get tested more often).

## Feature Engineering (145 features after exclusions)

| Category | Count | Examples |
|----------|------:|---------|
| Raw vitals | 8 | HR, Resp, MAP, O2Sat |
| Raw labs | 26 | WBC, Lactate, Creatinine |
| Demographics | 2 | Age, Gender (Unit1/Unit2/HospAdmTime excluded) |
| Time | 1 | ICULOS (hours in ICU) |
| Missingness flags | 26 | WBC_measured, Lactate_measured |
| Time since measurement | 26 | WBC_hours_since, Lactate_hours_since |
| Rolling window stats (6h) | 48 | HR_roll_mean, Lactate_roll_std, MAP_roll_max |
| Trend deltas | 8 | HR_delta, Resp_delta, MAP_delta |

Rolling window: 6-hour backward-looking window with min_periods=1 (no future data leakage).

## Architecture — Two Fundamentally Different Approaches

### Approach 1: Snapshot Models (LR + XGBoost) — CV Pipeline

**How it works:** Each hourly row is treated as an independent prediction. The model sees a snapshot of the patient at one moment in time — 170 engineered features including raw vitals/labs, rolling statistics over the last 6 hours, hour-over-hour deltas, missingness patterns, and clinical scores (SIRS, qSOFA, etc.). It asks: "given everything I see RIGHT NOW, is this patient heading toward sepsis?"

**Why it works:** The feature engineering does the temporal heavy lifting. Rolling means, mins, maxes, and standard deviations over 6 hours capture trends without the model needing to understand sequences. XGBoost is excellent at tabular data with complex interactions (e.g., "high lactate AND rising HR AND falling MAP").

**Evaluation:** 3-fold patient-level stratified CV (StratifiedGroupKFold). All hours for one patient stay in the same fold — no leakage. Both hospitals represented in every fold. XGBoost is Platt-calibrated so probabilities are meaningful.

**Current AUROC: 0.806 (XGBoost), 0.778 (LR)**

### Approach 2: LSTM Sequence Model — NOT YET TRAINED

**How it works:** Instead of one row = one prediction, the LSTM sees a *sequence* of the last 12 hours as a single input. Shape: (12 timesteps, 39 raw features). It uses only raw vitals + labs + demographics — NO engineered features (no rolling stats, no missingness flags, no clinical scores). The LSTM learns its own temporal patterns from the sequence structure.

**Why it might be better:** The LSTM sees the actual temporal evolution — not just "HR was high and variable over 6 hours" but the exact trajectory: "HR was 80, then 85, then 90, then 95, then 100, then 105." This can capture patterns that rolling statistics miss, like the *shape* of deterioration (sudden vs gradual).

**Why it might be worse:** LSTMs need more data to train well, are harder to interpret (no feature importance), and the 12-hour window is arbitrary. Also, 39 raw features per timestep means the LSTM has to discover patterns that XGBoost gets for free via feature engineering.

**Status:** Module built (`src/train_lstm.py`), but training hangs on MPS (Apple Silicon GPU). Nachiket plans to train on a proper GPU. Not included in the pipeline.

### Which to Use

For the PoC presentation, use the **CV XGBoost results** (0.806 AUROC). They're validated, interpretable (SHAP plots), and the model .pkl is saved. The LSTM is a stretch goal — if it beats 0.806 on the same CV setup, it becomes the primary model. If not, XGBoost with good feature engineering is hard to beat on tabular ICU data.

### Key Difference Summary

| | Snapshot (XGBoost/LR) | Sequence (LSTM) |
|---|---|---|
| Input shape | (1 row, 170 features) | (12 hours, 39 features) |
| Feature engineering | Heavy (rolling, trends, missingness, clinical scores) | None (raw values only) |
| Temporal awareness | Via engineered features (rolling stats, deltas) | Native (learns from sequence) |
| Interpretability | High (SHAP, feature importance) | Low (black box) |
| Training speed | Fast (~25 min for full CV) | Slow (needs GPU) |
| Current AUROC | **0.806** | Not yet trained |

### Pipeline Details
- Feature scaling: StandardScaler fit on train fold only
- XGBoost calibrated via Platt scaling (CalibratedClassifierCV)
- Class imbalance: scale_pos_weight (XGBoost), class_weight='balanced' (LR)

### XGBoost Hyperparameter Grid (V2 — with regularization)
```
max_depth: [3, 4, 5], learning_rate: [0.01, 0.05, 0.1], n_estimators: [100, 300, 500]
min_child_weight: [5, 10, 20], gamma: [0.1, 0.5, 1.0]
reg_alpha: [0.1, 1.0], reg_lambda: [1.0, 5.0]
subsample: [0.7, 0.8], colsample_bytree: [0.7, 0.8]
```
Tuned via RandomizedSearchCV (30 iterations, 3-fold inner CV).

### LSTM (Sequence Model)
- 2-layer LSTM (hidden=64, dropout=0.3) → FC(64→32→1) → Sigmoid
- 12-hour sliding window over raw vitals+labs+demographics (39 features, NO engineered features)
- LSTM learns its own temporal patterns from the sequence structure
- StandardScaler fit on Hospital A, applied to both
- Class-weighted BCE loss, Adam optimizer, 20 epochs
- **Currently disabled in run_pipeline.py** — run separately on GPU

## Project Structure

```
src/config.py              — Feature lists, hyperparams, paths (SINGLE SOURCE OF TRUTH)
src/data_loader.py         — PSV parsing, parquet I/O
src/imputation.py          — Forward-fill + missingness flags + median fill
src/features.py            — Rolling windows, trends, feature matrix, scale_features()
src/train.py               — LR + XGBoost training (cross-hospital)
src/train_cv.py            — Patient-level stratified CV with calibration + overfit check
src/train_lstm.py          — LSTM sequence model (PyTorch)
src/evaluate.py            — Metrics, ROC plots, patient-level analysis
src/feature_importance.py  — IV, XGBoost Gain %, SHAP values, combined ranking
src/temporal_analysis.py   — Early warning timing: risk trajectories, daily scoring
run_pipeline.py            — End-to-end pipeline (load → impute → CV → train → evaluate → save)
app.py                     — Streamlit dashboard (7 pages, dual audience toggle, dark theme)
tests/                     — 47 unit tests (pytest) using synthetic data
docs/model_report.md       — Comprehensive model documentation for presentation
docs/sessions/             — Session-by-session development log
```

## Running

```bash
source .venv/bin/activate
python run_pipeline.py      # Full pipeline (~45 min: CV + training + feature importance)
streamlit run app.py        # Dashboard at localhost:8501
pytest tests/ -v            # 47 unit tests
```

Pipeline caches raw data as parquet — subsequent runs skip the 40K file parse.
Model .pkl files saved to `data/processed/models/` (xgboost_model.pkl, logistic_model.pkl).

## Key Results

| Metric | XGBoost (CV) | LR (CV) | XGBoost (Cross-Hospital) |
|--------|:------------:|:-------:|:------------------------:|
| AUROC | **0.812** | 0.777 | ~0.65 |
| Gini | 0.624 | 0.554 | ~0.30 |

### Top Predictive Features (from SHAP + IV + Gain analysis)
1. **Lactate** (rolling std, min, hours_since) — validated sepsis biomarker
2. **Temperature** (rolling max) — fever is a classic sepsis sign
3. **Creatinine** (rolling max) — kidney dysfunction in sepsis
4. **Lab testing frequency** (hours_since features) — sicker patients tested more
5. **Heart rate / Respiratory rate trends** — tachycardia/tachypnea are early indicators

### What Fixed the Model (from AUROC 0.624 → 0.812)
1. **Dropped Unit1/Unit2/HospAdmTime** — site confounders were #1/#2 most important features
2. **Patient-level CV on both hospitals** — eliminated distribution shift
3. **Stronger regularization** — max_depth 3-5 (was 7), L1/L2 reg, min_child_weight
4. **Feature scaling** — fixed LR convergence, improved all models
5. **Platt calibration** — threshold moved from 0.018 to reasonable range

## Conventions

- **Immutability**: all pipeline functions return new DataFrames, never mutate inputs
- **Config is centralized**: column lists, hyperparams, paths all in `src/config.py`
- **Feature engineering order matters**: missingness flags → time-since-measured → forward-fill → median fill → rolling/trend features
- **LSTM uses raw features only** (no rolling/trend/missingness — it learns temporal patterns from sequences)
- **Snapshot models use engineered features** (145 total from features.build_feature_matrix)
- **EXCLUDED_FEATURES** in config.py: Unit1, Unit2, HospAdmTime — always check this list before adding features

## For Nachiket

- Model .pkl files: `data/processed/models/xgboost_model.pkl` and `logistic_model.pkl`
- Full model documentation: `docs/model_report.md`
- To load and use the model:
```python
import joblib
model = joblib.load("data/processed/models/xgboost_model.pkl")
# model.predict_proba(X)[:, 1] gives sepsis risk scores
```
- LSTM: run `src/train_lstm.py` on GPU — it's a PyTorch model, needs `train_lstm_pipeline(imputed_df)`
- Temporal analysis: `src/temporal_analysis.py` has `early_warning_summary()`, `daily_max_risk()`, `hourly_risk_trajectory()`
