# Sepsis Early Warning System - Model Report

Latest clean run: **2026-05-23**

## 1. Executive Summary

This project is a proof-of-concept early sepsis risk model using the PhysioNet/CinC 2019 ICU dataset: **40,111 patients** and **1,543,363 hourly observations** from two hospitals.

The current clean model is intentionally more conservative than earlier runs. It removes obvious leakage/confounding features, applies a leading-indicator filter, and prunes collinear features inside each CV fold.

| Model | Role | CV AUROC | CV PR-AUC |
|---|---|---:|---:|
| Logistic Regression | Baseline | 0.7480 | 0.0438 |
| XGBoost | Primary | **0.7988** | **0.0656** |
| LSTM | Sequence model | Not trained | Not trained |

Key interpretation:

> This is a cleaned early-risk PoC. It demonstrates learnable signal in hourly ICU data, but it is not severity-independent or clinically validated.

## 2. Dataset

Dataset: PhysioNet/Computing in Cardiology Challenge 2019.

| Property | Value |
|---|---:|
| Total patients | 40,111 |
| Sepsis-positive patients | 2,922 |
| Sepsis-negative patients | 37,189 |
| Total hourly rows | 1,543,363 |
| Label | `SepsisLabel = 1` beginning 6 hours before clinical onset |

Available variables:

- Vitals: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2
- Labs: 26 lab fields including FiO2, Lactate, WBC, Creatinine, Platelets
- Demographics/admin: Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS

Major missing clinical context:

- SOFA/APACHE
- diagnoses/comorbidities
- medications
- antibiotics
- vasopressors
- ventilation status
- urine output

This means the current dataset cannot prove prediction independent of baseline illness severity.

## 3. Cleaning and Feature Pipeline

Order matters:

1. Missingness flags.
2. Time since measurement.
3. Forward-fill within patient.
4. Vitals median-fill, labs zero-fill.
5. Clinical score features.
6. Normal-range deviation/drift features.
7. Rolling 6-hour features.
8. Trend features.
9. Leading-indicator filter.
10. Fold-local collinearity pruning.

Current feature policy excludes:

- `ICULOS`
- `Unit1`, `Unit2`, `HospAdmTime`
- raw `FiO2`
- lab absolute-level encodings treated as static cohort markers
- testing-frequency / missingness features
- lower-IV features from correlated pairs at `|Pearson r| >= 0.8`

Final saved model feature count: **119**.

Max retained absolute correlation: **0.773**.

## 4. Feature Selection

The current model applies two layers of feature cleanup.

### Leading-Indicator Filter

Drops features likely to represent:

- baseline sickness
- clinician action
- testing frequency
- intervention status
- direct symptom-state composites

This is controlled in `src/config.py` by `USE_LEADING_INDICATORS_ONLY = True`.

### Collinearity Pruning

Implemented in `src/feature_selection.py`.

Rule:

1. Compute IV on the training data.
2. Compute absolute Pearson correlation.
3. For any pair with `|r| >= 0.8`, drop the lower-IV feature.
4. In CV, this happens inside each training fold only.

The early-warning-score redundancy was resolved:

| Feature | Status |
|---|---|
| `early_warning_score_avg_6h` | Kept |
| `early_warning_score_max_6h` | Dropped |
| `early_warning_score` | Dropped |

Audit artifacts:

- `data/processed/feature_analysis/final_feature_list.csv`
- `data/processed/feature_analysis/collinearity_pruning_audit.csv`
- `data/processed/feature_analysis/retained_feature_correlation_audit.json`

## 5. Training and Validation

Primary evaluation:

- 3-fold patient-level stratified CV.
- All rows for a patient stay in one fold.
- Metrics come from concatenated held-out predictions.
- `StandardScaler` fit on training fold only.
- XGBoost calibrated with Platt scaling inside fold.
- Sepsis rows oversampled in training folds.

Early-onset exclusion:

- 703 sepsis patients with onset at or before ICU hour 6 are excluded.
- Reason: they are already septic on admission and inflate early-warning claims.

## 6. Results

### Cross-Validation Summary

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| AUROC | 0.7480 | **0.7988** |
| PR-AUC | 0.0438 | **0.0656** |
| Sensitivity | 0.7141 | 0.7447 |
| Specificity | 0.6571 | 0.7055 |
| Precision | 0.0287 | 0.0348 |
| F1 | 0.0552 | 0.0664 |

### Patient-Level Threshold Table

| Threshold | Sensitivity | Specificity | Precision | Flagged |
|---:|---:|---:|---:|---:|
| 0.10 | 99.1% | 14.8% | 6.5% | 33,896 |
| 0.20 | 96.5% | 35.8% | 8.2% | 26,012 |
| 0.30 | 91.4% | 52.5% | 10.3% | 19,704 |
| 0.50 | 72.1% | 79.7% | 17.5% | 9,133 |

At `threshold = 0.30`:

- TP: 2,028
- FP: 17,676
- FN: 191
- TN: 19,513

This is high sensitivity but still low precision. Alert fatigue remains a major limitation.

## 7. Feature Importance

Top combined features from the clean 119-feature model:

1. `early_warning_score_avg_6h`
2. `Temp_max_6h`
3. `Resp_avg_6h`
4. `pH_drift_from_normal_6h`
5. `MAP_min_6h`
6. `HR_max_6h`
7. `Temp_abs_deviation_from_normal`
8. `Temp_min_6h`
9. `Magnesium_drift_from_normal_6h`
10. `HR_drift_from_normal_6h`

Current feature-importance files:

- `data/processed/feature_analysis/iv_ranking.csv`
- `data/processed/feature_analysis/gain_ranking.csv`
- `data/processed/feature_analysis/shap_ranking.csv`
- `data/processed/feature_analysis/combined_ranking.csv`

Note: local SHAP PNG generation is blocked by a Python `pyexpat/libexpat` issue. `shap_ranking.csv` is generated using XGBoost native contribution values.

## 8. Clinical Interpretation

The defensible story is trajectory-oriented risk stratification.

Do say:

> The model uses recent vitals, clinical score trends, and patient-relative drift to identify elevated sepsis risk.

Do not say:

> The model predicts sepsis independent of illness severity.

Clinician concern about lactate/FiO2 being symptoms or treatment markers is valid. The current cleaned model addresses this by excluding raw FiO2 and down-weighting static lab absolute levels in favor of trajectory/drift features.

## 9. Limitations

1. No full severity adjustment.
2. ICU-only population.
3. No medication/intervention timing.
4. Low prevalence creates low precision.
5. PhysioNet labels are retrospective challenge labels.
6. Feature importance can still reflect ICU workflow artifacts.
7. Prospective clinical utility is untested.

## 10. Next Steps

Highest-value next work:

1. Validate on richer EHR data with SOFA/APACHE, medications, ventilation, urine output, and vasopressors.
2. Add patient-level alert aggregation and optimize for alert burden.
3. Run feature ablations for trajectory-only vs vitals-only vs labs-only.
4. Fix the local matplotlib/pyexpat issue and regenerate SHAP PNGs.
5. Update the presentation to use only the clean 119-feature model.
6. Build MCP/tooling for artifact audit, CV validation, and claim linting.

## 11. Project Agent System

Project-local agents and skills live in:

- `docs/agents/`
- `docs/skills/`

Use them before changing CV, feature selection, model claims, or presentation material.

*Data: PhysioNet/CinC Challenge 2019.*
*Dashboard: `streamlit run app.py`.*
