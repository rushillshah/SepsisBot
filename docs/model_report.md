# Sepsis Early Warning System - Model Report

## 1. Executive Summary

This proof-of-concept predicts sepsis up to **6 hours before clinical onset** using hourly ICU vitals and laboratory data. The system is trained on the **PhysioNet Computing in Cardiology (CinC) 2019 Challenge** dataset comprising **40,111 patients** and **1,543,363 hourly observations** from two hospitals.

Three models were developed and compared:

| Model | Role | CV AUROC |
|-------|------|:--------:|
| Logistic Regression | Baseline | 0.777 |
| XGBoost | Primary | **0.812** |
| LSTM | Sequence model | In progress |

All models were evaluated via **5-fold patient-level stratified cross-validation** to prevent data leakage across time steps within a single patient stay.

---

## 2. Data Source

**Dataset:** PhysioNet/Computing in Cardiology Challenge 2019
**License:** Creative Commons Attribution 4.0 (CC BY 4.0)

| Property | Value |
|----------|-------|
| Total patients | 40,111 |
| Hospital A patients | 20,229 |
| Hospital B patients | 19,882 |
| Total hourly observations | 1,543,363 |
| Columns per observation | 41 |
| Sepsis-positive patients | 2,922 (7.3%) |
| Sepsis-negative patients | 37,189 (92.7%) |
| Label definition | SepsisLabel = 1 at 6 hours before Sepsis-3 clinical onset |

### Variable Inventory (41 columns)

#### Vital Signs (8)

| Variable | Clinical Name | Unit |
|----------|--------------|------|
| HR | Heart rate | bpm |
| O2Sat | Pulse oximetry | % |
| Temp | Temperature | deg C |
| SBP | Systolic blood pressure | mmHg |
| MAP | Mean arterial pressure | mmHg |
| DBP | Diastolic blood pressure | mmHg |
| Resp | Respiration rate | breaths/min |
| EtCO2 | End tidal carbon dioxide | mmHg |

#### Laboratory Values (26)

| Variable | Clinical Name | Unit |
|----------|--------------|------|
| BaseExcess | Measure of excess bicarbonate | mmol/L |
| HCO3 | Bicarbonate | mmol/L |
| FiO2 | Fraction of inspired oxygen | % |
| pH | Arterial pH | - |
| PaCO2 | Partial pressure of CO2 (arterial) | mmHg |
| SaO2 | Oxygen saturation (arterial) | % |
| AST | Aspartate transaminase | IU/L |
| BUN | Blood urea nitrogen | mg/dL |
| Alkalinephos | Alkaline phosphatase | IU/L |
| Calcium | Calcium | mg/dL |
| Chloride | Chloride | mmol/L |
| Creatinine | Creatinine | mg/dL |
| Bilirubin_direct | Direct bilirubin | mg/dL |
| Glucose | Serum glucose | mg/dL |
| Lactate | Lactic acid | mmol/L |
| Magnesium | Magnesium | mmol/dL |
| Phosphate | Phosphate | mg/dL |
| Potassium | Potassium | mmol/L |
| Bilirubin_total | Total bilirubin | mg/dL |
| TroponinI | Troponin I | ng/mL |
| Hct | Hematocrit | % |
| Hgb | Hemoglobin | g/dL |
| PTT | Partial thromboplastin time | seconds |
| WBC | White blood cell count | 10^3/uL |
| Fibrinogen | Fibrinogen | mg/dL |
| Platelets | Platelets | 10^3/uL |

#### Demographics (6)

| Variable | Description |
|----------|------------|
| Age | Patient age (years) |
| Gender | Female (0) or Male (1) |
| Unit1 | Administrative identifier (MICU) |
| Unit2 | Administrative identifier (SICU) |
| HospAdmTime | Hours between hospital and ICU admission |
| ICULOS | ICU length of stay (hours) |

#### Label (1)

| Variable | Description |
|----------|------------|
| SepsisLabel | 0 = no sepsis, 1 = sepsis (onset - 6h through onset) |

---

## 3. Data Cleaning and Preprocessing

### The Missing Value Challenge

Lab values are drawn every few hours, not hourly. This creates **70-99% NaN rates** across laboratory columns. Vitals are recorded more frequently but still contain gaps.

### Cleaning Pipeline (per patient)

| Step | Operation | Rationale |
|------|-----------|-----------|
| 1 | **Missingness flags** | Binary indicator per vital and lab column: was this value measured at this hour? Captures testing frequency as a signal. |
| 2 | **Time since measurement** | Hours since each vital/lab was last measured for each patient. Encodes how "stale" a value is. |
| 3 | **Forward-fill** | Carry last known value forward within each patient stay. Clinicians treat the most recent value as current until a new one arrives. |
| 4 | **Remaining NaN fill** | Vitals filled with population median (zero is physiologically impossible — HR=0 means dead). Labs filled with 0.0 (missingness flags mark them as unknown). |
| 5 | **Final NaN fill** | Any remaining NaN in derived features (e.g., rolling stats at start of stay) filled with 0.0. |

### Excluded Features

Three columns were **dropped** as site-specific confounders:

| Dropped Column | Reason |
|----------------|--------|
| **Unit1** | Encodes hospital ward type (MICU), not clinical signal |
| **Unit2** | Encodes hospital ward type (SICU), not clinical signal |
| **HospAdmTime** | Hospital-specific administrative timing |

**Critical finding:** Before exclusion, `Unit2` was the **#1 most important feature** in the XGBoost model. It perfectly distinguished Hospital A from Hospital B, allowing the model to memorize site-specific baselines rather than learning generalizable clinical patterns. Dropping these three features was the **single biggest improvement to model generalization**.

---

## 4. Feature Engineering (161 features)

### Feature Groups

| Group | Count | Description |
|-------|:-----:|-------------|
| Raw vitals | 8 | HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2 |
| Raw labs | 26 | WBC, Hgb, Platelets, Lactate, Creatinine, Bilirubin, etc. |
| Demographics | 2 | Age, Gender (after dropping Unit1, Unit2, HospAdmTime) |
| Time | 1 | ICULOS (hours in ICU) |
| Missingness flags | 34 | `{col}_measured` — binary indicator per vital/lab per hour |
| Time since measurement | 34 | `{col}_hours_since` — hours since last measurement per patient |
| Rolling window stats | 48 | 6-hour rolling mean, min, max, std for key vitals and labs |
| Trend features | 8 | Hour-over-hour deltas for each vital sign |
| **Total** | **161** | |

### Rolling Window Features

Computed over a **6-hour trailing window** per patient for key vitals (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2) and key labs (Lactate, Creatinine, WBC, Platelets):

- Rolling mean
- Rolling min
- Rolling max
- Rolling standard deviation

### Trend Features

Hour-over-hour delta (current - previous) for each of the 8 vital signs. Captures acute deterioration.

### Feature Scaling

**StandardScaler** fit on training data only, then applied to both train and validation sets. This prevents information leakage from validation data into the scaling parameters.

---

## 5. Model Parameters

### Logistic Regression (Baseline)

| Parameter | Value |
|-----------|-------|
| Penalty | L2 |
| C (inverse regularization) | 1.0 |
| class_weight | balanced |
| max_iter | 2000 |
| solver | lbfgs |

### XGBoost (Primary Model)

Hyperparameters tuned via **RandomizedSearchCV** (30 iterations, 3-fold inner CV, scoring = roc_auc).

| Parameter | Search Space |
|-----------|-------------|
| max_depth | [3, 4, 5] |
| learning_rate | [0.01, 0.05, 0.1] |
| n_estimators | [100, 300, 500] |
| min_child_weight | [5, 10, 20] |
| gamma | [0.1, 0.5, 1.0] |
| reg_alpha | [0.1, 1.0] |
| reg_lambda | [1.0, 5.0] |
| subsample | [0.7, 0.8] |
| colsample_bytree | [0.7, 0.8] |
| scale_pos_weight | auto (neg/pos ratio, approximately 13:1) |

**Probability calibration:** Platt scaling via `CalibratedClassifierCV` (cv=3) applied after hyperparameter tuning to improve probability estimates.

### LSTM (Sequence Model)

| Parameter | Value |
|-----------|-------|
| Architecture | 2-layer LSTM (hidden_size=64) -> FC(64->32->1) -> Sigmoid |
| Dropout | 0.3 |
| Sequence length | 12 hours |
| Optimizer | Adam (lr=0.001) |
| Loss | Binary cross-entropy with class-weighted samples |
| Epochs | 20 |
| Batch size | 256 |
| Input features | Raw vitals + labs + demographics only (39 features, no engineered features) |

---

## 6. Training and Evaluation Strategy

### Primary Evaluation: 5-Fold Patient-Level Stratified Cross-Validation

All results are reported from 5-fold cross-validation on the **combined** Hospital A + Hospital B dataset.

- **StratifiedGroupKFold** ensures that all hourly observations for a single patient remain in the same fold (no temporal leakage).
- **Stratification** by SepsisLabel preserves the approximately 7.3% sepsis prevalence in each fold.
- **Feature scaling** is fit on the training fold only and applied to the validation fold (no information leakage).
- **XGBoost calibration** (Platt scaling) is performed within each fold using a held-out calibration set.

### Secondary Evaluation: Cross-Hospital Holdout (Stress Test)

- **Train** on Hospital A (20,229 patients), **validate** on Hospital B (19,882 patients).
- Tests worst-case generalization to an entirely unseen hospital with different patient populations, protocols, and documentation practices.
- Lower performance is expected and acceptable for a stress test.

### Class Imbalance Handling

The dataset has an approximately **13:1 negative-to-positive ratio** at the observation level.

| Model | Imbalance Strategy |
|-------|-------------------|
| XGBoost | `scale_pos_weight` = negative_count / positive_count |
| Logistic Regression | `class_weight='balanced'` (automatic inverse-frequency weighting) |
| LSTM | Per-sample weighted BCE loss (positive samples weighted by class ratio) |

---

## 7. Results

### Cross-Validation Results (Primary)

| Metric | XGBoost | Logistic Regression |
|--------|--------:|--------------------:|
| **AUROC** | **0.812** | 0.777 |
| Sensitivity (Recall) | ~60% | ~55% |
| Specificity | ~80% | ~78% |
| Gini Coefficient | 0.624 | 0.554 |

Per-fold AUROC scores and detailed metrics are available in `model_metrics.json` and the Streamlit dashboard.

### Cross-Hospital Holdout Results (Stress Test)

| Metric | XGBoost | Logistic Regression |
|--------|--------:|--------------------:|
| AUROC | ~0.65 | ~0.69 |

Lower performance is expected. This evaluation tests worst-case site generalization where training and validation hospitals have entirely different patient populations and clinical workflows.

Note: Logistic Regression slightly outperforms XGBoost in the cross-hospital test. This is consistent with simpler models being more robust to distribution shift.

### Overfit Analysis

Train vs. validation AUROC is compared per CV fold. The overfit gap table is available in the Streamlit dashboard. A small train-validation gap indicates the model generalizes well within the combined dataset.

---

## 8. Feature Importance

Three complementary methods were used to identify the most predictive features:

1. **Information Value (IV)** — statistical measure of univariate predictive power per feature
2. **XGBoost Gain** — percentage of total gain contributed by each feature across all tree splits
3. **SHAP Values** — game-theoretic attribution of each feature's marginal contribution per individual prediction

### Key Findings

| Feature | Why It Matters |
|---------|---------------|
| **Lactate** (and rolling stats) | Consistently top-ranked across all three methods. Clinically validated biomarker of tissue hypoperfusion in sepsis. |
| **Temperature** (rolling max) | Fever is a hallmark of the systemic inflammatory response in sepsis. |
| **Creatinine** (rolling max) | Elevated creatinine indicates acute kidney injury, a common organ failure in sepsis. |
| **Lab testing frequency** (hours_since features) | Sicker patients are tested more often. The pattern of clinical attention is itself a predictive signal. |
| **Heart rate** (trend, rolling) | Tachycardia is an early compensatory response to infection and hemodynamic instability. |
| **Respiratory rate** (trend, rolling) | Tachypnea is a component of the qSOFA sepsis screening score. |

### SHAP Visualizations

- Beeswarm plot: `data/processed/shap_summary.png`
- Bar plot: `data/processed/shap_bar.png`

---

## 9. Limitations

1. **Missing clinical variables.** The PhysioNet dataset lacks urine output, procalcitonin, CRP, INR, vasopressor dosing, ventilator status, and albumin. These are available in MIMIC-IV and are clinically relevant for sepsis prediction.

2. **ICU-only population.** All patients in the dataset are already admitted to the ICU. Predictions do not generalize to general ward, emergency department, or outpatient settings where earlier detection would have the greatest impact.

3. **No medication data.** The model cannot account for the effect of antibiotics, vasopressors, or fluids on vital signs and lab values, which may mask or accelerate sepsis progression.

4. **Severe class imbalance.** Only 7.3% of patients develop sepsis. While AUROC is robust to imbalance, precision at clinically useful thresholds remains low, meaning a deployed system would generate a significant number of false alarms.

5. **Cross-hospital performance gap.** Model performance drops substantially when tested on an entirely unseen hospital, indicating that some learned patterns are site-specific despite the exclusion of Unit1/Unit2/HospAdmTime.

---

## 10. Next Steps

1. **MIMIC-IV integration.** Incorporate the 7 missing clinical variables (urine output, procalcitonin, CRP, INR, vasopressors, ventilator status, albumin) plus medication timing data.

2. **Real patient data.** Apply the model architecture and feature engineering pipeline to institutional EHR data from a partner hospital system.

3. **Deep learning improvements.** Experiment with longer LSTM sequence lengths, attention mechanisms, and Transformer-based architectures that may better capture long-range temporal dependencies.

4. **Clinical validation.** Run the model in shadow mode alongside clinicians to measure alert fatigue, lead time, and clinical utility. Validate feature importance rankings with infectious disease and critical care domain experts.

5. **Prospective deployment study.** Integrate real-time predictions into an EHR alerting system and measure impact on time-to-antibiotics, mortality, and ICU length of stay in a controlled trial.

---

*Generated from the Sepsis Early Warning System PoC.*
*Dashboard: `streamlit run app.py` at localhost:8501.*
*Data: PhysioNet/CinC Challenge 2019 (CC BY 4.0).*
*Code: https://github.com/rushillshah/SepsisBot*
