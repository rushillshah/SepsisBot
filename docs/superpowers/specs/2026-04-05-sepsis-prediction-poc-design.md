# Sepsis Prediction Proof of Concept — Design Spec

## Context

We are building a predictive model that correlates patient vitals and lab values with sepsis onset. The goal is to demonstrate a working proof of concept using open-source data, which will justify access to actual patient data for a production system.

The core question: given a patient's hourly vitals and periodic lab draws, can we predict sepsis up to 6 hours before clinical onset?

## Data Source

**PhysioNet/CinC Challenge 2019** (Early Prediction of Sepsis from Clinical Data)

- **Patients:** 40,336 ICU patients across 2 hospital systems
- **Format:** PSV (pipe-separated values), one file per patient, hourly rows
- **License:** CC BY 4.0 (open access, no credentialing required)
- **Label:** `SepsisLabel` column — binary, flips to `1` at 6 hours before clinical sepsis onset (Sepsis-3 criteria)
- **URL:** https://physionet.org/content/challenge-2019/1.0.0/

### Variable Coverage

**Available (15 of 22 target variables):**

| Variable | CinC 2019 Column | Sparsity |
|---|---|---|
| Heart rate | HR | Low |
| Respiratory rate | Resp | Low |
| Systolic BP | SBP | Low |
| Diastolic BP | DBP | Low |
| Mean arterial pressure | MAP | Low |
| SpO2 | O2Sat | Low |
| Temperature | Temp | Low |
| EtCO2 | EtCO2 | Moderate |
| WBC count | WBC | High (~90%+ NaN) |
| Hemoglobin | Hgb | High |
| Hematocrit | Hct | High |
| Platelets | Platelets | High |
| PTT | PTT | High |
| Fibrinogen | Fibrinogen | High |
| FiO2 | FiO2 | High |
| Bilirubin (total) | Bilirubin_total | High |
| Bilirubin (direct) | Bilirubin_direct | High |

**Not available (7 variables):** Urine output, procalcitonin, CRP, INR, vasopressor/inotropic use, explicit ventilator status, albumin. These are accepted limitations for the PoC — MIMIC-IV can fill these gaps in a future iteration.

### Additional Columns Used

- **Demographics:** Age, Gender, Unit1, Unit2, HospAdmTime
- **Time:** ICULOS (hours since ICU admission)
- **Additional labs (included as features):** BaseExcess, HCO3, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Glucose, Lactate, Magnesium, Phosphate, Potassium, TroponinI — these are not in the original target variable list but are clinically relevant to sepsis and available in the dataset, so we include them as model features

## Data Pipeline

### Ingestion

1. Download training sets A and B from PhysioNet
2. Parse all `.psv` files into a single pandas DataFrame
3. Add `patient_id` column (from filename) and `hospital` column (A or B)
4. Store as parquet for fast subsequent loading

### Imputation Strategy: Forward-Fill + Missingness Flags

Lab values are drawn every few hours, not hourly. The imputation approach:

1. **Forward-fill:** Carry the last known lab value forward until a new measurement arrives
2. **Missingness flags:** Add a binary column per lab variable (e.g., `wbc_measured`) indicating whether the value was actually measured at that hour
3. **Time since measurement:** Add a column per lab variable counting hours since the last actual measurement

The missingness pattern is clinically informative — sicker patients get labs drawn more frequently.

### Feature Engineering

For each hourly row, compute:

- **Raw values:** Forward-filled vitals and labs (~34 features)
- **Missingness flags:** Binary indicators per lab variable (~17 features)
- **Time since last measurement:** Hours since each lab was last drawn (~17 features)
- **Rolling window statistics (6-hour):** Mean, min, max, std for vitals and key labs (~60 features)
- **Trend features:** Hour-over-hour deltas for vitals (~8 features)
- **Demographics:** Age, gender (static per patient) (~2 features)
- **ICULOS:** Hours in ICU (~1 feature)

**Estimated total: ~140 features per row**

### Class Imbalance

Sepsis-positive hours are approximately 2-5% of all hourly rows. Mitigation:

- XGBoost `scale_pos_weight` parameter (ratio of negative to positive samples)
- Stratified train/test splits (preserve class ratio)
- AUROC as primary metric (robust to imbalance)

## Model Architecture

### Baseline: Logistic Regression (L2)

- Interpretable coefficients show which features matter
- Serves as a "is this problem learnable?" sanity check
- Scikit-learn `LogisticRegression` with `class_weight='balanced'`

### Primary: XGBoost

- Gradient boosted decision trees — strong on tabular data with missing values
- Hyperparameters tuned via 5-fold stratified cross-validation:
  - `max_depth`: [3, 5, 7]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `n_estimators`: [100, 300, 500]
  - `scale_pos_weight`: computed from class ratio
  - `subsample`: [0.7, 0.8]
  - `colsample_bytree`: [0.7, 0.8]
- Feature importance analysis (gain-based and SHAP values)

### Evaluation

- **Primary metric:** AUROC
- **Secondary metrics:** Precision-Recall AUC, sensitivity at 80% specificity, F1 score
- **Patient-level analysis:** For each sepsis patient, what was the earliest hour the model flagged risk above threshold?
- **Train/validation split:** Train on Hospital A, validate on Hospital B (tests cross-hospital generalization)
- **Additional:** 5-fold cross-validation on combined data for robust performance estimate

### Target Performance

- CinC 2019 winning entries achieved ~0.85+ AUROC
- A credible PoC should aim for AUROC >= 0.80

## Project Structure

```
SepsisDataModel/
├── data/
│   ├── raw/                    # Downloaded CinC 2019 PSV files (gitignored)
│   │   ├── training_setA/
│   │   └── training_setB/
│   └── processed/              # Processed parquet files (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py               # Feature lists, hyperparams, file paths
│   ├── data_loader.py          # Parse PSV files, combine into DataFrame
│   ├── imputation.py           # Forward-fill + missingness flags
│   ├── features.py             # Rolling windows, trends, feature engineering
│   ├── train.py                # Model training + cross-validation
│   └── evaluate.py             # Metrics, plots, patient-level analysis
├── notebooks/
│   └── exploration.ipynb       # EDA + results visualization
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_imputation.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Dependencies

```
pandas>=2.0
numpy>=1.24
pyarrow           # parquet support
scikit-learn>=1.3
xgboost>=2.0
shap              # model interpretability
matplotlib>=3.7
seaborn>=0.12
jupyter
pytest
```

## Known Limitations

1. **7 missing variables:** Urine output, procalcitonin, CRP, INR, vasopressors, vent status, albumin are not in CinC 2019
2. **Lab sparsity:** 90%+ NaN rates for lab values — imputation introduces assumptions
3. **ICU-only data:** Predictions only apply to ICU patients, not general ward or ED
4. **Single prediction task:** Binary classification at each hour, not a multi-step clinical decision system
5. **No medication data:** Cannot account for treatment effects on vitals/labs

## Future Work (Post-PoC)

1. **MIMIC-IV integration:** Fill the 7 missing variables, add medication data
2. **Deep learning comparison:** LSTM/GRU for temporal sequence modeling
3. **Real patient data:** Apply model architecture to actual clinical data once PoC demonstrates viability
4. **Clinical validation:** Work with clinicians to validate feature importance and prediction thresholds

## Verification Plan

1. **Data integrity:** Verify parsed DataFrame matches expected patient counts and column names from CinC 2019 documentation
2. **Feature sanity:** Check feature distributions, ensure no data leakage (no future values in rolling windows)
3. **Model training:** Confirm convergence, check for overfitting (train vs. validation AUROC gap)
4. **Baseline comparison:** Logistic regression AUROC should be meaningfully above random (0.5)
5. **XGBoost performance:** Target AUROC >= 0.80 on Hospital B holdout
6. **Feature importance:** Top features should be clinically plausible (e.g., lactate, HR, MAP)
7. **End-to-end test:** Load raw PSV -> process -> predict -> evaluate in a single pipeline run
