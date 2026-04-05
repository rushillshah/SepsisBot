# Sepsis Early Warning System — Proof of Concept

A predictive model that correlates patient vitals and lab values with sepsis onset, using the [PhysioNet/CinC 2019 Challenge](https://physionet.org/content/challenge-2019/1.0.0/) dataset. Built as a proof of concept to demonstrate feasibility before accessing real patient data.

## What It Does

- Ingests hourly ICU data (vitals, labs, demographics) for 40,111 patients across 2 hospitals
- Engineers 148 features per hour: raw values, rolling statistics, trends, and lab testing patterns
- Trains Logistic Regression (baseline) and XGBoost (primary) classifiers
- Predicts sepsis up to **6 hours before clinical onset** (Sepsis-3 criteria)
- Provides an interactive Streamlit dashboard explaining results for both clinical and technical audiences

## Current Results

| Metric | XGBoost | Logistic Regression |
|--------|--------:|--------------------:|
| AUROC | 0.624 | 0.685 |
| Sensitivity (Recall) | 56.1% | 55.0% |
| Specificity | 63.0% | 77.9% |
| Precision | 2.1% | 3.5% |
| Gini | 0.247 | 0.370 |

Validated on Hospital B (unseen during training). Below the 0.80 AUROC target due to cross-hospital distribution shift and overfitting — see the dashboard's "Limitations & Next Steps" page for the improvement roadmap.

## Quick Start

```bash
# 1. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download data (~40K files, takes ~30 min)
# Training Set A
mkdir -p data/raw/training_setA
curl -s "https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/" \
  | grep -oE 'p[0-9]+\.psv' | sort -u \
  | xargs -P 20 -I{} curl -s -o "data/raw/training_setA/{}" \
    "https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/{}"

# Training Set B
mkdir -p data/raw/training_setB
curl -s "https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/" \
  | grep -oE 'p[0-9]+\.psv' | sort -u \
  | xargs -P 20 -I{} curl -s -o "data/raw/training_setB/{}" \
    "https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/{}"

# 3. Run the pipeline (load, impute, train, evaluate)
python run_pipeline.py

# 4. Launch the dashboard
streamlit run app.py
```

The dashboard runs at `http://localhost:8501`.

## Project Structure

```
SepsisDataModel/
├── app.py                  # Streamlit dashboard (7 pages, dual-audience)
├── run_pipeline.py         # End-to-end pipeline script
├── requirements.txt        # Python dependencies
├── src/
│   ├── config.py           # Constants, feature lists, hyperparameters
│   ├── data_loader.py      # PSV parsing, parquet I/O
│   ├── imputation.py       # Forward-fill + missingness flags + median fill
│   ├── features.py         # Rolling windows, trends, feature matrix assembly
│   ├── train.py            # Logistic Regression + XGBoost training
│   └── evaluate.py         # AUROC, PR-AUC, ROC plots, patient-level analysis
├── tests/                  # 40 unit tests (pytest)
├── data/
│   ├── raw/                # CinC 2019 PSV files (gitignored)
│   └── processed/          # Parquet files + model metrics JSON (gitignored)
└── docs/superpowers/specs/ # Design spec
```

## Data Source

**PhysioNet/Computing in Cardiology Challenge 2019** — Early Prediction of Sepsis from Clinical Data

- 40,111 ICU patients (20,229 Hospital A + 19,882 Hospital B)
- 1,543,363 hourly observations
- 41 columns: 8 vitals, 26 labs, 6 demographics, 1 sepsis label
- Open access (CC BY 4.0)

### Variables Used

**Vitals:** HR, SpO2, Temperature, SBP, DBP, MAP, Respiratory Rate, EtCO2

**Labs:** WBC, Hemoglobin, Hematocrit, Platelets, PTT, Fibrinogen, FiO2, Bilirubin (total + direct), Lactate, Creatinine, BUN, Glucose, and 14 others

**Not available in this dataset (target for future work):** Urine output, Procalcitonin, CRP, INR, Vasopressor use, Ventilator status, Albumin

## Pipeline Details

### Imputation Strategy

Lab values have 70-99% missing rates (labs are drawn every few hours, not hourly). The imputation approach:

1. **Missingness flags** — binary indicator per lab: was it actually measured this hour?
2. **Time since measurement** — hours since each lab was last drawn
3. **Forward-fill** — carry last known value forward per patient
4. **Median fill** — remaining NaN (before first measurement) filled with column median

The missingness pattern itself is clinically informative — sicker patients get labs drawn more frequently.

### Feature Engineering (148 features)

- Raw vitals and labs (34)
- Missingness flags (26)
- Time since last measurement (26)
- 6-hour rolling statistics: mean, min, max, std (48)
- Hour-over-hour deltas for vitals (8)
- Demographics + ICU length of stay (6)

### Training

- **Split:** Train on Hospital A, validate on Hospital B (cross-hospital generalization)
- **Logistic Regression:** L2 regularization, balanced class weights
- **XGBoost:** RandomizedSearchCV (20 iterations, 5-fold stratified CV, scoring=roc_auc)
- **Class imbalance:** ~13:1 ratio handled via scale_pos_weight / balanced class weights

## Dashboard

The Streamlit dashboard has 7 pages with a toggle between clinical and technical explanations:

| Page | Content |
|------|---------|
| Executive Summary | Key numbers, class imbalance visualization, bottom line |
| The Data | Dataset breakdown, variable descriptions, missing data analysis |
| How the Model Works | Feature explanation, model parameters, training approach |
| Model Performance | Full metrics table (AUC, Gini, Recall, Precision, F1, PR-AUC), ROC curve, PR curve, confusion matrix |
| Patient Deep Dive | Per-patient vital sign and lab timelines with sepsis shading |
| What Drives Predictions | Color-coded feature importance by category |
| Limitations & Next Steps | Honest assessment and improvement roadmap |

## Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

40 tests covering data loading, imputation, and feature engineering.

## License

Code: MIT. Data: [PhysioNet CinC 2019](https://physionet.org/content/challenge-2019/1.0.0/) under CC BY 4.0.
