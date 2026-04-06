# Session 3: Nachiket's Review & Findings (2026-04-07)

## What Happened

Nachiket ran the model on his machine using predict.py (inference only, no retraining). He reviewed the dashboard and dug into the data, finding several critical issues.

## Critical Findings

### 1. Confusion Matrix is Fabricated
The "What Happens in Practice" confusion matrix on the dashboard was NOT computed from actual model predictions. It was estimated by multiplying hour-level sensitivity/specificity from CV by patient counts. This produces materially wrong numbers.

### 2. Metrics Are Hour-Level, Not Patient-Level
The dashboard page titled "Patient-Level Cross-Validation Results" reports hour-level metrics (each hourly row is a separate TP/FP/TN/FN). This is misleading because:
- A sepsis patient alerting for 20 hours contributes 20 FP hours before the label flips, tanking precision
- Clinically what matters is: did the model catch the patient? Not: how many hours did it catch them for?

### 3. True Patient-Level Performance
When evaluated properly at the patient level (did model ever alert on this patient?):
- **Sensitivity: 95.7%** (catches almost every sepsis patient)
- **Specificity: 56.0%** (but also fires on 44% of non-sepsis patients)
- **Precision: 14.6%** (better than the 4.6% hour-level number)

This is actually a more useful clinical picture — the model is very sensitive but generates many false alarms.

### 4. Threshold Too Low
XGBoost probabilities max out at ~0.27, so the default threshold of 0.5 catches nothing. Optimal threshold from Youden's J is ~0.025, which is very trigger-happy.

### 5. Median Imputation Concerns
Labs with >50% pre-first-measurement missingness (Bilirubin_direct 96%, Fibrinogen 90%, TroponinI 84%) are filled with population median. This invents data for the majority of rows. XGBoost can handle NaN natively — median fill is actively harmful.

### 6. ICULOS Still a Top Feature
Despite removing Unit1/Unit2, ICULOS is still #2 in feature importance. It may be learning site-specific admission patterns rather than clinical signal.

## Nachiket's Recommendations
1. Oversample sepsis (keep all sepsis, downsample non-sepsis)
2. Configurable thresholds for precision-recall tradeoff
3. Patient-level alert aggregation (X consecutive hours above threshold)
4. Fix dashboard to show real patient-level metrics

## Data Exploration Notes
- 40,278 patients, 1.55M hourly rows
- 7.3% sepsis prevalence (patient-level), 1.8% (hour-level)
- Vitals mostly complete (HR 10%, MAP 12.5% missing)
- Labs 80-99% missing at hour level
- Glucose first measured at median 5h, Bilirubin_direct at 10h
- Sepsis patients: higher HR (89.5 vs 84.4), lower MAP (77.5 vs 83.2), lower DBP (59 vs 64)
