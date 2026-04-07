# Session 3: Nachiket's Review & All Fixes (2026-04-07)

## Nachiket's Findings

Nachiket ran the model on his machine and found several critical issues:

1. **Confusion matrix fabricated** — estimated from hour-level rates, not actual predictions
2. **Metrics labeled patient-level but computed hour-level** — misleading dashboard
3. **Median imputation harmful** — inventing data for 70-96% missing labs
4. **ICULOS confounder** — still #2 in feature importance
5. **Threshold too low** — model very trigger-happy at Youden's J threshold
6. **No oversampling** — 13:1 class imbalance unaddressed at data level

## All Fixes Applied

### 1. Real Patient-Level Metrics
- `train_cv.py`: compute actual patient-level CM per fold (max prob per patient vs threshold)
- Dashboard: separate "Hour-Level Metrics" and "Patient-Level Metrics" sections
- Confusion matrix now from real predictions, not fabricated

### 2. Zero-Fill Imputation
- `imputation.py`: replaced median fill with zero fill
- Missingness flags + hours_since already tell the model values are unknown
- Zero is a neutral sentinel; median was inventing plausible but fake data

### 3. ICULOS Normalization
- Raw ICULOS added to EXCLUDED_FEATURES
- Replaced with `iculos_log` (log(ICULOS+1)) and `iculos_bucket` (0-6h/6-24h/24-72h/72h+)
- Removes site-specific admission timing confounder

### 4. Sepsis Oversampling
- Sepsis rows duplicated 18x during training (target 1:3 ratio)
- All original data preserved — no undersampling
- Helps model learn sepsis patterns from more balanced training signal

### 5. Threshold Analysis
- `threshold_analysis.py`: patient-level metrics at 11 thresholds
- Dashboard shows full tradeoff table + plot
- Sweet spot at threshold 0.10

## Results

**Before all fixes (session 1):**
- AUROC: 0.624 (cross-hospital), false alarm: 96.8%

**After all fixes:**
- **Hour AUROC: 0.818** (3-fold CV)
- **At threshold 0.10:** Sensitivity 87.6%, Specificity 93.4%, Precision 50.9%
- 5,030 patients flagged out of 40,111

## Evidence: Hour vs Patient Level Metrics

| Level | Sensitivity | Specificity | Precision |
|-------|:-----------:|:-----------:|:---------:|
| Hour (at Youden's J) | 72.6% | 75.5% | 5.2% |
| Patient (at Youden's J) | ~96% | ~24% | ~9% |
| Patient (at t=0.10) | **87.6%** | **93.4%** | **50.9%** |

The patient-level metrics at t=0.10 are the clinically relevant numbers.
