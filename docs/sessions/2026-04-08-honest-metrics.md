# Session 4: Honest CV Metrics & Final Model Removal (2026-04-08)

## What Changed

### Removed Final Model Training
Previously: trained a "final model" on ALL data, then scored ALL data for threshold analysis. This was leaky — scoring training data produces inflated metrics.

Now: all metrics come from concatenated CV fold predictions. Each patient is scored exactly once, by a model that never saw them during training.

### Evidence of Inflation

| Metric (t=0.10) | Old (scoring training data) | Honest (CV concat) |
|---|---|---|
| Patient Sensitivity | 87.6% | 98.4% |
| Patient Specificity | 93.4% | 14.5% |
| Patient Precision | 50.9% | 8.3% |

The old numbers looked great because the model memorized its training data. The honest numbers show the model is very sensitive (catches almost everyone) but also very trigger-happy at low thresholds.

### How CV Concat Works
3-fold CV splits patients into 3 groups. Each fold:
- Trains on 2 groups (~67% of patients)
- Scores the held-out group (~33% of patients)

Concatenate all 3 held-out predictions → every patient scored once, never by their training model.

### Honest Threshold Analysis

| Threshold | Sensitivity | Specificity | Precision | Flagged |
|:---------:|:-----------:|:-----------:|:---------:|--------:|
| 0.10 | 98.4% | 14.5% | 8.3% | 34,688 |
| 0.20 | 91.2% | 53.8% | 13.4% | 19,855 |
| 0.25 | 87.5% | 61.7% | 15.2% | 16,791 |
| 0.30 | 83.6% | 68.6% | 17.3% | 14,127 |
| 0.50 | 63.6% | 87.5% | 28.5% | 6,519 |

### Other Changes
- n_estimators capped at 150 (was 500) → hour AUROC actually improved to 0.834
- val_patient_ids and val_labels now returned from each CV fold for concatenation
- Pipeline steps reduced from 5 to 4 (no final model training)
