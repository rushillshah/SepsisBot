# Patient-Baseline Deviation Features — Design Spec

## Problem

29% of non-sepsis patients (10,695) have max model probability >= 0.30, overlapping with sepsis patients. These are sick ICU patients whose absolute vitals look similar to pre-sepsis patients. The model can't distinguish "sick but stable" from "sick and deteriorating" because it only sees absolute values and population-level rolling stats.

## Evidence

```
Sepsis max prob:     25th=0.36, 50th=0.57, 75th=0.73
Non-sepsis max prob: 25th=0.13, 50th=0.17, 75th=0.33, 90th=0.50
```

The top 29% of non-sepsis patients overlap with the bottom 50% of sepsis patients.

## Solution

For each vital sign, compute the patient's own baseline (mean of first 6 hours) and track deviation from it at every subsequent hour.

### New Features

For 6 vitals (HR, Resp, SBP, MAP, O2Sat, Temp):
- `{vital}_baseline_dev` = current value - mean(first 6 hours for THIS patient)

These go through the existing rolling window pipeline, adding:
- `{vital}_baseline_dev_roll_mean` (6h)
- `{vital}_baseline_dev_roll_std` (6h)
- `{vital}_baseline_dev_roll_min` (6h)
- `{vital}_baseline_dev_roll_max` (6h)

Total: 6 raw + 24 rolling = 30 new features

### Why It Helps

```
Non-sepsis FP (sick but stable):  HR=95, HR_baseline_dev=+2, dev_roll_std=3
Pre-sepsis TP (deteriorating):    HR=95, HR_baseline_dev=+20, dev_roll_std=12
```

Same absolute HR. Completely different trajectory relative to patient's own baseline.

### Implementation

1. `src/config.py`: Add `BASELINE_VITALS = ["HR", "Resp", "SBP", "MAP", "O2Sat", "Temp"]`, add deviation cols to `ROLLING_COLS`
2. `src/features.py`: Add `add_baseline_deviations(df)` — groupby patient_id, compute mean of first 6 rows per vital, subtract from each row
3. Call in `build_feature_matrix()` before `add_rolling_features()`
4. Retrain (~10 min with cached imputed data)

### Expected Impact

- Directly reduces the 10,695 non-sepsis false positives
- "Sick but stable" patients get low deviation → lower probability
- "Sick and deteriorating" patients get high deviation → model can distinguish them
- Patient-level specificity at threshold 0.30 should improve from 71% toward 80%+
