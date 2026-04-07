# Dynamic Baseline via CUSUM Change Detection — Design Spec

## Problem

The current static baseline (mean of first 6 hours) is arbitrary:
- Patient deteriorating from hour 1 → bad baseline
- Patient stable for 20 hours then crashes → wasted stable data, only uses first 6h
- Doesn't adapt per-feature: HR might stabilize at hour 3, but lactate isn't measured until hour 8

## Solution

Use CUSUM (Cumulative Sum) change detection to find the **changepoint** per patient per feature — the hour where the feature starts deviating from the patient's stable state. Everything before the changepoint is baseline.

## CUSUM Algorithm

For patient P, feature F:
1. Initial estimate: `μ₀ = mean(first 6 non-NaN values)`, `σ₀ = std(first 6 non-NaN values)`
2. If σ₀ == 0 (constant values), set σ₀ = 1 to avoid division by zero
3. Initialize `S_high = 0`, `S_low = 0` (detect both upward and downward shifts)
4. For each subsequent value: 
   - `S_high = max(0, S_high + (value - μ₀) / σ₀ - k)`
   - `S_low = max(0, S_low - (value - μ₀) / σ₀ - k)`
5. When `S_high > h` or `S_low > h` → changepoint detected at this hour
6. If no changepoint → entire stay is baseline

Parameters: `k = 0.5` (slack), `h = 4.0` (threshold). These are standard CUSUM defaults.

## Features Added (11 features × 3 = 33 new, replacing 6 old static deviations)

For each of 11 features (HR, Resp, SBP, MAP, O2Sat, Temp, Lactate, WBC, Creatinine, Platelets, BUN):

- `{feature}_dynamic_dev`: current value - mean(pre-changepoint values)
- `{feature}_changepoint_hour`: ICULOS at which change was detected (or max ICULOS if none)
- `{feature}_baseline_length`: number of hours/measurements in stable baseline

The `_dynamic_dev` columns go into ROLLING_COLS so they get 6h rolling stats (mean/min/max/std) automatically.

Total: 33 raw + 44 rolling (11 dev cols × 4 stats) = 77 new features, minus 30 old static baseline features = net +47

## Handling Sparse Labs

For labs (Lactate, WBC, Creatinine, Platelets, BUN):
- CUSUM runs on actual measurements only (skip NaN/forward-filled hours)
- Baseline = mean of first N actual measurements before changepoint
- `_baseline_length` counts actual measurements, not hours

## Implementation

### Config changes (`src/config.py`):
```python
DYNAMIC_BASELINE_FEATURES = ["HR", "Resp", "SBP", "MAP", "O2Sat", "Temp",
                              "Lactate", "WBC", "Creatinine", "Platelets", "BUN"]
CUSUM_SLACK = 0.5
CUSUM_THRESHOLD = 4.0
```

Replace `BASELINE_VITALS` and `BASELINE_DEVIATION_COLS` with dynamic equivalents.

### Feature function (`src/features.py`):
- Replace `add_baseline_deviations()` with `add_dynamic_baselines(df)`
- Vectorized where possible: compute CUSUM per patient-feature using groupby + custom transform
- For each feature:
  1. Group by patient_id
  2. Sort by ICULOS
  3. Run CUSUM to find changepoint
  4. Compute mean of pre-changepoint values
  5. Deviation = current - baseline_mean

### Update `build_feature_matrix()`:
- Replace `add_baseline_deviations(enriched)` with `add_dynamic_baselines(enriched)`

## Verification

1. Stable patients (no sepsis, no deterioration) → changepoint at last hour, deviation ~0
2. Patients with sudden deterioration → changepoint near the spike, high deviation after
3. Labs: changepoint based on actual measurements, not forward-filled values
4. No NaN in output features (fill with 0 for pre-first-measurement)
5. Pipeline runs, AUROC maintained or improved
6. Feature importance: dynamic_dev features should rank higher than old static_dev features
