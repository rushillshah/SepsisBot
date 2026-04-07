# Todos — Sepsis Model

## Resolved

- [x] Confusion matrix fabricated → Real from CV concat predictions
- [x] Metrics labeled wrong (hour vs patient) → Clearly separated
- [x] Median imputation harmful → Zero-fill
- [x] Oversampling → 18x sepsis oversampling
- [x] ICULOS confounder → Excluded raw, replaced with log + bucket
- [x] Configurable thresholds → Threshold table at 11 points
- [x] Remove old cross-hospital pipeline → CV only
- [x] Final model scored training data → Removed, all metrics from CV concat
- [x] n_estimators capped at 150
- [x] Extended label window (early_label) for training
- [x] Imputation caching (skip if parquet exists)

## Next Priority: Patient-Baseline Deviation Features

**Spec:** `docs/superpowers/specs/2026-04-08-patient-baseline-deviations-design.md`

**Problem:** 29% of non-sepsis patients (10,695) overlap with sepsis patients because absolute vitals look similar. The model can't tell "sick but stable" from "sick and deteriorating."

**Solution:** For each vital, compute deviation from patient's own first-6h baseline. HR=95 with baseline_dev=+2 (stable) vs baseline_dev=+20 (deteriorating) gives the model the signal to separate them.

**Impact:** Should reduce false positives at threshold 0.30 from 10,695 toward ~6,000-7,000.

## Other Remaining

- [ ] Patient-level alert aggregation — X consecutive hours above threshold
- [ ] Multi-timescale slopes — 3h/6h/12h rate-of-change
- [ ] Trajectory shape features — acceleration, volatility change
- [ ] Feature selection — cut from ~170 to top 30-50 by SHAP
- [ ] Run LSTM on GPU — `src/train_lstm.py`
- [ ] MIMIC-IV integration — adds 7 missing variables

## Current Results (Honest CV Concat, with early_label training)

**Hour-Level AUROC: 0.831**

| Threshold | Sens | Spec | Prec | Early Det (1h before) |
|:---------:|:----:|:----:|:----:|:---------------------:|
| 0.20 | 90% | 56% | 14% | 52% |
| 0.30 | 81% | 71% | 18% | 46% |
| 0.50 | 59% | 90% | 31% | 32% |

## Key Analysis Finding

Model probability distributions:
- Sepsis patient max prob: median 0.57, 75th 0.73
- Non-sepsis patient max prob: median 0.17, 75th 0.33, 90th 0.50
- 29% of non-sepsis patients have max prob >= 0.30 (the false alarm problem)
- Pre-sepsis hours (7-12h before onset) avg prob: 0.403 vs non-sepsis avg: 0.152 (2.7x gap)
- The model CAN distinguish them on average, but the tails overlap
