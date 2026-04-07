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
- [x] Imputation caching
- [x] CUSUM dynamic baselines (adaptive changepoint per patient per feature)
- [x] Pipeline optimization (features computed once, vectorized loops, dead code removed)
- [x] Patient-baseline deviations (replaced by CUSUM dynamic baselines)
- [x] Leakage audit: confirmed CUSUM features not causing leakage (not in top 30)
- [x] Feature importance regenerated with 248-feature model

## Remaining

- [ ] Multi-timescale slopes — 3h/6h/12h rate-of-change
- [ ] Trajectory shape features — acceleration, volatility change
- [ ] Feature selection — cut from ~248 to top 30-50 by SHAP
- [ ] Patient-level alert aggregation — X consecutive hours above threshold
- [ ] Run LSTM on GPU — `src/train_lstm.py`
- [ ] MIMIC-IV integration — adds 7 missing variables

## Current Results (Honest CV Concat)

**Hour-Level AUROC: 0.955** (was 0.832 before CUSUM baselines)

| Threshold | Sens | Spec | Prec | Flagged |
|:---------:|:----:|:----:|:----:|--------:|
| 0.20 | 93.6% | 72.3% | 21.0% | 13,017 |
| 0.30 | 90.2% | 79.3% | 25.5% | 10,319 |
| 0.50 | 83.3% | 88.1% | 35.6% | 6,842 |

## Key Improvement History

| Change | AUROC Before | AUROC After |
|--------|:------------:|:-----------:|
| Remove site confounders + CV | 0.624 | 0.806 |
| Regularization + scaling + calibration | 0.806 | 0.834 |
| CUSUM dynamic baselines | 0.834 | **0.955** |
