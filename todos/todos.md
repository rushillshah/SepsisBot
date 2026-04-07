# Todos — Sepsis Model

## Resolved (2026-04-07)

- [x] ~~Fix confusion matrix (fabricated)~~ → Real patient-level CM from actual predictions
- [x] ~~Fix metric labels (hour vs patient)~~ → Dashboard clearly separates both
- [x] ~~Fix median imputation~~ → Changed to zero-fill
- [x] ~~Oversample sepsis~~ → 18x oversampling during training
- [x] ~~ICULOS confounder~~ → Excluded raw, replaced with log + bucket
- [x] ~~Configurable thresholds~~ → Threshold analysis table + plot in dashboard
- [x] ~~Remove old cross-hospital pipeline~~ → CV-only evaluation

## Remaining

- [ ] Patient-level alert aggregation — add persistence logic (X consecutive hours above threshold) to reduce alarm fatigue from hour-level score flipping. Module built (`src/threshold_analysis.py:consecutive_hour_alerts`), not yet in dashboard.
- [ ] Patient-baseline deviation features — compare vitals to patient's own first-6h baseline
- [ ] Multi-timescale slopes — 3h/6h/12h rate-of-change (currently only 1h deltas)
- [ ] Feature selection — cut from ~170 to top 30-50 by SHAP to reduce overfit gap
- [ ] Run LSTM on GPU — `src/train_lstm.py`, needs proper GPU
- [ ] MIMIC-IV integration — adds 7 missing variables (urine output, procalcitonin, CRP, INR, vasopressors, vent status, albumin)

## Current Best Results

**XGBoost CV AUROC: 0.818** (3-fold, oversampled, ICULOS normalized)

At threshold 0.10 (patient-level):
- Sensitivity: 87.6%
- Specificity: 93.4%
- Precision: 50.9%
- 5,030 patients flagged out of 40,111
