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
- [x] n_estimators too high → Capped at 150, AUROC improved to 0.834

## Remaining

- [ ] Patient-level alert aggregation — X consecutive hours above threshold to reduce flip-flopping. Module built (`src/threshold_analysis.py:consecutive_hour_alerts`), not in dashboard.
- [ ] Patient-baseline deviation features — compare vitals to patient's own first-6h baseline
- [ ] Multi-timescale slopes — 3h/6h/12h rate-of-change (currently only 1h deltas)
- [ ] Feature selection — cut from ~170 to top 30-50 by SHAP to reduce overfit gap
- [ ] Run LSTM on GPU — `src/train_lstm.py`
- [ ] MIMIC-IV integration — adds 7 missing variables

## Current Best Results (Honest, from CV Concat)

**Hour-Level AUROC: 0.834**

Patient-level at threshold 0.50: sens 63.6%, spec 87.5%, prec 28.5%
Patient-level at threshold 0.25: sens 87.5%, spec 61.7%, prec 15.2%
