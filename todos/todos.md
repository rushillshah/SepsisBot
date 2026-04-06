# Todos — Sepsis Model

## Critical Bugs (from Nachiket's review 2026-04-07)

- [ ] **Fix "What Happens in Practice" confusion matrix** — currently FABRICATED by multiplying hour-level sensitivity/specificity by patient counts, not computed from actual predictions. True patient-level numbers are materially different: real TP=2,801 vs displayed 2,158, real FP=16,438 vs displayed 10,388, real Sensitivity=0.957 vs displayed 0.739, real Specificity=0.560 vs displayed 0.721
- [ ] **Fix metric reporting on "Patient-Level CV Results" dashboard page** — AUROC, Sensitivity, Specificity, Precision, F1, PR-AUC are all computed at hour level but page is labelled patient-level; recompute as true patient-level metrics (did the model alert on this patient before onset? yes/no)

## Model Improvements

- [ ] Oversample sepsis cases to address class imbalance (~13:1 negative:positive ratio)
- [ ] Precision-recall curve with configurable model output thresholds defining "sepsis positive" — allow tuning of what counts as a bad/alert prediction
- [ ] Patient-level alert aggregation — add persistence logic so an alert only fires after X consecutive hours above threshold, reducing alarm fatigue from hour-level score flipping
- [ ] Fix median imputation for high-missingness labs — columns like Bilirubin_direct (96% pre-first-measurement), Fibrinogen (90%), TroponinI (84%) should use sentinel fill (0 or -1) instead of median, or be dropped entirely; XGBoost should use native NaN handling rather than median fill which invents signal
- [ ] ICULOS normalization — replace raw ICULOS with log + buckets (still a top feature, possible confounder)
- [ ] Patient-baseline deviation features — compare vitals to patient's own first-6h baseline
- [ ] Multi-timescale slopes — 3h/6h/12h rate-of-change (currently only 1h deltas)
- [ ] Feature selection — cut from ~170 to top 30-50 by SHAP to reduce overfit gap (train 0.985 vs val 0.806)

## Key Finding: Hour-Level vs Patient-Level Metrics

The model scores every hour independently. Current evaluation metrics (AUROC, sensitivity, specificity, precision) are all hour-level. But clinically what matters is patient-level: did the model catch this patient before sepsis onset?

Evidence from Nachiket's analysis:

| | Dashboard (fabricated) | True Patient-Level | Hour-Level (actual) |
|---|---|---|---|
| TP | 2,158 | 2,801 | 23,267 |
| FN | 764 | 126 | 4,603 |
| FP | 10,388 | 16,438 | 268,840 |
| TN | 26,801 | 20,913 | 1,253,467 |
| Sensitivity | 0.739 | **0.957** | 0.835 |
| Specificity | 0.721 | **0.560** | 0.823 |

The true patient-level picture: model catches 95.7% of sepsis patients but also fires on 44% of non-sepsis patients. The dashboard was showing fabricated numbers that were neither hour-level nor patient-level.
