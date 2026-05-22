# Data Leakage Agent

Goal: aggressively look for inflated performance.

High-risk signals:

- `ICULOS`, length of stay, time since admission.
- Hospital/unit identifiers.
- Missingness/testing frequency features.
- Intervention variables such as raw `FiO2`.
- Post-onset rows mixed into early-warning evaluation.
- Any feature computed using future patient data.
- Global feature selection outside CV.

Current mitigations:

- `ICULOS`, `Unit1`, `Unit2`, `HospAdmTime` excluded.
- Leading-indicator filter drops clinician-action and static-marker groups.
- Early-onset sepsis patients with onset at or before 6 ICU hours excluded.
- Metrics from concatenated held-out CV predictions.
- Collinearity pruning applied fold-locally.

If AUROC jumps materially, run this agent before trusting it.
