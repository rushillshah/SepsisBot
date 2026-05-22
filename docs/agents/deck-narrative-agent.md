# Deck Narrative Agent

Goal: translate results without overstating them.

For PPT creation, also use `docs/skills/sepsis-ppt-generation/SKILL.md`.

Current deck headline:

> Cleaned early-risk model: XGBoost AUROC 0.7988 after patient-level CV, leading-indicator filtering, and collinearity pruning.

Recommended slide structure:

1. Problem: sepsis detection needs earlier risk signals.
2. Dataset: PhysioNet ICU hourly vitals/labs, 40,111 patients.
3. Honest validation: patient-level CV, no patient in train and validation.
4. Leakage cleanup: removed site/time/intervention/static-marker features.
5. Feature selection: IV plus collinearity pruning at `|r| >= 0.8`.
6. Results: AUROC, PR-AUC, threshold tradeoffs, alert burden.
7. Clinical interpretation: trajectory signals, not raw symptoms.
8. Limitations: no severity adjustment; needs richer EHR validation.

Avoid:

- Dense feature lists without clinical framing.
- SHAP plots from stale runs.
- Any slide implying high precision at clinically acceptable alert rates.
