# Session: Clean Feature Selection, Retraining, and Agent System (2026-05-23)

## Context

The model artifacts and docs had drifted:

- Old docs referenced AUROC values from earlier model states.
- Some SHAP/model artifacts still included `ICULOS`.
- The deck claimed IV/collinearity checks, but collinearity pruning was not actually enforced in training.
- Clinician critique highlighted that raw lactate/FiO2 can be symptoms, interventions, or baseline severity markers rather than genuine early-warning predictors.

## Changes Made

### Feature Selection

- Added `src/feature_selection.py`.
- Implemented `prune_collinear_by_iv`.
- Rule: for any feature pair with `|Pearson r| >= 0.8`, keep the higher-IV feature.
- Applied pruning inside each CV fold to avoid validation-label leakage.

### Training

- Reran full pipeline.
- Leading-indicator filter kept 158 features before collinearity pruning.
- CV fold feature counts after pruning:
  - Fold 1: 121
  - Fold 2: 119
  - Fold 3: 120
- Final saved model feature count: 119.

### Results

| Metric | LR | XGBoost |
|---|---:|---:|
| AUROC | 0.7480 | 0.7988 |
| PR-AUC | 0.0438 | 0.0656 |
| Precision | 0.0287 | 0.0348 |

At threshold `0.30`:

- Sensitivity: 91.4%
- Specificity: 52.5%
- Precision: 10.3%
- Flagged: 19,704 patients

### Artifact Audit

Verified absent:

- `ICULOS`
- `Unit1`
- `Unit2`
- `HospAdmTime`
- raw `FiO2`

Max retained absolute correlation: 0.773.

### Early-Warning Score Redundancy

Resolved:

- Kept: `early_warning_score_avg_6h`
- Dropped: `early_warning_score_max_6h`
- Dropped: `early_warning_score`

### Known Environment Issue

SHAP PNG generation failed because local Python 3.14 has a `pyexpat/libexpat` symbol error. `shap_ranking.csv` was generated using XGBoost native contribution values.

## Docs Added

- Rewrote `AGENTS.md`.
- Rewrote `docs/model_report.md`.
- Added project-local agent specs under `docs/agents/`.
- Added project-local skill specs under `docs/skills/`.

## Current Position

Use the clean 119-feature result for future claims:

> XGBoost AUROC 0.7988 after patient-level CV, leading-indicator filtering, and fold-local collinearity pruning.

Do not use old 0.955/0.854/0.812 values as the current headline.
