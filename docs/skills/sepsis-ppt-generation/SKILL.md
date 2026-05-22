---
name: sepsis-ppt-generation
description: Use when creating or updating PowerPoint decks for the SepsisDataModel project, especially clinician-facing, technical-clinician, advisor, investor, or presentation-review decks about model results, feature exclusions, leakage cleanup, validation metrics, and clinical claims.
---

# Sepsis PPT Generation

Use this workflow for any deck in this project.

## Current Story

The deck must use the cleaned model state:

- XGBoost AUROC: `0.7988`
- XGBoost PR-AUC: `0.0656`
- final features: `119`
- max retained absolute correlation: `0.773`
- collinearity threshold: `|Pearson r| >= 0.8`
- validation: 3-fold patient-level stratified CV

Do not headline stale values such as `0.955`, `0.854`, or `0.812` unless explicitly presenting historical model evolution.

## Required Slides

Every serious deck must cover:

1. The cleaned claim.
2. Feature exclusions.
3. Patient-level CV validation.
4. AUROC and PR-AUC.
5. Patient-level threshold tradeoff.
6. Current top features.
7. Clinical limitations.
8. Next validation step.

## Feature Exclusion Must Be Visible

Explicitly show that the model excludes:

- `ICULOS`
- `Unit1`, `Unit2`, `HospAdmTime`
- raw `FiO2`
- testing-frequency / missingness features
- static marker features where configured
- lower-IV correlated duplicates

Also show the resolved early-warning-score family:

- kept `early_warning_score_avg_6h`
- dropped `early_warning_score_max_6h`
- dropped `early_warning_score`

## Audience Modes

### Doctor

Emphasize:

- honest claim
- no severity-independent claim
- feature exclusion
- clinical usefulness vs alert burden
- why richer EHR data is needed

Avoid:

- dense CV implementation details
- overuse of SHAP/IV jargon
- claiming deployment readiness

### Technical Doctor

Add:

- fold-local feature selection
- IV/correlation pruning mechanics
- artifact audit
- threshold table
- feature-rank evidence
- remaining leakage/severity risks

## Data Sources

Read these before building:

- `data/processed/model_metrics.json`
- `data/processed/feature_analysis/combined_ranking.csv`
- `data/processed/feature_analysis/collinearity_pruning_audit.csv`
- `data/processed/feature_analysis/final_feature_list.csv`
- `data/processed/feature_analysis/retained_feature_correlation_audit.json`
- `docs/model_report.md`

## QA

Before delivery:

- Render a contact sheet.
- Check the cover slide for text overlap.
- Check tables remain readable.
- Check no stale metric appears.
- Check no stale feature appears in a current-model claim.
- Verify the final PPTX opens and has the expected slide count.
