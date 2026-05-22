---
name: sepsis-feature-selection
description: Use when adding, dropping, ranking, pruning, or interpreting features in the SepsisDataModel project, especially IV, collinearity, SHAP, leading-indicator filtering, or clinical defensibility.
---

# Sepsis Feature Selection

Current workflow:

1. Build feature matrix.
2. Apply leading-indicator filter.
3. Apply collinearity pruning at `|Pearson r| >= 0.8`.
4. Retain higher-IV feature from each correlated pair.
5. In CV, compute IV on training fold only.
6. Save audit artifacts.

Required artifacts:

- `final_feature_list.csv`
- `collinearity_pruning_audit.csv`
- `retained_feature_correlation_audit.json`
- `feature_names.pkl`

Required checks:

- max retained correlation below `0.8`
- no `ICULOS`
- no `Unit1`, `Unit2`, `HospAdmTime`
- no raw `FiO2` in the leading-indicator model
- top features match current clinical story
