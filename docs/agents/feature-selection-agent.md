# Feature Selection Agent

Goal: keep the model clinically defensible and technically clean.

Current policy:

- First apply leading-indicator filtering.
- Then apply collinearity pruning.
- Correlation threshold: `|Pearson r| >= 0.8`.
- For each correlated pair, keep the feature with higher IV.
- In CV, compute IV using training rows only.

Artifacts to check:

- `data/processed/feature_analysis/final_feature_list.csv`
- `data/processed/feature_analysis/collinearity_pruning_audit.csv`
- `data/processed/feature_analysis/retained_feature_correlation_audit.json`
- `data/processed/models/feature_names.pkl`

Required sanity checks:

- `ICULOS` absent.
- `Unit1`, `Unit2`, `HospAdmTime` absent.
- raw `FiO2` absent in the leading-indicator model.
- max retained absolute correlation is below `0.8`.
- suspicious correlated feature families have only one representative unless there is a documented exception.

Known resolved issue:

- `early_warning_score_avg_6h`, `early_warning_score_max_6h`, and `early_warning_score` were highly correlated.
- Current model keeps only `early_warning_score_avg_6h`.
