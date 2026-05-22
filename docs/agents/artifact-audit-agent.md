# Artifact Audit Agent

Goal: make sure the files being presented match the code and current run.

Checklist:

- Compare `model_metrics.json["n_features"]` to `feature_names.pkl`.
- Verify `feature_names.pkl` matches `final_feature_list.csv`.
- Verify banned features are absent.
- Verify `retained_feature_correlation_audit.json` max correlation is below threshold.
- Verify `combined_ranking.csv`, `gain_ranking.csv`, `iv_ranking.csv`, and `shap_ranking.csv` have the same feature universe.
- Verify leadup IV uses the current final feature list.
- Flag missing SHAP PNGs as environment issue, not model issue.

Known environment issue:

Local Python 3.14 has a `pyexpat/libexpat` import failure that blocks matplotlib/SHAP plots. CSV rankings are still generated using XGBoost native contribution values.
