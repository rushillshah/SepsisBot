# Feature Analysis & Selection — Design Spec

## Context

Two needs:
1. **Nachiket needs feature analysis deliverables** — Top features by IV, XGBoost gain, SHAP plot of top 50, plus precision/recall on development (CV) and holdout (cross-hospital) sets
2. **Model improvement** — Feature selection to reduce the 0.18 overfit gap (train 0.985 vs val 0.806) by pruning from 170 features to the most predictive subset

## Part 1: Feature Analysis Report for Nachiket

### Deliverables

**A. Top Features — Three Rankings**
Generate and save to `data/processed/feature_analysis/`:
- `iv_ranking.csv` — All features ranked by Information Value with IV strength labels
- `gain_ranking.csv` — All features ranked by XGBoost gain percentage
- `shap_ranking.csv` — All features ranked by mean |SHAP|
- `combined_ranking.csv` — Top 50 features with IV, gain%, mean |SHAP|, and average rank

**B. SHAP Plot of Top 50**
- `shap_top50_beeswarm.png` — SHAP beeswarm (summary) plot showing top 50 features
- `shap_top50_bar.png` — SHAP bar plot (mean |SHAP|) for top 50
- Use `shap.summary_plot(shap_values, X_sample, max_display=50)` with high-res output

**C. Precision & Recall on Development vs Holdout**
- **Development set**: 3-fold CV results — per-fold precision, recall, and averaged
- **Holdout set**: Cross-hospital (train A → validate B) precision and recall
- Save as `precision_recall_comparison.csv`:

| Set | Precision | Recall | AUROC | F1 |
|-----|-----------|--------|-------|----|
| CV Fold 1 | ... | ... | ... | ... |
| CV Fold 2 | ... | ... | ... | ... |
| CV Fold 3 | ... | ... | ... | ... |
| CV Mean | ... | ... | ... | ... |
| Cross-Hospital Holdout | ... | ... | ... | ... |

### Implementation
- New script: `scripts/generate_feature_report.py` — standalone script that reads existing model_metrics.json and model .pkl files, generates all the above without retraining
- Uses existing functions from `src/feature_importance.py`
- For the SHAP top-50 plot, re-run SHAP on the saved XGBoost model with 10K sample from the full dataset

## Part 2: Feature Selection

### After Nachiket reviews the analysis, proceed with selection:

1. **Rank features** by mean |SHAP| (most reliable importance measure for trees)
2. **Train CV at K = [20, 30, 40, 50, 70]** — for each K, use only top-K SHAP features, run 3-fold patient-level CV, record AUROC + overfit gap
3. **Pick optimal K** — best val AUROC with acceptable overfit gap (<0.12)
4. **Save `SELECTED_FEATURES`** to `src/config.py`
5. **Update `build_feature_matrix()`** to filter to SELECTED_FEATURES when set

### New file: `src/feature_selection.py`
- `run_feature_selection(df, k_values, shap_ranking)` — trains CV for each K, returns comparison DataFrame
- `select_optimal_k(results_df)` — picks K that maximizes val AUROC with gap < 0.12

### Modified files
- `src/config.py` — add `SELECTED_FEATURES` (initially empty list, populated after selection)
- `src/features.py` — `build_feature_matrix()` filters columns when `SELECTED_FEATURES` is non-empty

## Verification

1. All CSVs and PNGs generated in `data/processed/feature_analysis/`
2. SHAP top-50 plot renders clearly with 50 features visible
3. Precision/recall table shows both CV and holdout numbers
4. Feature selection comparison table shows AUROC and gap for each K
5. After selection, pipeline runs with reduced feature set
6. Val AUROC >= 0.80 with selected features
7. Overfit gap < 0.12 (down from 0.18)
