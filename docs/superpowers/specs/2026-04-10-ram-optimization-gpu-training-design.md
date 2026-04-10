# RAM Optimization & GPU Training

**Date:** 2026-04-10
**Problem:** Pipeline exceeds 32 GB RAM during `run_pipeline.py`, making it unrunnable on both a 32 GB custom PC and 32 GB Macs.
**Goal:** Reduce peak RAM to ~8-10 GB while adding GPU acceleration for XGBoost.

## Root Cause Analysis

Peak memory during CV training reaches ~31-34 GB due to:

| What | Size | When |
|------|------|------|
| `X_all` feature matrix (1.5M rows x 248 cols x 8 bytes float64) | ~3 GB | Always alive in `run_pipeline.py` |
| `imputed_df` not freed after feature matrix built | ~1.3 GB | Stays in scope |
| `raw_df` not freed after imputation | ~0.5 GB | Stays in scope |
| Feature engineering: 5 functions each `df.copy()` | +2-3 GB peak | During `build_feature_matrix` |
| Per CV fold: X_train + oversampled + scaled copies | ~7 GB | During `_train_fold` |
| `RandomizedSearchCV(n_jobs=-1)` workers, each copying data | ~16-20 GB | During hyperparameter search |

## Design

### 1. Dtype Downcasting (float64 -> float32)

**Files:** `src/data_loader.py`

Add `_downcast_floats(df)` helper that converts all float64 columns to float32. Call after `load_all_data()` and `load_processed()`. Halves all DataFrame memory. float32 has ~7 decimal digits of precision — more than sufficient for clinical vitals and lab values.

### 2. In-Place Feature Engineering (eliminate df.copy())

**Files:** `src/features.py`, `src/imputation.py`

Remove `result = df.copy()` from all functions, mutate input directly:

**features.py** — `add_rolling_features()`, `add_trend_features()`, `add_dynamic_baselines()`, `add_iculos_normalized()`, `add_clinical_scores()`, `create_early_label()`

**imputation.py** — `add_missingness_flags()`, `add_time_since_measured()`, `forward_fill_per_patient()`, `fill_remaining_nans()`

Pipeline already reassigns (`enriched = add_X(enriched)`) so callers are unaffected. Update docstrings and any tests that assert input immutability.

### 3. Free Intermediates in run_pipeline.py

**Files:** `run_pipeline.py`

```python
del raw_df; gc.collect()    # after imputation
del imputed_df; gc.collect() # after build_feature_matrix
```

### 4. Cap n_jobs for RandomizedSearchCV

**Files:** `src/config.py`, `src/train_cv.py`, `src/train.py`

Add `MAX_NJOBS = 2` to `config.py`. Apply in `_train_fold()` and `tune_xgboost()`. Two workers is the sweet spot — each extra worker copies the full training matrix.

### 5. Convert DataFrames to numpy before training

**Files:** `src/train_cv.py`

After `scale_features()`, convert to `.values` (numpy float32). Drops pandas index/column overhead.

### 6. XGBoost GPU Acceleration

**Files:** `src/config.py`, `src/train_cv.py`, `src/train.py`

Add `get_xgb_tree_params()` to `config.py`:
- CUDA available -> `{"tree_method": "hist", "device": "cuda"}`
- Otherwise -> `{"tree_method": "hist", "device": "cpu"}`

Detection uses `torch.cuda.is_available()` (torch already in deps). `tree_method="hist"` is faster than default on CPU too. XGBoost does not support MPS — Mac stays CPU, which is fine at this dataset size.

Apply in `XGBClassifier()` constructors in `train_cv.py` and `train.py`. Print device at pipeline start.

### 7. Drop large arrays from LSTM pipeline return

**Files:** `src/train_lstm.py`

Remove `X_train`, `y_train`, `X_val`, `y_val` from `train_lstm_pipeline()` return dict. Memory cleanup only — LSTM training is out of scope.

## Files Changed

| File | Changes |
|------|---------|
| `src/config.py` | Add `MAX_NJOBS`, `get_xgb_tree_params()` |
| `src/data_loader.py` | Add `_downcast_floats()`, call in `load_all_data()` and `load_processed()` |
| `src/imputation.py` | Remove `df.copy()` from 4 functions, mutate in place |
| `src/features.py` | Remove `df.copy()` from 6 functions, mutate in place |
| `src/train_cv.py` | Cap `n_jobs`, use `get_xgb_tree_params()`, convert to numpy |
| `src/train.py` | Cap `n_jobs`, use `get_xgb_tree_params()` |
| `src/train_lstm.py` | Remove large arrays from return dict |
| `run_pipeline.py` | `gc.collect()` after freeing intermediates, print device |
| `tests/` | Update tests that assert DataFrame immutability |

## Expected Memory Profile

| Stage | Before | After |
|-------|--------|-------|
| Raw data load | 0.5 GB (f64) | 0.25 GB (f32) |
| After imputation | 1.8 GB total | 0.65 GB (f32, raw freed) |
| Feature matrix | 4.8 GB (f64 + imputed alive) | 1.5 GB (f32, imputed freed) |
| CV fold training | ~30 GB peak (n_jobs=-1) | ~6-8 GB peak (n_jobs=2, f32) |
| XGBoost on GPU | N/A | VRAM used for tree building |

## What This Does NOT Change

- Model results / accuracy (float32 has sufficient precision)
- Pipeline structure (same steps, same order, same CV)
- LSTM training logic (out of scope, only return dict cleanup)
- Dashboard (reads from saved JSON/parquet, unaffected)
