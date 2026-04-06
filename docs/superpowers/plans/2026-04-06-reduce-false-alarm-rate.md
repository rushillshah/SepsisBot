# Reduce False Alarm Rate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the patient-level false alarm rate from 96.8% to <20% while maintaining ~60% sepsis detection, targeting AUROC >= 0.80 via patient-level cross-validation.

**Architecture:** Remove site-specific confounders (Unit1, Unit2, HospAdmTime) from features, switch from single-hospital training to patient-level stratified 5-fold CV on both hospitals, add stronger XGBoost regularization, scale features for logistic regression, and calibrate output probabilities via Platt scaling.

**Tech Stack:** Python, scikit-learn (StratifiedGroupKFold, CalibratedClassifierCV, StandardScaler), XGBoost, pandas, numpy. Existing modules: `src/config.py`, `src/features.py`, `src/train.py`, `src/evaluate.py`, `run_pipeline.py`, `app.py`.

**Spec:** `docs/superpowers/specs/2026-04-06-reduce-false-alarm-rate-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/config.py` | Modify | Add EXCLUDED_FEATURES, XGBOOST_PARAM_GRID_V2 |
| `src/features.py` | Modify | Drop excluded features, add scale_features() |
| `src/train_cv.py` | Create | Patient-level stratified CV, calibration, full pipeline |
| `tests/test_train_cv.py` | Create | Tests for CV splitting, calibration, pipeline |
| `run_pipeline.py` | Modify | Add CV pipeline step, update JSON with CV metrics |
| `app.py` | Modify | Show CV results as primary, cross-hospital as secondary |

---

### Task 1: Add Config Constants

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Add EXCLUDED_FEATURES and XGBOOST_PARAM_GRID_V2**

In `src/config.py`, add after the existing `XGBOOST_PARAM_GRID`:

```python
# ── Features to Exclude (site-specific confounders) ────────────────────────

EXCLUDED_FEATURES = ["Unit1", "Unit2", "HospAdmTime"]

# ── Improved Model Hyperparameters ─────────────────────────────────────────

XGBOOST_PARAM_GRID_V2 = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500],
    "min_child_weight": [5, 10, 20],
    "gamma": [0.1, 0.5, 1.0],
    "reg_alpha": [0.1, 1.0],
    "reg_lambda": [1.0, 5.0],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
}

CV_N_ITER = 30
```

- [ ] **Step 2: Commit**

```bash
git add src/config.py
git commit -m "feat: add EXCLUDED_FEATURES and stronger XGBoost param grid"
```

---

### Task 2: Update features.py — Drop Confounders and Add Scaling

**Files:**
- Modify: `src/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing tests for excluded features and scaling**

Append to `tests/test_features.py`:

```python
from src.config import EXCLUDED_FEATURES
from src.features import build_feature_matrix, scale_features


class TestExcludedFeatures:
    """Verify site-specific confounders are dropped from X."""

    def test_excluded_features_not_in_x(self, imputed_df):
        X, y = build_feature_matrix(imputed_df)
        for col in EXCLUDED_FEATURES:
            assert col not in X.columns, f"{col} should be excluded from X"


class TestScaleFeatures:
    """Verify StandardScaler integration."""

    def test_returns_scaled_arrays_and_scaler(self, imputed_df):
        X, y = build_feature_matrix(imputed_df)
        # Split in half to simulate train/val
        mid = len(X) // 2
        X_train, X_val = X.iloc[:mid], X.iloc[mid:]
        X_tr_s, X_val_s, scaler = scale_features(X_train, X_val)
        assert X_tr_s.shape == X_train.shape
        assert X_val_s.shape == X_val.shape
        assert scaler is not None

    def test_scaled_train_has_zero_mean(self, imputed_df):
        X, y = build_feature_matrix(imputed_df)
        mid = len(X) // 2
        X_tr_s, _, _ = scale_features(X.iloc[:mid], X.iloc[mid:])
        # Mean of each column should be near 0
        import numpy as np
        means = np.abs(X_tr_s.mean())
        assert (means < 0.1).all(), f"Scaled train means not near 0: {means[means >= 0.1].to_dict()}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_features.py::TestExcludedFeatures -v
pytest tests/test_features.py::TestScaleFeatures -v
```

Expected: FAIL — `EXCLUDED_FEATURES` not imported or `scale_features` not found.

- [ ] **Step 3: Update features.py — drop excluded features**

In `src/features.py`, add the import at the top:

```python
from src.config import (
    ALL_FEATURE_COLS,
    DEMOGRAPHIC_COLS,
    EXCLUDED_FEATURES,
    LABEL_COL,
    LAB_COLS,
    ROLLING_COLS,
    ROLLING_STATS,
    ROLLING_WINDOW_HOURS,
    TIME_COL,
    VITAL_COLS,
)
```

Update `_DROP_COLS`:

```python
_DROP_COLS = {"patient_id", "hospital", LABEL_COL} | set(EXCLUDED_FEATURES)
```

- [ ] **Step 4: Add scale_features function**

Add at the end of `src/features.py`:

```python
from sklearn.preprocessing import StandardScaler


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale features using StandardScaler fit on training data only.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Feature matrices (same columns).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, StandardScaler]
        Scaled train, scaled val, fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    return X_train_scaled, X_val_scaled, scaler
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_features.py -v
```

Expected: ALL PASS (including existing 14 tests + new ones).

- [ ] **Step 6: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: drop site confounders from features, add scale_features"
```

---

### Task 3: Build train_cv.py — Patient-Level Stratified Cross-Validation

**Files:**
- Create: `src/train_cv.py`
- Create: `tests/test_train_cv.py`

- [ ] **Step 1: Write failing test for patient_stratified_split**

Create `tests/test_train_cv.py`:

```python
"""Tests for patient-level stratified cross-validation."""

import numpy as np
import pandas as pd
import pytest

from src.config import LABEL_COL, VITAL_COLS, LAB_COLS, DEMOGRAPHIC_COLS, TIME_COL


@pytest.fixture
def small_cv_df():
    """Build a small DataFrame with 20 patients for CV testing."""
    np.random.seed(42)
    rows = []
    for i in range(20):
        pid = f"p{i:03d}"
        n_hours = np.random.randint(10, 30)
        has_sepsis = 1 if i < 3 else 0  # 3/20 = 15% sepsis
        for h in range(1, n_hours + 1):
            row = {"patient_id": pid, TIME_COL: h, LABEL_COL: 0, "hospital": "A" if i < 10 else "B"}
            for v in VITAL_COLS:
                row[v] = np.random.normal(80, 10)
            for lab in LAB_COLS:
                row[lab] = np.random.normal(5, 2) if np.random.random() > 0.8 else np.nan
            for d in DEMOGRAPHIC_COLS:
                row[d] = np.random.choice([0, 1])
            if has_sepsis and h >= n_hours - 5:
                row[LABEL_COL] = 1
            rows.append(row)
    return pd.DataFrame(rows)


class TestPatientStratifiedSplit:
    def test_returns_correct_number_of_folds(self, small_cv_df):
        from src.train_cv import patient_stratified_split
        folds = patient_stratified_split(small_cv_df, n_splits=3)
        assert len(folds) == 3

    def test_no_patient_in_both_train_and_val(self, small_cv_df):
        from src.train_cv import patient_stratified_split
        folds = patient_stratified_split(small_cv_df, n_splits=3)
        for train_idx, val_idx in folds:
            train_pids = set(small_cv_df.iloc[train_idx]["patient_id"])
            val_pids = set(small_cv_df.iloc[val_idx]["patient_id"])
            assert train_pids.isdisjoint(val_pids), "Patient appears in both train and val"

    def test_all_rows_covered(self, small_cv_df):
        from src.train_cv import patient_stratified_split
        folds = patient_stratified_split(small_cv_df, n_splits=3)
        all_val_idx = set()
        for _, val_idx in folds:
            all_val_idx.update(val_idx)
        assert all_val_idx == set(range(len(small_cv_df)))

    def test_sepsis_patients_in_every_fold(self, small_cv_df):
        from src.train_cv import patient_stratified_split
        folds = patient_stratified_split(small_cv_df, n_splits=3)
        for train_idx, val_idx in folds:
            train_labels = small_cv_df.iloc[train_idx].groupby("patient_id")[LABEL_COL].max()
            assert train_labels.sum() > 0, "No sepsis patients in training fold"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train_cv.py -v
```

Expected: FAIL — `train_cv` module not found.

- [ ] **Step 3: Implement train_cv.py**

Create `src/train_cv.py`:

```python
"""Patient-level stratified cross-validation for sepsis prediction.

Trains on both hospitals with patient-level fold assignment so no
patient's hours leak between train and validation.  Includes feature
scaling and probability calibration.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    CV_N_ITER,
    LABEL_COL,
    RANDOM_STATE,
    XGBOOST_PARAM_GRID_V2,
)
from src.evaluate import compute_metrics, find_optimal_threshold
from src.features import build_feature_matrix, scale_features


def patient_stratified_split(
    df: pd.DataFrame,
    n_splits: int = CV_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split data into folds at the patient level, stratified by sepsis outcome.

    All hours for a given patient land in the same fold.  Sepsis prevalence
    is approximately preserved across folds.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``patient_id`` and ``SepsisLabel`` columns.
    n_splits : int
        Number of CV folds.

    Returns
    -------
    list of (train_indices, val_indices) tuples
        Row-level integer indices into *df*.
    """
    patient_labels = df.groupby("patient_id")[LABEL_COL].max()
    patient_ids = patient_labels.index.values
    patient_y = patient_labels.values

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # StratifiedGroupKFold needs groups — here each row's group is its patient_id.
    groups = df["patient_id"].values
    y_rows = df[LABEL_COL].values

    folds = []
    for train_idx, val_idx in splitter.split(df, y_rows, groups):
        folds.append((train_idx, val_idx))

    return folds


def _train_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_num: int,
) -> dict:
    """Train and evaluate one CV fold.

    Returns a dict with metrics for both logistic regression and XGBoost.
    """
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    # Feature engineering
    X_train, y_train = build_feature_matrix(df_train)
    X_val, y_val = build_feature_matrix(df_val)

    # Scale features
    X_train_s, X_val_s, scaler = scale_features(X_train, X_val)

    # Class weight
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # ── Logistic Regression ──────────────────────────────────────────────
    lr_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    lr_model.fit(X_train_s, y_train)
    lr_prob = lr_model.predict_proba(X_val_s)[:, 1]
    lr_threshold = find_optimal_threshold(y_val, lr_prob)
    lr_metrics = compute_metrics(y_val, lr_prob, threshold=lr_threshold)

    # ── XGBoost with tuning ──────────────────────────────────────────────
    base_xgb = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=XGBOOST_PARAM_GRID_V2,
        n_iter=CV_N_ITER,
        scoring="roc_auc",
        cv=inner_cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train_s, y_train)
    xgb_model = search.best_estimator_

    # ── Calibrate XGBoost ────────────────────────────────────────────────
    calibrated_xgb = CalibratedClassifierCV(
        xgb_model, method="sigmoid", cv=3,
    )
    calibrated_xgb.fit(X_train_s, y_train)

    xgb_prob = calibrated_xgb.predict_proba(X_val_s)[:, 1]
    xgb_threshold = find_optimal_threshold(y_val, xgb_prob)
    xgb_metrics = compute_metrics(y_val, xgb_prob, threshold=xgb_threshold)

    print(
        f"  Fold {fold_num}: "
        f"LR AUROC={lr_metrics['auroc']:.4f}  "
        f"XGB AUROC={xgb_metrics['auroc']:.4f}  "
        f"XGB threshold={xgb_threshold:.3f}"
    )

    return {
        "lr_metrics": lr_metrics,
        "xgb_metrics": xgb_metrics,
        "xgb_best_params": search.best_params_,
        "xgb_threshold": xgb_threshold,
        "lr_threshold": lr_threshold,
        "y_val": np.asarray(y_val),
        "xgb_prob": xgb_prob,
        "lr_prob": lr_prob,
        "val_patient_ids": df_val["patient_id"].values if "patient_id" in df_val.columns else None,
    }


def cross_validate_pipeline(df: pd.DataFrame) -> dict:
    """Run patient-level stratified CV on both hospitals combined.

    Returns averaged metrics across folds, plus per-fold details.
    """
    print("=" * 60)
    print("PATIENT-LEVEL STRATIFIED CROSS-VALIDATION")
    print("=" * 60)

    folds = patient_stratified_split(df, n_splits=CV_FOLDS)
    fold_results = []

    for i, (train_idx, val_idx) in enumerate(folds, 1):
        result = _train_fold(df, train_idx, val_idx, fold_num=i)
        fold_results.append(result)

    # Average metrics across folds
    metric_keys = ["auroc", "pr_auc", "sensitivity", "specificity", "precision", "f1"]

    avg_xgb = {}
    avg_lr = {}
    for key in metric_keys:
        xgb_vals = [f["xgb_metrics"][key] for f in fold_results]
        lr_vals = [f["lr_metrics"][key] for f in fold_results]
        avg_xgb[key] = float(np.mean(xgb_vals))
        avg_xgb[f"{key}_std"] = float(np.std(xgb_vals))
        avg_lr[key] = float(np.mean(lr_vals))
        avg_lr[f"{key}_std"] = float(np.std(lr_vals))

    avg_xgb["gini"] = 2 * avg_xgb["auroc"] - 1
    avg_lr["gini"] = 2 * avg_lr["auroc"] - 1

    print("\n" + "-" * 60)
    print(f"  {'Metric':<20s} {'LR Mean':>10s} {'XGB Mean':>10s}")
    print(f"  {'-' * 42}")
    for key in metric_keys:
        print(
            f"  {key:<20s} "
            f"{avg_lr[key]:>10.4f} "
            f"{avg_xgb[key]:>10.4f}"
        )
    print(f"  {'gini':<20s} {avg_lr['gini']:>10.4f} {avg_xgb['gini']:>10.4f}")
    print("=" * 60)

    return {
        "avg_xgb_metrics": avg_xgb,
        "avg_lr_metrics": avg_lr,
        "fold_results": fold_results,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_train_cv.py -v
```

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/train_cv.py tests/test_train_cv.py
git commit -m "feat: patient-level stratified CV with calibration and scaling"
```

---

### Task 4: Update run_pipeline.py — Add CV Step

**Files:**
- Modify: `run_pipeline.py`

- [ ] **Step 1: Add CV import and step**

At the top of `run_pipeline.py`, add:

```python
from src.train_cv import cross_validate_pipeline
```

In the `run()` function, add a new step between the imputation step and the existing training step. Insert after `save_processed(imputed_df, "imputed_data")`:

```python
    # ── 3. Patient-level cross-validation (primary evaluation) ─────────────
    print("\n[Step 3/X] Running patient-level cross-validation ...")
    cv_results = cross_validate_pipeline(imputed_df)
```

Update the `_build_dashboard_json` function to include CV metrics. Add these keys to the returned dict:

```python
        # Cross-validation metrics (primary)
        "cv_xgb_auroc": cv_results["avg_xgb_metrics"]["auroc"],
        "cv_xgb_auroc_std": cv_results["avg_xgb_metrics"].get("auroc_std", 0),
        "cv_xgb_sensitivity": cv_results["avg_xgb_metrics"]["sensitivity"],
        "cv_xgb_specificity": cv_results["avg_xgb_metrics"]["specificity"],
        "cv_xgb_precision": cv_results["avg_xgb_metrics"]["precision"],
        "cv_xgb_f1": cv_results["avg_xgb_metrics"]["f1"],
        "cv_xgb_gini": cv_results["avg_xgb_metrics"]["gini"],
        "cv_lr_auroc": cv_results["avg_lr_metrics"]["auroc"],
        "cv_lr_sensitivity": cv_results["avg_lr_metrics"]["sensitivity"],
        "cv_lr_specificity": cv_results["avg_lr_metrics"]["specificity"],
        "cv_lr_precision": cv_results["avg_lr_metrics"]["precision"],
        "cv_lr_f1": cv_results["avg_lr_metrics"]["f1"],
        "cv_lr_gini": cv_results["avg_lr_metrics"]["gini"],
```

Pass `cv_results` through to `_build_dashboard_json` by updating the function signature and call site.

- [ ] **Step 2: Verify pipeline imports**

```bash
source .venv/bin/activate && python -c "from run_pipeline import main; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add run_pipeline.py
git commit -m "feat: add patient-level CV step to pipeline"
```

---

### Task 5: Update Dashboard — Show CV Results

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add CV metrics to the Model Performance page**

In the `page_model_performance()` function, after the existing metrics table, add a new section:

```python
    # ── Cross-Validation Results (Primary) ───────────────────────────────
    cv_xgb_auroc = metrics.get("cv_xgb_auroc")
    if cv_xgb_auroc is not None:
        st.markdown("### Patient-Level Cross-Validation Results (Primary)")
        st.caption("5-fold stratified CV on both hospitals combined — no patient in both train and val")

        cv_table = pd.DataFrame({
            "Metric": ["AUROC", "Sensitivity (Recall)", "Specificity", "Precision", "F1", "Gini"],
            "XGBoost (CV)": [
                _fmt(metrics.get("cv_xgb_auroc")),
                _fmt(metrics.get("cv_xgb_sensitivity")),
                _fmt(metrics.get("cv_xgb_specificity")),
                _fmt(metrics.get("cv_xgb_precision")),
                _fmt(metrics.get("cv_xgb_f1")),
                _fmt(metrics.get("cv_xgb_gini")),
            ],
            "Logistic Reg (CV)": [
                _fmt(metrics.get("cv_lr_auroc")),
                _fmt(metrics.get("cv_lr_sensitivity")),
                _fmt(metrics.get("cv_lr_specificity")),
                _fmt(metrics.get("cv_lr_precision")),
                _fmt(metrics.get("cv_lr_f1")),
                _fmt(metrics.get("cv_lr_gini")),
            ],
        })
        st.dataframe(cv_table, use_container_width=True, hide_index=True)

        explain(
            clinical=(
                "These are the <b>primary results</b> — the model was tested on patients it never "
                "saw during training, with both hospitals mixed in every fold. This is a much "
                "fairer test than the cross-hospital results below."
            ),
            technical=(
                f"5-fold patient-level stratified CV. XGBoost with Platt calibration. "
                f"AUROC {metrics.get('cv_xgb_auroc', 'N/A')} "
                f"(+/- {metrics.get('cv_xgb_auroc_std', 'N/A')}). "
                f"Features scaled via StandardScaler. Site confounders (Unit1, Unit2, HospAdmTime) removed."
            ),
            audience=audience,
        )

        st.markdown("### Cross-Hospital Holdout (Stress Test)")
        st.caption("Train on Hospital A only, validate on Hospital B — tests site generalization")
```

Move the existing metrics table under this "Cross-Hospital Holdout" header.

- [ ] **Step 2: Verify app compiles**

```bash
source .venv/bin/activate && python -c "import app" 2>&1 | grep -i error || echo "OK"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: show CV results as primary metrics in dashboard"
```

---

### Task 6: Run Full Pipeline and Verify

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

```bash
source .venv/bin/activate && pytest tests/ -v
```

Expected: ALL PASS (40 existing + 4 new = 44 tests).

- [ ] **Step 2: Run the pipeline**

```bash
source .venv/bin/activate && python run_pipeline.py
```

Expected output should show:
- CV fold metrics with AUROC > 0.75 per fold
- Calibrated XGBoost threshold between 0.3-0.7
- Improved cross-hospital AUROC (> 0.65)

- [ ] **Step 3: Launch dashboard and verify**

```bash
streamlit run app.py
```

Verify at `http://localhost:8501`:
- Model Performance page shows CV results as primary
- Cross-hospital results shown as secondary
- Metrics table includes all 3 models
- ROC curves render for all models

- [ ] **Step 4: Commit results and push**

```bash
git add -A
git commit -m "feat: complete false alarm rate reduction pipeline"
git push origin main
```
