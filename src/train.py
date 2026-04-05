"""Model training for the sepsis prediction pipeline.

Provides a logistic regression baseline and an XGBoost primary model.
Training uses a hospital-based split: Hospital A for training, Hospital B
for validation — mimicking real-world generalization to a new site.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.config import CV_FOLDS, RANDOM_STATE, XGBOOST_PARAM_GRID
from src.features import build_feature_matrix


def split_by_hospital(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a raw DataFrame into Hospital A and Hospital B subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (pre-feature-extraction) DataFrame with a ``hospital`` column
        whose values are ``"A"`` or ``"B"``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(df_a, df_b)`` — disjoint subsets for each hospital.

    Raises
    ------
    KeyError
        If the ``hospital`` column is missing.
    ValueError
        If either hospital subset is empty.
    """
    if "hospital" not in df.columns:
        raise KeyError(
            "Column 'hospital' not found in DataFrame. "
            "Ensure the data was loaded with load_all_data()."
        )

    df_a = df.loc[df["hospital"] == "A"].reset_index(drop=True)
    df_b = df.loc[df["hospital"] == "B"].reset_index(drop=True)

    if df_a.empty:
        raise ValueError("Hospital A subset is empty after split.")
    if df_b.empty:
        raise ValueError("Hospital B subset is empty after split.")

    return df_a, df_b


def _compute_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Compute class weight ratio for imbalanced binary labels.

    Returns
    -------
    float
        count(negatives) / count(positives).

    Raises
    ------
    ValueError
        If there are zero positive samples.
    """
    y_arr = np.asarray(y)
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))

    if n_pos == 0:
        raise ValueError(
            "Cannot compute scale_pos_weight: no positive samples in y."
        )

    return n_neg / n_pos


def train_logistic_baseline(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> LogisticRegression:
    """Train a logistic regression baseline with L2 regularization.

    Uses ``class_weight='balanced'`` to handle the sepsis label imbalance
    without requiring manual weight tuning.

    Parameters
    ----------
    X_train : array-like
        Feature matrix.
    y_train : array-like
        Binary labels.

    Returns
    -------
    LogisticRegression
        Fitted model.
    """
    print("Training logistic regression baseline ...")

    model = LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    print("  Logistic regression training complete.")
    return model


def train_xgboost(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    scale_pos_weight: float | None = None,
) -> XGBClassifier:
    """Train an XGBoost classifier with sensible defaults.

    If ``scale_pos_weight`` is not provided, it is computed automatically
    from the class ratio in ``y_train``.

    Parameters
    ----------
    X_train : array-like
        Feature matrix.
    y_train : array-like
        Binary labels.
    scale_pos_weight : float | None
        Ratio of negative to positive samples. Computed from *y_train*
        when ``None``.

    Returns
    -------
    XGBClassifier
        Fitted model.
    """
    if scale_pos_weight is None:
        scale_pos_weight = _compute_scale_pos_weight(y_train)

    print(f"Training XGBoost (scale_pos_weight={scale_pos_weight:.2f}) ...")

    model = XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    print("  XGBoost training complete.")
    return model


def tune_xgboost(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    param_grid: dict | None = None,
) -> tuple[XGBClassifier, dict]:
    """Tune XGBoost via randomized search with stratified cross-validation.

    Uses ``RandomizedSearchCV`` rather than ``GridSearchCV`` because the
    full parameter grid is prohibitively large.

    Parameters
    ----------
    X_train : array-like
        Feature matrix.
    y_train : array-like
        Binary labels.
    param_grid : dict | None
        Hyperparameter distributions. Falls back to
        ``config.XGBOOST_PARAM_GRID`` when ``None``.

    Returns
    -------
    tuple[XGBClassifier, dict]
        ``(best_model, best_params)`` from the search.
    """
    if param_grid is None:
        param_grid = XGBOOST_PARAM_GRID

    scale_pos_weight = _compute_scale_pos_weight(y_train)

    print("Tuning XGBoost with RandomizedSearchCV ...")
    print(f"  n_iter=20, cv={CV_FOLDS}, scoring=roc_auc")

    base_model = XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_params: dict = search.best_params_
    best_model: XGBClassifier = search.best_estimator_

    print(f"  Best AUROC: {search.best_score_:.4f}")
    print(f"  Best params: {best_params}")

    return best_model, best_params


def train_pipeline(df: pd.DataFrame) -> dict:
    """Run the full training pipeline.

    Steps:
        1. Split raw data by hospital (A=train, B=validation).
        2. Build feature matrices for each split.
        3. Train a logistic regression baseline on Hospital A.
        4. Tune and train an XGBoost model on Hospital A.

    Parameters
    ----------
    df : pd.DataFrame
        Raw combined DataFrame with ``hospital`` and ``SepsisLabel``
        columns (output of ``data_loader.load_all_data``).

    Returns
    -------
    dict
        Keys: ``"logistic"``, ``"xgboost"``, ``"best_params"``,
        ``"X_train"``, ``"y_train"``, ``"X_val"``, ``"y_val"``.
    """
    print("=" * 60)
    print("SEPSIS PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Hospital-based split
    print("\n[1/4] Splitting data by hospital ...")
    df_a, df_b = split_by_hospital(df)
    print(f"  Hospital A: {len(df_a):,} rows")
    print(f"  Hospital B: {len(df_b):,} rows")

    # 2. Feature engineering
    print("\n[2/4] Building feature matrices ...")
    X_train, y_train = build_feature_matrix(df_a)
    X_val, y_val = build_feature_matrix(df_b)
    print(f"  Training features: {X_train.shape}")
    print(f"  Validation features: {X_val.shape}")

    # 3. Logistic baseline
    print("\n[3/4] Logistic regression baseline ...")
    logistic_model = train_logistic_baseline(X_train, y_train)

    # 4. Tuned XGBoost
    print("\n[4/4] XGBoost with hyperparameter tuning ...")
    xgb_model, best_params = tune_xgboost(X_train, y_train)

    print("\n" + "=" * 60)
    print("Training pipeline complete.")
    print("=" * 60)

    return {
        "logistic": logistic_model,
        "xgboost": xgb_model,
        "best_params": best_params,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }
