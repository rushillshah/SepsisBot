"""Feature engineering for the sepsis prediction pipeline.

Adds temporal features (rolling window statistics, hour-over-hour deltas)
to the imputed per-hour snapshot data, then assembles the final feature
matrix for model training.

Expected input: a DataFrame that has already passed through
``imputation.impute`` — forward-filled values with ``{col}_measured``
and ``{col}_hours_since`` columns present.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

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

# Deduplicate ROLLING_COLS (MAP appears in both VITAL_COLS and the extras).
_ROLLING_COLS_UNIQUE: list[str] = list(dict.fromkeys(ROLLING_COLS))

# Columns that are metadata / identifiers, never features.
_DROP_COLS = {"patient_id", "hospital", LABEL_COL} | set(EXCLUDED_FEATURES)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute backward-looking rolling window statistics per patient.

    For each column in ``ROLLING_COLS`` and each statistic in
    ``ROLLING_STATS``, a new column ``{col}_roll_{stat}`` is added.
    Rolling windows are anchored to the right (no future data leakage)
    and use ``min_periods=1`` so early hours still receive values.

    Parameters
    ----------
    df : pd.DataFrame
        Imputed hourly data with a ``patient_id`` column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with rolling-feature columns appended.
    """
    result = df.copy()
    grouped = result.groupby("patient_id")

    for col in _ROLLING_COLS_UNIQUE:
        rolling = grouped[col].rolling(
            window=ROLLING_WINDOW_HOURS,
            min_periods=1,
        )
        for stat in ROLLING_STATS:
            col_name = f"{col}_roll_{stat}"
            # .agg / getattr dispatches to the correct rolling method.
            computed = getattr(rolling, stat)()
            # rolling().stat() produces a MultiIndex (patient_id, row);
            # droplevel + sort restores alignment with the original index.
            values = computed.droplevel("patient_id").sort_index()
            # std with fewer than 2 observations returns NaN; a single
            # observation has zero variability, so 0.0 is correct.
            if stat == "std":
                values = values.fillna(0.0)
            result[col_name] = values

    return result


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hour-over-hour deltas for vital-sign columns per patient.

    For each column in ``VITAL_COLS``, a new column ``{col}_delta`` holds
    the difference from the previous hour within the same patient.  The
    first hour of each patient stay receives ``NaN`` (filled to ``0.0``
    so downstream models can consume the column directly).

    Parameters
    ----------
    df : pd.DataFrame
        Imputed hourly data with a ``patient_id`` column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with delta columns appended.
    """
    result = df.copy()

    for col in VITAL_COLS:
        result[f"{col}_delta"] = (
            result.groupby("patient_id")[col]
            .diff()
            .fillna(0.0)
        )

    return result


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of feature column names in the matrix.

    Includes every column in *df* except ``patient_id``, ``hospital``,
    and ``SepsisLabel``.  Useful for model interpretation and SHAP
    explanations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that has been through the full feature-engineering
        pipeline (rolling + trend features added).

    Returns
    -------
    list[str]
        Column names in the same order they appear in *df*.
    """
    return [c for c in df.columns if c not in _DROP_COLS]


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble the final feature matrix and label vector.

    Runs the full feature-engineering pipeline (rolling stats + trend
    deltas) and then separates features from labels.

    The returned ``X`` contains:
    * Raw vital and lab values (``ALL_FEATURE_COLS``)
    * Demographic columns (``DEMOGRAPHIC_COLS``)
    * ICU length-of-stay (``ICULOS``)
    * Missingness flags (``{col}_measured`` for each lab column)
    * Time-since-measured (``{col}_hours_since`` for each lab column)
    * Rolling statistics (``{col}_roll_{stat}``)
    * Trend deltas (``{col}_delta``)

    Parameters
    ----------
    df : pd.DataFrame
        Imputed hourly data with ``patient_id``, ``hospital``, and
        ``SepsisLabel`` columns present.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` where ``X`` is the feature matrix and ``y`` is the
        binary sepsis label.

    Raises
    ------
    KeyError
        If ``SepsisLabel`` is missing from *df*.
    """
    if LABEL_COL not in df.columns:
        raise KeyError(
            f"Label column '{LABEL_COL}' not found in DataFrame. "
            "Ensure the raw data has been loaded correctly."
        )

    enriched = add_rolling_features(df)
    enriched = add_trend_features(enriched)

    y = enriched[LABEL_COL].copy()
    X = enriched.drop(columns=[c for c in _DROP_COLS if c in enriched.columns])

    # Fill any remaining NaN in derived features (rolling stats on sparse
    # labs, hours_since sentinel values, etc.) so all models can consume X.
    X = X.fillna(0.0)

    return X, y


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale features using StandardScaler fit on training data only."""
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
