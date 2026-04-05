"""Imputation of missing values for the CinC 2019 sepsis dataset.

Strategy: forward-fill + missingness flags + time-since-measured.
Missingness metadata is computed BEFORE forward-fill so flags reflect
actual measurements, not imputed values.
"""

import numpy as np
import pandas as pd

from src.config import ALL_FEATURE_COLS, LAB_COLS


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary `{col}_measured` columns for each lab column.

    1 where the original value is present, 0 where NaN.
    """
    result = df.copy()
    for col in LAB_COLS:
        result[f"{col}_measured"] = df[col].notna().astype(np.int8)
    return result


def _hours_since_last_measurement(series: pd.Series) -> pd.Series:
    """Compute hours since last non-NaN value within a single patient.

    Returns -1 for rows before the first measurement.
    """
    is_measured = series.notna()

    # Assign NaN where not measured, cumulative count where measured
    # We use cumsum of the measured flag to create groups, then cumcount
    # within each group gives us the distance from the last measurement.
    group_id = is_measured.cumsum()

    # Before any measurement, group_id == 0 → sentinel
    before_first = group_id == 0

    # Within each group, the distance from the group start is the
    # position within the group (0-indexed). The first row of each
    # group is the measurement itself (distance 0), subsequent rows
    # increment by 1 hour each.
    hours_since = group_id.groupby(group_id).cumcount()

    hours_since = hours_since.astype(np.float32)
    hours_since[before_first] = -1.0

    return hours_since


def add_time_since_measured(df: pd.DataFrame) -> pd.DataFrame:
    """Add `{col}_hours_since` for each lab column, computed per patient.

    Counts hours since the lab was last measured. Uses -1 as a sentinel
    for hours before any measurement exists for that patient.
    """
    result = df.copy()
    for col in LAB_COLS:
        result[f"{col}_hours_since"] = (
            df.groupby("patient_id")[col]
            .transform(_hours_since_last_measurement)
        )
    return result


def forward_fill_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill NaN values in feature columns, grouped by patient."""
    result = df.copy()
    filled = (
        result.groupby("patient_id")[ALL_FEATURE_COLS]
        .ffill()
    )
    result[ALL_FEATURE_COLS] = filled
    return result


def fill_remaining_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Fill any NaN values remaining after forward-fill with column medians.

    Forward-fill leaves NaN for hours before a patient's first measurement.
    These are filled with the global column median so downstream models
    (e.g. LogisticRegression) that cannot handle NaN still work.
    """
    result = df.copy()
    for col in ALL_FEATURE_COLS:
        if result[col].isna().any():
            result[col] = result[col].fillna(result[col].median())
    return result


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full imputation pipeline.

    Order matters:
      1. Add missingness flags (reflects raw measurements)
      2. Add time-since-measured (reflects raw measurements)
      3. Forward-fill per patient (carries last known value forward)
      4. Fill remaining NaN with column medians
    """
    result = add_missingness_flags(df)
    result = add_time_since_measured(result)
    result = forward_fill_per_patient(result)
    result = fill_remaining_nans(result)
    return result
