"""Tests for src.train_cv: patient-level stratified CV splits."""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DEMOGRAPHIC_COLS,
    LAB_COLS,
    VITAL_COLS,
)
from src.train_cv import patient_stratified_split


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    """Build a synthetic DataFrame with 20 patients, 10 hourly rows each.

    3 of the 20 patients have sepsis (SepsisLabel=1 in their last 5 hours).
    All feature columns expected by build_feature_matrix are present so the
    split function, which only reads patient_id and SepsisLabel, works cleanly.
    """
    rng = np.random.default_rng(42)
    n_patients = 20
    hours_per_patient = 10
    sepsis_patient_ids = {f"p{i:03d}" for i in range(3)}  # p000, p001, p002

    rows: list[dict] = []

    for i in range(n_patients):
        pid = f"p{i:03d}"
        is_sepsis_patient = pid in sepsis_patient_ids
        hospital = "A" if i < 10 else "B"

        for hour in range(1, hours_per_patient + 1):
            # Sepsis patients flip the label in their last 5 hours.
            if is_sepsis_patient and hour > (hours_per_patient - 5):
                label = 1
            else:
                label = 0

            row: dict = {
                "patient_id": pid,
                "hospital": hospital,
                "ICULOS": hour,
                "SepsisLabel": label,
            }

            for col in VITAL_COLS:
                row[col] = float(rng.uniform(60.0, 120.0))

            for col in LAB_COLS:
                row[col] = float(rng.uniform(1.0, 10.0))
                row[f"{col}_measured"] = int(rng.integers(0, 2))
                row[f"{col}_hours_since"] = float(rng.uniform(0.0, 6.0))

            for col in DEMOGRAPHIC_COLS:
                row[col] = float(rng.uniform(20.0, 80.0))

            rows.append(row)

    return pd.DataFrame(rows)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestPatientStratifiedSplit:
    def test_returns_correct_number_of_folds(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        folds = patient_stratified_split(synthetic_df, n_splits=3)
        assert len(folds) == 3

    def test_no_patient_in_both_train_and_val(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Patient IDs must be completely disjoint between train and val."""
        folds = patient_stratified_split(synthetic_df, n_splits=3)

        for train_idx, val_idx in folds:
            train_patients = set(synthetic_df.iloc[train_idx]["patient_id"].unique())
            val_patients = set(synthetic_df.iloc[val_idx]["patient_id"].unique())
            overlap = train_patients & val_patients
            assert overlap == set(), (
                f"Patients appear in both train and val for the same fold: {overlap}"
            )

    def test_all_rows_covered(self, synthetic_df: pd.DataFrame) -> None:
        """The union of all val indices must equal the full row index set."""
        folds = patient_stratified_split(synthetic_df, n_splits=3)

        all_val_indices: set[int] = set()
        for _, val_idx in folds:
            all_val_indices.update(val_idx.tolist())

        expected = set(range(len(synthetic_df)))
        assert all_val_indices == expected, (
            f"Val indices do not cover all rows. "
            f"Missing {len(expected - all_val_indices)} rows."
        )

    def test_sepsis_patients_in_every_fold(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Every training fold must contain at least one sepsis patient."""
        folds = patient_stratified_split(synthetic_df, n_splits=3)

        for fold_num, (train_idx, _) in enumerate(folds, start=1):
            train_df = synthetic_df.iloc[train_idx]
            has_sepsis = (train_df["SepsisLabel"] == 1).any()
            assert has_sepsis, (
                f"Fold {fold_num} training set has no sepsis patients."
            )
