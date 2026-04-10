"""Tests for src.imputation: missingness flags, time-since, forward-fill."""

import numpy as np
import pandas as pd
import pytest

from src.config import ALL_FEATURE_COLS, LAB_COLS, VITAL_COLS
from src.imputation import (
    add_missingness_flags,
    add_time_since_measured,
    forward_fill_per_patient,
    impute,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_synthetic_patients() -> pd.DataFrame:
    """Build a small DataFrame with 2 patients, ~10 rows each.

    Vital columns are fully populated; select lab columns contain NaN
    gaps at controlled positions so we can assert exact imputation
    behaviour.
    """
    rows: list[dict] = []
    rng = np.random.default_rng(42)

    for pid in ("p001", "p002"):
        for hour in range(1, 11):
            row: dict = {"patient_id": pid, "ICULOS": hour, "SepsisLabel": 0}

            # Vitals: mostly present, but controlled NaN gaps for testing.
            #   - First 2 hours: NaN (patient not yet on monitor)
            #   - Hours 3-10: measured
            for col in VITAL_COLS:
                if hour <= 2:
                    row[col] = np.nan
                else:
                    row[col] = rng.uniform(60.0, 100.0)

            # Labs: introduce controlled NaN pattern.
            #   - First 3 hours: all NaN (no labs drawn yet)
            #   - Hour 4: measured
            #   - Hours 5-6: NaN
            #   - Hour 7: measured
            #   - Hours 8-10: NaN
            for col in LAB_COLS:
                if hour in (4, 7):
                    row[col] = rng.uniform(1.0, 10.0)
                else:
                    row[col] = np.nan

            # Demographics (constant per patient)
            row["Age"] = 65.0 if pid == "p001" else 55.0
            row["Gender"] = 1 if pid == "p001" else 0
            row["Unit1"] = 1
            row["Unit2"] = 0
            row["HospAdmTime"] = -24.0
            row["hospital"] = "A"

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return _build_synthetic_patients()


# ---------------------------------------------------------------------------
# Tests: add_missingness_flags
# ---------------------------------------------------------------------------


class TestMissingnessFlags:
    def test_flag_columns_exist_for_labs(self, raw_df: pd.DataFrame) -> None:
        result = add_missingness_flags(raw_df)
        for col in LAB_COLS:
            assert f"{col}_measured" in result.columns

    def test_flag_columns_exist_for_vitals(self, raw_df: pd.DataFrame) -> None:
        result = add_missingness_flags(raw_df)
        for col in VITAL_COLS:
            assert f"{col}_measured" in result.columns

    def test_lab_flag_is_one_where_present(self, raw_df: pd.DataFrame) -> None:
        result = add_missingness_flags(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in LAB_COLS:
            flag = f"{col}_measured"
            # Hour 4 (index 3) should be measured
            assert p1.iloc[3][flag] == 1
            # Hour 7 (index 6) should be measured
            assert p1.iloc[6][flag] == 1

    def test_lab_flag_is_zero_where_nan(self, raw_df: pd.DataFrame) -> None:
        result = add_missingness_flags(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in LAB_COLS:
            flag = f"{col}_measured"
            # Hours 1-3 (indices 0-2) should be missing
            assert p1.iloc[0][flag] == 0
            assert p1.iloc[1][flag] == 0
            assert p1.iloc[2][flag] == 0

    def test_vital_flag_is_zero_where_nan(self, raw_df: pd.DataFrame) -> None:
        result = add_missingness_flags(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in VITAL_COLS:
            flag = f"{col}_measured"
            # Hours 1-2 (indices 0-1) vitals are NaN
            assert p1.iloc[0][flag] == 0
            assert p1.iloc[1][flag] == 0

    def test_vital_flag_is_one_where_present(self, raw_df: pd.DataFrame) -> None:
        result = add_missingness_flags(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in VITAL_COLS:
            flag = f"{col}_measured"
            # Hour 3+ (index 2+) vitals are measured
            assert p1.iloc[2][flag] == 1
            assert p1.iloc[5][flag] == 1

    def test_does_not_mutate_input(self, raw_df: pd.DataFrame) -> None:
        original_cols = list(raw_df.columns)
        _ = add_missingness_flags(raw_df)
        assert list(raw_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Tests: add_time_since_measured
# ---------------------------------------------------------------------------


class TestTimeSinceMeasured:
    def test_hours_since_columns_exist_for_labs(self, raw_df: pd.DataFrame) -> None:
        result = add_time_since_measured(raw_df)
        for col in LAB_COLS:
            assert f"{col}_hours_since" in result.columns

    def test_hours_since_columns_exist_for_vitals(self, raw_df: pd.DataFrame) -> None:
        result = add_time_since_measured(raw_df)
        for col in VITAL_COLS:
            assert f"{col}_hours_since" in result.columns

    def test_lab_sentinel_before_first_measurement(self, raw_df: pd.DataFrame) -> None:
        result = add_time_since_measured(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in LAB_COLS:
            hs = f"{col}_hours_since"
            # Hours 1-3 are before any measurement -> -1
            assert p1.iloc[0][hs] == -1.0
            assert p1.iloc[1][hs] == -1.0
            assert p1.iloc[2][hs] == -1.0

    def test_vital_sentinel_before_first_measurement(self, raw_df: pd.DataFrame) -> None:
        result = add_time_since_measured(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in VITAL_COLS:
            hs = f"{col}_hours_since"
            # Hours 1-2 are before any vital measurement -> -1
            assert p1.iloc[0][hs] == -1.0
            assert p1.iloc[1][hs] == -1.0

    def test_vital_zero_at_first_measurement(self, raw_df: pd.DataFrame) -> None:
        result = add_time_since_measured(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in VITAL_COLS:
            hs = f"{col}_hours_since"
            # Hour 3 (index 2): first vital measurement -> 0
            assert p1.iloc[2][hs] == 0.0

    def test_lab_zero_at_measurement_hour(self, raw_df: pd.DataFrame) -> None:
        result = add_time_since_measured(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in LAB_COLS:
            hs = f"{col}_hours_since"
            # Hour 4 (index 3): measurement itself -> 0
            assert p1.iloc[3][hs] == 0.0
            # Hour 7 (index 6): second measurement -> 0
            assert p1.iloc[6][hs] == 0.0

    def test_lab_counts_increment_after_measurement(self, raw_df: pd.DataFrame) -> None:
        result = add_time_since_measured(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in LAB_COLS:
            hs = f"{col}_hours_since"
            # After first measurement at hour 4: hours 5,6 are 1,2 since
            assert p1.iloc[4][hs] == 1.0
            assert p1.iloc[5][hs] == 2.0
            # After second measurement at hour 7: hours 8,9,10 are 1,2,3
            assert p1.iloc[7][hs] == 1.0
            assert p1.iloc[8][hs] == 2.0
            assert p1.iloc[9][hs] == 3.0

    def test_patients_are_independent(self, raw_df: pd.DataFrame) -> None:
        """Time-since counters should reset between patients."""
        result = add_time_since_measured(raw_df)
        p2 = result[result["patient_id"] == "p002"]

        for col in LAB_COLS:
            hs = f"{col}_hours_since"
            # Patient 2 also has NaN for hours 1-3 -> sentinel -1
            assert p2.iloc[0][hs] == -1.0
            assert p2.iloc[1][hs] == -1.0
            assert p2.iloc[2][hs] == -1.0

        for col in VITAL_COLS:
            hs = f"{col}_hours_since"
            # Patient 2 also has NaN for hours 1-2 -> sentinel -1
            assert p2.iloc[0][hs] == -1.0
            assert p2.iloc[1][hs] == -1.0


# ---------------------------------------------------------------------------
# Tests: forward_fill_per_patient
# ---------------------------------------------------------------------------


class TestForwardFill:
    def test_nans_filled_after_measurement(self, raw_df: pd.DataFrame) -> None:
        result = forward_fill_per_patient(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in LAB_COLS:
            measured_val = raw_df[raw_df["patient_id"] == "p001"].iloc[3][col]
            # Hours 5-6 (indices 4-5) should be forward-filled from hour 4
            assert p1.iloc[4][col] == measured_val
            assert p1.iloc[5][col] == measured_val

    def test_nans_remain_before_first_measurement(
        self, raw_df: pd.DataFrame
    ) -> None:
        result = forward_fill_per_patient(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in LAB_COLS:
            # Hours 1-3 (indices 0-2) have no prior value -> still NaN
            assert pd.isna(p1.iloc[0][col])
            assert pd.isna(p1.iloc[1][col])
            assert pd.isna(p1.iloc[2][col])

    def test_no_cross_patient_leakage(self, raw_df: pd.DataFrame) -> None:
        """Patient 2's pre-measurement rows must not inherit from patient 1."""
        result = forward_fill_per_patient(raw_df)
        p2 = result[result["patient_id"] == "p002"]

        for col in LAB_COLS:
            assert pd.isna(p2.iloc[0][col])
            assert pd.isna(p2.iloc[1][col])
            assert pd.isna(p2.iloc[2][col])

    def test_does_not_mutate_input(self, raw_df: pd.DataFrame) -> None:
        original_values = raw_df[LAB_COLS].copy()
        _ = forward_fill_per_patient(raw_df)
        pd.testing.assert_frame_equal(raw_df[LAB_COLS], original_values)


# ---------------------------------------------------------------------------
# Tests: impute (full pipeline)
# ---------------------------------------------------------------------------


class TestImpute:
    def test_produces_expected_columns(self, raw_df: pd.DataFrame) -> None:
        result = impute(raw_df)

        # Original columns still present
        for col in ALL_FEATURE_COLS:
            assert col in result.columns

        # Missingness flags added for both labs and vitals
        for col in LAB_COLS:
            assert f"{col}_measured" in result.columns
            assert f"{col}_hours_since" in result.columns
        for col in VITAL_COLS:
            assert f"{col}_measured" in result.columns
            assert f"{col}_hours_since" in result.columns

    def test_lab_flags_reflect_original_missingness(
        self, raw_df: pd.DataFrame
    ) -> None:
        """Flags must be computed BEFORE forward-fill.

        After imputation, hour 5 has a forward-filled lab value, but
        the measured flag should still be 0.
        """
        result = impute(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in LAB_COLS:
            # Hour 5 (index 4): value present after ffill, but flag is 0
            assert not pd.isna(p1.iloc[4][col])
            assert p1.iloc[4][f"{col}_measured"] == 0

            # Hour 4 (index 3): actually measured, flag is 1
            assert p1.iloc[3][f"{col}_measured"] == 1

    def test_vital_flags_reflect_original_missingness(
        self, raw_df: pd.DataFrame
    ) -> None:
        """Vital flags must reflect raw presence, not forward-filled values."""
        result = impute(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in VITAL_COLS:
            # Hours 1-2 (indices 0-1): originally NaN, flag should be 0
            assert p1.iloc[0][f"{col}_measured"] == 0
            assert p1.iloc[1][f"{col}_measured"] == 0
            # Hour 3 (index 2): measured, flag should be 1
            assert p1.iloc[2][f"{col}_measured"] == 1

    def test_vitals_filled_with_median_not_zero(
        self, raw_df: pd.DataFrame
    ) -> None:
        """Vitals before first measurement should get population median, not 0."""
        result = impute(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in VITAL_COLS:
            val = p1.iloc[0][col]
            # Must not be NaN
            assert not pd.isna(val), f"{col} still NaN after imputation"
            # Must not be zero (physiologically impossible for vitals)
            assert val != 0.0, f"{col} filled with 0.0 instead of median"
            # Should be the population median of measured values
            measured_vals = raw_df[col].dropna()
            if len(measured_vals) > 0:
                expected_median = measured_vals.median()
                assert val == pytest.approx(expected_median), (
                    f"{col} expected median {expected_median}, got {val}"
                )

    def test_labs_still_filled_with_zero(
        self, raw_df: pd.DataFrame
    ) -> None:
        """Labs before first measurement should still get 0.0 (regression guard)."""
        result = impute(raw_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in LAB_COLS:
            # Hours 1-3 (indices 0-2): before first lab measurement
            assert p1.iloc[0][col] == 0.0, (
                f"{col} should be 0.0 before first measurement"
            )

    def test_row_count_unchanged(self, raw_df: pd.DataFrame) -> None:
        result = impute(raw_df)
        assert len(result) == len(raw_df)
