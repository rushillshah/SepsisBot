"""Tests for src.features: rolling stats, trend deltas, feature matrix."""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    ALL_FEATURE_COLS,
    EXCLUDED_FEATURES,
    LAB_COLS,
    ROLLING_COLS,
    ROLLING_STATS,
    ROLLING_WINDOW_HOURS,
    VITAL_COLS,
)
from src.features import (
    add_rolling_features,
    add_trend_features,
    build_feature_matrix,
    get_feature_names,
    scale_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Deduplicate the same way the module does.
_ROLLING_COLS_UNIQUE: list[str] = list(dict.fromkeys(ROLLING_COLS))


def _build_imputed_df() -> pd.DataFrame:
    """Build a small imputed DataFrame with 2 patients, 10 rows each.

    All feature columns are populated (no NaN) to simulate the output of
    ``imputation.impute`` with forward-fill applied. Lab missingness flags
    and hours_since columns are included.
    """
    rows: list[dict] = []
    rng = np.random.default_rng(99)

    for pid in ("p001", "p002"):
        for hour in range(1, 11):
            row: dict = {
                "patient_id": pid,
                "hospital": "A",
                "ICULOS": hour,
                "SepsisLabel": 1 if hour >= 8 else 0,
            }

            # Vital signs: deterministic linear ramp for testable deltas
            for i, col in enumerate(VITAL_COLS):
                row[col] = 70.0 + i * 2.0 + hour * 1.0

            # Lab values: constant per patient for simpler assertions
            for col in LAB_COLS:
                row[col] = rng.uniform(1.0, 10.0)

            # Demographics
            row["Age"] = 60.0
            row["Gender"] = 1
            row["Unit1"] = 1
            row["Unit2"] = 0
            row["HospAdmTime"] = -12.0

            # Missingness metadata (simulate: labs measured at hours 1,4,7)
            for col in LAB_COLS:
                row[f"{col}_measured"] = 1 if hour in (1, 4, 7) else 0
                if hour in (1, 4, 7):
                    row[f"{col}_hours_since"] = 0.0
                elif hour < 1:
                    row[f"{col}_hours_since"] = -1.0
                else:
                    # Find distance from nearest prior measurement
                    for mh in (7, 4, 1):
                        if hour > mh:
                            row[f"{col}_hours_since"] = float(hour - mh)
                            break

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def imputed_df() -> pd.DataFrame:
    return _build_imputed_df()


# ---------------------------------------------------------------------------
# Tests: add_rolling_features
# ---------------------------------------------------------------------------


class TestRollingFeatures:
    def test_rolling_columns_created(self, imputed_df: pd.DataFrame) -> None:
        result = add_rolling_features(imputed_df)
        for col in _ROLLING_COLS_UNIQUE:
            for stat in ROLLING_STATS:
                expected_name = f"{col}_roll_{stat}"
                assert expected_name in result.columns, (
                    f"Missing rolling column: {expected_name}"
                )

    def test_no_future_data_leakage(self, imputed_df: pd.DataFrame) -> None:
        """Rolling mean at hour 1 should equal the hour-1 value itself.

        If future data leaked in, the mean would differ.
        """
        result = add_rolling_features(imputed_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in _ROLLING_COLS_UNIQUE:
            first_val = p1.iloc[0][col]
            first_roll_mean = p1.iloc[0][f"{col}_roll_mean"]
            assert first_roll_mean == pytest.approx(first_val), (
                f"Rolling mean at hour 1 for {col} should equal the raw value "
                f"({first_val}), got {first_roll_mean}"
            )

    def test_rolling_window_size_respected(
        self, imputed_df: pd.DataFrame
    ) -> None:
        """By the 7th hour the rolling window should span exactly 6 values."""
        result = add_rolling_features(imputed_df)
        p1 = result[result["patient_id"] == "p001"]

        col = VITAL_COLS[0]  # HR
        # Hours 2-7 (indices 1-6): at index 6 we have 7 hours of data,
        # but window=6 so only the last 6 are included.
        expected_vals = [p1.iloc[i][col] for i in range(1, 7)]
        expected_mean = sum(expected_vals) / len(expected_vals)
        actual_mean = p1.iloc[6][f"{col}_roll_mean"]
        assert actual_mean == pytest.approx(expected_mean, rel=1e-6)

    def test_single_observation_std_is_zero(
        self, imputed_df: pd.DataFrame
    ) -> None:
        """std of a single observation should be 0.0, not NaN."""
        result = add_rolling_features(imputed_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in _ROLLING_COLS_UNIQUE:
            std_val = p1.iloc[0][f"{col}_roll_std"]
            assert std_val == 0.0, (
                f"Expected 0.0 for single-obs std of {col}, got {std_val}"
            )
            assert not pd.isna(std_val)

    def test_does_not_mutate_input(self, imputed_df: pd.DataFrame) -> None:
        original_cols = list(imputed_df.columns)
        _ = add_rolling_features(imputed_df)
        assert list(imputed_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Tests: add_trend_features
# ---------------------------------------------------------------------------


class TestTrendFeatures:
    def test_delta_columns_created(self, imputed_df: pd.DataFrame) -> None:
        result = add_trend_features(imputed_df)
        for col in VITAL_COLS:
            assert f"{col}_delta" in result.columns

    def test_delta_values_match_expected(
        self, imputed_df: pd.DataFrame
    ) -> None:
        """Vitals have a linear ramp of +1.0 per hour, so deltas should be 1.0."""
        result = add_trend_features(imputed_df)
        p1 = result[result["patient_id"] == "p001"]

        for col in VITAL_COLS:
            # First hour delta is 0.0 (fillna)
            assert p1.iloc[0][f"{col}_delta"] == pytest.approx(0.0)
            # Subsequent hours: each value increases by 1.0
            for i in range(1, len(p1)):
                assert p1.iloc[i][f"{col}_delta"] == pytest.approx(1.0), (
                    f"{col}_delta at row {i} should be 1.0"
                )

    def test_first_hour_delta_is_zero(self, imputed_df: pd.DataFrame) -> None:
        result = add_trend_features(imputed_df)
        # Check both patients
        for pid in ("p001", "p002"):
            patient = result[result["patient_id"] == pid]
            for col in VITAL_COLS:
                assert patient.iloc[0][f"{col}_delta"] == 0.0

    def test_deltas_independent_per_patient(
        self, imputed_df: pd.DataFrame
    ) -> None:
        """Patient 2's first delta should be 0.0, not a diff from patient 1's last row."""
        result = add_trend_features(imputed_df)
        p2 = result[result["patient_id"] == "p002"]

        for col in VITAL_COLS:
            assert p2.iloc[0][f"{col}_delta"] == 0.0


# ---------------------------------------------------------------------------
# Tests: build_feature_matrix
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrix:
    def test_returns_correct_shapes(self, imputed_df: pd.DataFrame) -> None:
        X, y = build_feature_matrix(imputed_df)

        assert len(X) == len(imputed_df)
        assert len(y) == len(imputed_df)
        # X should have more columns than the raw feature set (rolling + delta added)
        assert X.shape[1] > len(ALL_FEATURE_COLS)

    def test_x_excludes_metadata_columns(
        self, imputed_df: pd.DataFrame
    ) -> None:
        X, _ = build_feature_matrix(imputed_df)

        assert "patient_id" not in X.columns
        assert "hospital" not in X.columns
        assert "SepsisLabel" not in X.columns

    def test_y_matches_sepsis_label(self, imputed_df: pd.DataFrame) -> None:
        _, y = build_feature_matrix(imputed_df)

        expected = imputed_df["SepsisLabel"]
        pd.testing.assert_series_equal(y, expected, check_names=False)

    def test_raises_without_label_column(
        self, imputed_df: pd.DataFrame
    ) -> None:
        df_no_label = imputed_df.drop(columns=["SepsisLabel"])
        with pytest.raises(KeyError, match="SepsisLabel"):
            build_feature_matrix(df_no_label)

    def test_x_contains_rolling_and_delta_columns(
        self, imputed_df: pd.DataFrame
    ) -> None:
        X, _ = build_feature_matrix(imputed_df)

        # Spot-check a few rolling columns
        assert f"{VITAL_COLS[0]}_roll_mean" in X.columns
        assert f"{VITAL_COLS[0]}_roll_std" in X.columns

        # Spot-check delta columns
        for col in VITAL_COLS:
            assert f"{col}_delta" in X.columns


# ---------------------------------------------------------------------------
# Tests: get_feature_names
# ---------------------------------------------------------------------------


class TestGetFeatureNames:
    def test_excludes_metadata(self, imputed_df: pd.DataFrame) -> None:
        X, _ = build_feature_matrix(imputed_df)
        # Re-add metadata columns to simulate a full DataFrame
        full = pd.concat(
            [
                imputed_df[["patient_id", "hospital"]].reset_index(drop=True),
                X.reset_index(drop=True),
                imputed_df[["SepsisLabel"]].reset_index(drop=True),
            ],
            axis=1,
        )

        names = get_feature_names(full)
        assert "patient_id" not in names
        assert "hospital" not in names
        assert "SepsisLabel" not in names

    def test_returns_list_of_strings(self, imputed_df: pd.DataFrame) -> None:
        names = get_feature_names(imputed_df)
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_order_matches_dataframe(self, imputed_df: pd.DataFrame) -> None:
        names = get_feature_names(imputed_df)
        expected = [
            c for c in imputed_df.columns
            if c not in {"patient_id", "hospital", "SepsisLabel"}
            and c not in set(EXCLUDED_FEATURES)
        ]
        assert names == expected


# ---------------------------------------------------------------------------
# Tests: excluded features
# ---------------------------------------------------------------------------


class TestExcludedFeatures:
    def test_excluded_features_not_in_x(self, imputed_df: pd.DataFrame) -> None:
        X, y = build_feature_matrix(imputed_df)
        for col in EXCLUDED_FEATURES:
            assert col not in X.columns, f"{col} should be excluded from X"


# ---------------------------------------------------------------------------
# Tests: scale_features
# ---------------------------------------------------------------------------


class TestScaleFeatures:
    def test_returns_scaled_arrays_and_scaler(self, imputed_df: pd.DataFrame) -> None:
        X, y = build_feature_matrix(imputed_df)
        mid = len(X) // 2
        X_tr_s, X_val_s, scaler = scale_features(X.iloc[:mid], X.iloc[mid:])
        assert X_tr_s.shape == X.iloc[:mid].shape
        assert X_val_s.shape == X.iloc[mid:].shape
        assert scaler is not None

    def test_scaled_train_has_zero_mean(self, imputed_df: pd.DataFrame) -> None:
        X, y = build_feature_matrix(imputed_df)
        mid = len(X) // 2
        X_tr_s, _, _ = scale_features(X.iloc[:mid], X.iloc[mid:])
        means = np.abs(X_tr_s.mean())
        assert (means < 0.1).all()
