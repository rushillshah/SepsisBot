"""Tests for src.data_loader: PSV parsing and parquet round-tripping."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_processed, load_psv, save_processed

# Guard parquet tests so they skip when no engine is installed.
try:
    import pyarrow  # noqa: F401

    _HAS_PARQUET_ENGINE = True
except ImportError:
    try:
        import fastparquet  # noqa: F401

        _HAS_PARQUET_ENGINE = True
    except ImportError:
        _HAS_PARQUET_ENGINE = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_psv(tmp_path: Path) -> Path:
    """Write a minimal pipe-separated file and return its path."""
    content = (
        "HR|O2Sat|Temp|SBP|MAP|ICULOS|SepsisLabel\n"
        "80|97.0|36.8|120|80|1|0\n"
        "82|96.5|36.9|118|79|2|0\n"
        "85|95.0|37.1|115|77|3|1\n"
    )
    psv_path = tmp_path / "p000001.psv"
    psv_path.write_text(content)
    return psv_path


@pytest.fixture()
def empty_psv(tmp_path: Path) -> Path:
    """Write a PSV file that has a header but zero data rows."""
    content = "HR|O2Sat|Temp|SBP|MAP|ICULOS|SepsisLabel\n"
    psv_path = tmp_path / "p_empty.psv"
    psv_path.write_text(content)
    return psv_path


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """A small DataFrame suitable for save/load round-trip testing."""
    return pd.DataFrame(
        {
            "HR": [80, 82, 85],
            "O2Sat": [97.0, 96.5, 95.0],
            "SepsisLabel": [0, 0, 1],
        }
    )


# ---------------------------------------------------------------------------
# load_psv tests
# ---------------------------------------------------------------------------


class TestLoadPsv:
    def test_loads_valid_psv(self, sample_psv: Path) -> None:
        df = load_psv(sample_psv)
        assert len(df) == 3
        assert "HR" in df.columns
        assert "SepsisLabel" in df.columns
        assert df["HR"].iloc[0] == 80

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.psv"
        with pytest.raises(FileNotFoundError, match="PSV file not found"):
            load_psv(missing)

    def test_raises_value_error_for_empty_file(self, empty_psv: Path) -> None:
        with pytest.raises(ValueError, match="PSV file is empty"):
            load_psv(empty_psv)


# ---------------------------------------------------------------------------
# save_processed / load_processed round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_PARQUET_ENGINE,
    reason="pyarrow or fastparquet required for parquet round-trip tests",
)
class TestParquetRoundTrip:
    def test_round_trip_preserves_data(
        self, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Save then load a parquet file; the data should be identical."""
        # Redirect DATA_PROCESSED to a temp directory so tests don't write
        # into the real project tree.
        monkeypatch.setattr("src.data_loader.DATA_PROCESSED", tmp_path)

        out_path = save_processed(sample_df, "test_data")
        assert out_path.exists()
        assert out_path.suffix == ".parquet"

        loaded = load_processed("test_data")
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_round_trip_with_explicit_extension(
        self, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Passing 'name.parquet' should not double-append the extension."""
        monkeypatch.setattr("src.data_loader.DATA_PROCESSED", tmp_path)

        out_path = save_processed(sample_df, "explicit.parquet")
        assert out_path.name == "explicit.parquet"

        loaded = load_processed("explicit.parquet")
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_save_raises_on_empty_dataframe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("src.data_loader.DATA_PROCESSED", tmp_path)
        with pytest.raises(ValueError, match="empty DataFrame"):
            save_processed(pd.DataFrame(), "should_fail")

    def test_load_raises_on_missing_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("src.data_loader.DATA_PROCESSED", tmp_path)
        with pytest.raises(FileNotFoundError, match="Processed file not found"):
            load_processed("nonexistent")
