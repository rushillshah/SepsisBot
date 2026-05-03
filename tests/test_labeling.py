"""Tests for src.labeling: post-onset censoring."""

import numpy as np
import pandas as pd
import pytest

from src.config import EARLY_LABEL_EXTRA_HOURS
from src.labeling import censor_post_onset


def _make_inputs(records: list[dict]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bundle a list of (pid, iculos, sepsis, train_label) dicts into the 5-array shape
    censor_post_onset expects.

    Records use keys: ``pid``, ``iculos``, ``sepsis`` (eval label), ``train`` (train label),
    plus a stub feature ``f1`` so X has columns.
    """
    df = pd.DataFrame(records)
    X = pd.DataFrame({"f1": df["iculos"].astype(float).values})
    return (
        X,
        df["train"].to_numpy(),
        df["sepsis"].to_numpy(),
        df["pid"].to_numpy(),
        df["iculos"].to_numpy(),
    )


class TestCensorPostOnset:
    def test_non_septic_patients_kept_entirely(self) -> None:
        records = [
            {"pid": "p1", "iculos": h, "sepsis": 0, "train": 0}
            for h in range(1, 11)
        ]
        X, y_train, y_eval, pids, iculos = _make_inputs(records)

        X_out, yt_out, ye_out, pids_out, iculos_out, info = censor_post_onset(
            X, y_train, y_eval, pids, iculos
        )

        assert len(X_out) == 10
        assert info["n_rows_censored"] == 0
        assert info["n_patients_with_censored_rows"] == 0
        assert (pids_out == "p1").all()

    def test_septic_patient_drops_rows_at_or_after_onset(self) -> None:
        # SepsisLabel flips on at hour 5 -> first_pos=5 -> t_onset = 5 + 6 = 11
        # Keep iculos < 11 (all of hours 1..10). Discharge at hour 16.
        # Expect: drop hours 11..16 (6 rows), keep 1..10 (10 rows).
        records = []
        for h in range(1, 17):
            records.append({
                "pid": "p1",
                "iculos": h,
                "sepsis": 1 if h >= 5 else 0,
                "train": 1 if h >= 5 else 0,
            })
        X, y_train, y_eval, pids, iculos = _make_inputs(records)

        X_out, yt_out, ye_out, pids_out, iculos_out, info = censor_post_onset(
            X, y_train, y_eval, pids, iculos
        )

        assert info["n_rows_censored"] == 6
        assert info["n_patients_with_censored_rows"] == 1
        # Kept hours: 1..10 inclusive (iculos < 11)
        assert sorted(iculos_out.tolist()) == list(range(1, 11))
        # Pre-onset window (h=5..10) preserves SepsisLabel=1
        assert int((ye_out == 1).sum()) == 6  # hours 5,6,7,8,9,10

    def test_mixed_cohort(self) -> None:
        # p1: never septic, 8 rows -> all kept.
        # p2: SepsisLabel=1 from h=10 -> t_onset = 10 + 6 = 16. Rows 1..20.
        #     Keep iculos < 16 -> 15 rows. Drop iculos>=16 -> 5 rows.
        records = []
        for h in range(1, 9):
            records.append({"pid": "p1", "iculos": h, "sepsis": 0, "train": 0})
        for h in range(1, 21):
            records.append({
                "pid": "p2", "iculos": h,
                "sepsis": 1 if h >= 10 else 0,
                "train": 1 if h >= 10 else 0,
            })
        X, y_train, y_eval, pids, iculos = _make_inputs(records)

        X_out, _, _, pids_out, iculos_out, info = censor_post_onset(
            X, y_train, y_eval, pids, iculos
        )

        assert info["n_rows_censored"] == 5  # p2 hours 16..20
        assert info["n_patients_with_censored_rows"] == 1
        assert int((pids_out == "p1").sum()) == 8
        assert int((pids_out == "p2").sum()) == 15
        kept_p2 = sorted(iculos_out[pids_out == "p2"].tolist())
        assert kept_p2 == list(range(1, 16))

    def test_uses_early_label_extra_hours_constant(self) -> None:
        # Verify t_onset = first_pos + EARLY_LABEL_EXTRA_HOURS (not hard-coded 6).
        # If someone changes the constant, this test should still pass.
        first_pos_h = 4
        t_onset = first_pos_h + EARLY_LABEL_EXTRA_HOURS
        post_onset_extra = 3
        last_h = t_onset + post_onset_extra - 1  # last hour kept post-onset = t_onset + 2

        records = [
            {
                "pid": "p1", "iculos": h,
                "sepsis": 1 if h >= first_pos_h else 0,
                "train": 1 if h >= first_pos_h else 0,
            }
            for h in range(1, last_h + 1)
        ]
        X, y_train, y_eval, pids, iculos = _make_inputs(records)

        _, _, _, _, iculos_out, info = censor_post_onset(X, y_train, y_eval, pids, iculos)

        # Censored rows = those with iculos in [t_onset, last_h] = post_onset_extra rows
        assert info["n_rows_censored"] == post_onset_extra
        assert iculos_out.max() == t_onset - 1

    def test_arrays_stay_aligned(self) -> None:
        # After censoring, features, train labels, eval labels, pids, iculos
        # must still describe the same row in the same order.
        records = [
            {"pid": "p1", "iculos": 1, "sepsis": 0, "train": 0},
            {"pid": "p1", "iculos": 2, "sepsis": 0, "train": 1},  # train=1 to test alignment
            {"pid": "p1", "iculos": 3, "sepsis": 1, "train": 1},  # first_pos=3 -> t_onset=9
            {"pid": "p1", "iculos": 4, "sepsis": 1, "train": 1},
            {"pid": "p1", "iculos": 9, "sepsis": 1, "train": 1},  # at t_onset, dropped
            {"pid": "p1", "iculos": 10, "sepsis": 1, "train": 1},  # dropped
        ]
        X, y_train, y_eval, pids, iculos = _make_inputs(records)
        # Make X[i] uniquely identifiable so we can check post-filter alignment.
        X = pd.DataFrame({"row_id": np.arange(len(X), dtype=float)})

        X_out, yt_out, ye_out, _, iculos_out, _ = censor_post_onset(
            X, y_train, y_eval, pids, iculos
        )

        # 4 rows survive (h=1..4)
        assert len(X_out) == 4
        kept_indices = X_out["row_id"].astype(int).tolist()
        assert kept_indices == [0, 1, 2, 3]
        assert yt_out.tolist() == [0, 1, 1, 1]
        assert ye_out.tolist() == [0, 0, 1, 1]
        assert iculos_out.tolist() == [1, 2, 3, 4]

    def test_length_mismatch_raises(self) -> None:
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            censor_post_onset(
                X,
                np.array([0, 0]),  # mismatched
                np.array([0, 0, 0]),
                np.array(["p1", "p1", "p1"]),
                np.array([1, 2, 3]),
            )
