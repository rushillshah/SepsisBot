"""Tests for src.threshold_analysis.patient_intersection_at_thresholds."""

import pandas as pd

from src.threshold_analysis import patient_intersection_at_thresholds


def _make_preds(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["patient_id", "label", "prob"])


class TestPatientIntersectionAtThresholds:
    def test_basic_venn_counts(self) -> None:
        # 4 patients. Cook the per-patient max probs so we know who is flagged
        # at threshold 0.30.
        # p1: full=0.9, early=0.8 -> flagged by both, actual=1
        # p2: full=0.6, early=0.1 -> flagged by full only, actual=1
        # p3: full=0.1, early=0.5 -> flagged by early only, actual=0
        # p4: full=0.05, early=0.05 -> flagged by neither, actual=0
        full = _make_preds([
            ("p1", 1, 0.9),
            ("p2", 1, 0.6),
            ("p3", 0, 0.1),
            ("p4", 0, 0.05),
        ])
        early = _make_preds([
            ("p1", 1, 0.8),
            ("p2", 1, 0.1),
            ("p3", 0, 0.5),
            ("p4", 0, 0.05),
        ])

        out = patient_intersection_at_thresholds(full, early, thresholds=[0.30])
        row = out.iloc[0]
        assert row["total_patients"] == 4
        assert row["actual_sepsis"] == 2
        assert row["flagged_full"] == 2  # p1, p2
        assert row["flagged_early"] == 2  # p1, p3
        assert row["flagged_both"] == 1  # p1
        assert row["flagged_full_only"] == 1  # p2
        assert row["flagged_early_only"] == 1  # p3
        assert row["flagged_neither"] == 1  # p4
        assert row["tp_intersection"] == 1  # p1 (both flagged + sepsis)
        assert row["fp_intersection"] == 0
        assert row["sensitivity_intersection"] == 0.5  # 1 of 2 sepsis patients
        assert row["precision_intersection"] == 1.0  # 1 of 1 both-flagged patients

    def test_max_per_patient_takes_max_across_hours(self) -> None:
        # p1 has 3 hours: max prob in full = 0.9, max in early = 0.05
        full = _make_preds([
            ("p1", 1, 0.05),
            ("p1", 1, 0.9),
            ("p1", 1, 0.7),
        ])
        early = _make_preds([
            ("p1", 1, 0.05),
            ("p1", 1, 0.04),
            ("p1", 1, 0.05),
        ])
        out = patient_intersection_at_thresholds(full, early, thresholds=[0.30])
        row = out.iloc[0]
        assert row["flagged_full"] == 1
        assert row["flagged_early"] == 0
        assert row["flagged_both"] == 0

    def test_inner_join_drops_missing_patients(self) -> None:
        # p2 only in full set -> excluded from intersection.
        full = _make_preds([("p1", 0, 0.4), ("p2", 0, 0.4)])
        early = _make_preds([("p1", 0, 0.4)])
        out = patient_intersection_at_thresholds(full, early, thresholds=[0.30])
        assert out.iloc[0]["total_patients"] == 1
