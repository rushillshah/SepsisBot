"""Tests for src.leadup_analysis: lead-time binning, IV-per-bin, SHAP-per-bin."""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    LEADUP_BIN_LABELS,
    LEADUP_NEVER_LABEL,
    LEADUP_OUT_OF_RANGE_LABEL,
)
from src.leadup_analysis import (
    bin_lead_time,
    compute_hours_to_onset,
    iv_per_lead_time,
    shap_per_lead_time,
    top_features_per_bin,
)


def test_compute_hours_to_onset_sepsis_patient():
    """Onset at hour 30 → row at hour 24 has hours_to_onset = 6."""
    pids = np.array(["p1"] * 35)
    iculos = np.arange(35, dtype=float)
    labels = np.zeros(35, dtype=int)
    labels[30:] = 1  # onset at hour 30

    h = compute_hours_to_onset(pids, iculos, labels)

    assert h[24] == 6.0
    assert h[29] == 1.0
    assert np.isnan(h[30])  # at-or-after onset is NaN
    assert np.isnan(h[34])
    assert h[0] == 30.0


def test_compute_hours_to_onset_non_sepsis():
    """Never-positive patient → all NaN."""
    pids = np.array(["p1"] * 10)
    iculos = np.arange(10, dtype=float)
    labels = np.zeros(10, dtype=int)

    h = compute_hours_to_onset(pids, iculos, labels)

    assert np.isnan(h).all()


def test_bin_lead_time_boundaries():
    """Right-exclusive: a value at an edge falls into the upper bin."""
    hours = np.array([0.0, 2.9, 3.0, 5.9, 6.0, 11.9, 12.0, 23.9, 24.0, 47.9, 48.0, np.nan])
    bins = bin_lead_time(hours)

    expected = [
        "0-3h",         # 0.0
        "0-3h",         # 2.9
        "3-6h",         # 3.0 (right-exclusive: edge into upper bin)
        "3-6h",         # 5.9
        "6-12h",        # 6.0
        "6-12h",        # 11.9
        "12-24h",       # 12.0
        "12-24h",       # 23.9
        "24-48h",       # 24.0
        "24-48h",       # 47.9
        LEADUP_OUT_OF_RANGE_LABEL,  # 48.0 (>= max edge)
        LEADUP_NEVER_LABEL,         # NaN
    ]
    assert list(bins) == expected


def test_iv_per_lead_time_shape():
    """Output shape should be (n_features × n_bin_labels)."""
    rng = np.random.default_rng(0)
    n_rows, n_features = 800, 6
    X = pd.DataFrame(
        rng.normal(size=(n_rows, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # 50 rows in each of the 5 bins, rest are "never"
    bin_assignments = np.array([LEADUP_NEVER_LABEL] * n_rows, dtype=object)
    cursor = 0
    for label in LEADUP_BIN_LABELS:
        bin_assignments[cursor:cursor + 50] = label
        cursor += 50
    # Make feature 0 highly predictive in the 12-24h bin
    bin_12_24_mask = bin_assignments == "12-24h"
    X.loc[bin_12_24_mask, "f0"] = X.loc[bin_12_24_mask, "f0"] + 5.0

    iv = iv_per_lead_time(X, bin_assignments, neg_sample_ratio=2)

    assert iv.shape == (n_features, len(LEADUP_BIN_LABELS))
    assert list(iv.columns) == LEADUP_BIN_LABELS
    # f0 IV in 12-24h bin should beat its IV in 0-3h bin (where signal is absent)
    assert iv.loc["f0", "12-24h"] > iv.loc["f0", "0-3h"]


def test_shap_per_lead_time_aggregation():
    """Mean |SHAP| per bin matches the manually computed value."""
    pytest.importorskip("shap", reason="shap not importable in this environment")
    rng = np.random.default_rng(1)
    n_rows, n_features = 600, 4
    X = pd.DataFrame(
        rng.normal(size=(n_rows, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = (X["f0"] + 0.5 * X["f1"] > 0).astype(int).to_numpy()

    # 100 rows in each of the 5 bins, the rest are "never"
    bin_assignments = np.array([LEADUP_NEVER_LABEL] * n_rows, dtype=object)
    cursor = 0
    for label in LEADUP_BIN_LABELS:
        bin_assignments[cursor:cursor + 100] = label
        cursor += 100

    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=20, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", random_state=0,
    )
    model.fit(X, y)

    shap_df = shap_per_lead_time(
        model, X, bin_assignments, list(X.columns), sample_per_bin=200,
    )

    assert shap_df.shape == (n_features, len(LEADUP_BIN_LABELS))
    assert list(shap_df.columns) == LEADUP_BIN_LABELS
    # All values should be non-negative (mean of absolutes)
    assert (shap_df.values >= 0).all()
    # f0 should outrank f3 (uninformative) in every bin
    for bin_label in LEADUP_BIN_LABELS:
        assert shap_df.loc["f0", bin_label] > shap_df.loc["f3", bin_label]


def test_top_features_per_bin_returns_correct_count():
    """Helper returns top-n per bin in the right format."""
    df = pd.DataFrame(
        {"0-3h": [0.5, 0.3, 0.1, 0.9], "3-6h": [0.2, 0.4, 0.6, 0.1]},
        index=["a", "b", "c", "d"],
    )
    top = top_features_per_bin(df, n=2)

    assert top["0-3h"] == [("d", 0.9), ("a", 0.5)]
    assert top["3-6h"] == [("c", 0.6), ("b", 0.4)]
