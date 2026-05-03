"""Tests for _causal_cusum_baselines and add_dynamic_baselines.

The central correctness property: ``baselines[t]`` must depend only on
``values[:t+1]``. Truncating future values must not change the prefix of
the baseline output.
"""

import numpy as np
import pandas as pd

from src.config import TIME_COL
from src.features import _causal_cusum_baselines, add_dynamic_baselines


class TestCausalCusumBaselines:
    def test_causality_random_walk(self) -> None:
        """Truncating future values must not change earlier baselines."""
        rng = np.random.default_rng(42)
        full_values = rng.normal(loc=80.0, scale=5.0, size=50)
        # Inject a regime shift at hour 30 to make CUSUM trigger.
        full_values[30:] += 30.0

        full_baselines = _causal_cusum_baselines(full_values)

        # For every prefix length k, baseline computed on the prefix must
        # equal the prefix of the baseline computed on the full series.
        for k in range(1, len(full_values) + 1):
            prefix_baselines = _causal_cusum_baselines(full_values[:k])
            np.testing.assert_allclose(
                prefix_baselines,
                full_baselines[:k],
                rtol=1e-9,
                err_msg=f"Causality violated at prefix length {k}",
            )

    def test_running_mean_when_no_changepoint(self) -> None:
        # Stable values — CUSUM should never trigger; baselines = running mean.
        values = np.array([5.0] * 20)
        baselines = _causal_cusum_baselines(values)
        # All running means of constant series equal 5.0
        np.testing.assert_allclose(baselines, np.full_like(baselines, 5.0))

    def test_freezes_baseline_after_changepoint(self) -> None:
        # First 10 hours stable around 0, then large jump.
        values = np.concatenate([np.zeros(10), np.full(20, 100.0)])
        baselines = _causal_cusum_baselines(values)
        # Hours 0..5 use running mean (will be 0 for stable zeros).
        assert baselines[5] == 0.0
        # Eventually changepoint should fire and freeze baseline near 0.
        # Once frozen, baselines stay constant for the rest of the series.
        assert len(set(baselines[-5:].round(8).tolist())) == 1, "Tail must be frozen constant"
        # Frozen baseline should be a small number (computed from pre-change zeros).
        assert abs(baselines[-1]) < 1.0

    def test_short_series_uses_running_mean(self) -> None:
        values = np.array([10.0, 20.0, 30.0])
        baselines = _causal_cusum_baselines(values)
        np.testing.assert_allclose(baselines, [10.0, 15.0, 20.0])

    def test_empty_series(self) -> None:
        baselines = _causal_cusum_baselines(np.array([]))
        assert len(baselines) == 0


class TestAddDynamicBaselinesCausal:
    def test_two_patient_dataframe_causal(self) -> None:
        # Build a tiny df with one stable patient and one with a regime shift.
        rows = []
        for h in range(20):
            rows.append({"patient_id": "stable", TIME_COL: h, "HR": 75.0})
            hr_shift = 75.0 if h < 10 else 130.0
            rows.append({"patient_id": "shift", TIME_COL: h, "HR": hr_shift})

        df = pd.DataFrame(rows)
        out = add_dynamic_baselines(df)
        assert "HR_baseline_dev" in out.columns

        # Stable patient should have near-zero deviations everywhere.
        stable_dev = out[out["patient_id"] == "stable"]["HR_baseline_dev"].values
        np.testing.assert_allclose(stable_dev, np.zeros(20), atol=1e-9)

        # Shift patient: hours 0..9 should have small deviations (running
        # mean of 75 minus 75 = 0). Hours 10+ should have large positive
        # deviations once the CUSUM triggers.
        shift_dev = out[out["patient_id"] == "shift"].sort_values(TIME_COL)["HR_baseline_dev"].values
        np.testing.assert_allclose(shift_dev[:10], np.zeros(10), atol=1e-9)
        # Last hour deviation should be ≈ 130 - 75 = 55 (or close, depending on
        # exactly when CUSUM triggered).
        assert shift_dev[-1] > 30.0
