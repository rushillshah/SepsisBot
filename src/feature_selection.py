"""Feature selection utilities for clinically cleaner model training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_importance import compute_information_value


def prune_collinear_by_iv(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    threshold: float = 0.80,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Drop collinear features, retaining the higher-IV feature in each pair.

    IV is computed only from the supplied training rows. The validation fold
    must not be included when this is used inside cross-validation.
    """
    y_series = pd.Series(y_train, index=X_train.index)
    iv_df = compute_information_value(X_train, y_series)
    iv_by_feature = iv_df.set_index("feature")["iv"].to_dict()

    corr = X_train.corr(method="pearson").abs()
    columns = list(corr.columns)
    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    row_idx, col_idx = np.where((corr.to_numpy() >= threshold) & upper)

    pairs = []
    for i, j in zip(row_idx, col_idx, strict=False):
        left = columns[i]
        right = columns[j]
        pairs.append((float(corr.iat[i, j]), left, right))
    pairs.sort(reverse=True)

    dropped: set[str] = set()
    audit_rows: list[dict] = []
    for corr_value, left, right in pairs:
        if left in dropped or right in dropped:
            continue

        left_iv = float(iv_by_feature.get(left, 0.0))
        right_iv = float(iv_by_feature.get(right, 0.0))
        if left_iv > right_iv:
            keep, drop = left, right
            keep_iv, drop_iv = left_iv, right_iv
        elif right_iv > left_iv:
            keep, drop = right, left
            keep_iv, drop_iv = right_iv, left_iv
        else:
            keep, drop = sorted([left, right])[0], sorted([left, right])[1]
            keep_iv, drop_iv = left_iv, right_iv

        dropped.add(drop)
        audit_rows.append({
            "feature_a": left,
            "feature_b": right,
            "abs_corr": corr_value,
            "kept_feature": keep,
            "kept_iv": keep_iv,
            "dropped_feature": drop,
            "dropped_iv": drop_iv,
            "reason": f"abs_corr>={threshold}; retained higher IV",
        })

    kept = [col for col in X_train.columns if col not in dropped]
    audit_df = pd.DataFrame(audit_rows)
    return kept, audit_df, iv_df
