"""Feature importance analysis for the sepsis prediction model.

Provides three complementary methods — Information Value (IV),
XGBoost gain-based importance, and SHAP values — plus a combined
ranking that merges all three into a single prioritized list.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ── Information Value ────────────────────────────────────────────────────────


def _iv_strength_label(iv: float) -> str:
    """Classify an IV score into a human-readable strength category."""
    if iv < 0.02:
        return "Useless"
    if iv < 0.1:
        return "Weak"
    if iv < 0.3:
        return "Medium"
    if iv < 0.5:
        return "Strong"
    return "Suspicious"


def _compute_iv_for_feature(
    values: pd.Series,
    target: pd.Series,
    n_bins: int = 10,
) -> float:
    """Compute Information Value for a single feature column.

    Uses quantile binning with smoothing to handle bins that contain
    zero events or zero non-events.
    """
    total_events = target.sum()
    total_non_events = len(target) - total_events

    if total_events == 0 or total_non_events == 0:
        return 0.0

    # Constant features produce a single bin — IV is zero.
    if values.nunique() <= 1:
        return 0.0

    try:
        bins = pd.qcut(values, q=n_bins, duplicates="drop")
    except ValueError:
        return 0.0

    iv = 0.0
    for _, group_target in target.groupby(bins):
        events = group_target.sum()
        non_events = len(group_target) - events

        # Smoothing: add 0.5 to avoid log(0).
        events_smooth = events + 0.5
        non_events_smooth = non_events + 0.5

        dist_events = events_smooth / (total_events + 0.5 * len(bins.cat.categories))
        dist_non_events = non_events_smooth / (
            total_non_events + 0.5 * len(bins.cat.categories)
        )

        woe = np.log(dist_events / dist_non_events)

        if np.isfinite(woe):
            iv += (dist_events - dist_non_events) * woe

    return iv if np.isfinite(iv) else 0.0


def compute_information_value(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Compute Information Value for every feature in X.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows = observations, columns = features).
    y : pd.Series
        Binary target (1 = sepsis, 0 = no sepsis).

    Returns
    -------
    pd.DataFrame
        Columns: feature, iv, iv_strength.  Sorted by iv descending.
    """
    records = []
    for col in X.columns:
        iv = _compute_iv_for_feature(X[col], y)
        records.append({"feature": col, "iv": iv})

    df = pd.DataFrame(records)
    df["iv_strength"] = df["iv"].apply(_iv_strength_label)
    return df.sort_values("iv", ascending=False).reset_index(drop=True)


def compute_woe_buckets(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str] | None = None,
    n_bins: int = 10,
) -> dict[str, pd.DataFrame]:
    """Compute WOE bucket detail for selected features.

    For each feature, returns a DataFrame with one row per bucket showing
    the value range, observation count, event rate, WOE, and IV contribution.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target (1 = sepsis).
    features : list[str] | None
        Features to compute. Defaults to top 20 by IV.
    n_bins : int
        Number of quantile buckets.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of feature name to bucket-detail DataFrame.
    """
    if features is None:
        iv_df = compute_information_value(X, y)
        features = iv_df.head(20)["feature"].tolist()

    total_events = y.sum()
    total_non_events = len(y) - total_events
    result = {}

    for col in features:
        if col not in X.columns:
            continue
        values = X[col]
        if values.nunique() <= 1:
            continue

        try:
            bins = pd.qcut(values, q=n_bins, duplicates="drop")
        except ValueError:
            continue

        rows = []
        for bucket, group_idx in y.groupby(bins):
            events = group_idx.sum()
            non_events = len(group_idx) - events
            event_rate = events / len(group_idx) if len(group_idx) > 0 else 0

            events_s = events + 0.5
            non_events_s = non_events + 0.5
            n_cats = len(bins.cat.categories)
            dist_e = events_s / (total_events + 0.5 * n_cats)
            dist_ne = non_events_s / (total_non_events + 0.5 * n_cats)

            woe = float(np.log(dist_e / dist_ne))
            iv_contrib = float((dist_e - dist_ne) * woe) if np.isfinite(woe) else 0.0

            rows.append({
                "bucket": str(bucket),
                "count": int(len(group_idx)),
                "events": int(events),
                "event_rate": round(float(event_rate), 4),
                "woe": round(woe, 4),
                "iv_contribution": round(iv_contrib, 4),
            })

        if rows:
            result[col] = pd.DataFrame(rows)

    return result


# ── Gain-Based Importance ────────────────────────────────────────────────────


def compute_gain_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract and normalize gain-based feature importances from an XGBoost model.

    Parameters
    ----------
    model
        A fitted model exposing ``.feature_importances_`` (e.g. XGBClassifier).
    feature_names : list[str]
        Ordered list of feature names matching the model's training columns.

    Returns
    -------
    pd.DataFrame
        Columns: feature, gain, gain_pct.  Sorted by gain_pct descending.
    """
    raw = model.feature_importances_
    total = raw.sum()
    pct = (raw / total * 100.0) if total > 0 else raw

    df = pd.DataFrame({
        "feature": feature_names,
        "gain": raw,
        "gain_pct": pct,
    })
    return df.sort_values("gain_pct", ascending=False).reset_index(drop=True)


# ── SHAP Values ──────────────────────────────────────────────────────────────


def compute_shap_values(
    model,
    X_sample,
    feature_names: list[str],
    save_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compute SHAP values and optionally save summary plots.

    Parameters
    ----------
    model
        A fitted tree-based model compatible with ``shap.TreeExplainer``.
    X_sample : np.ndarray | pd.DataFrame
        A sample of the feature matrix (recommend ~10 000 rows for speed).
    feature_names : list[str]
        Ordered list of feature names matching the columns of *X_sample*.
    save_dir : str | Path | None
        If provided, directory where ``shap_summary.png`` and
        ``shap_bar.png`` will be saved.

    Returns
    -------
    pd.DataFrame
        Columns: feature, mean_abs_shap.  Sorted descending.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers, shap_values may be a list of two arrays;
    # use the positive-class array.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    })
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Ensure X_sample is a DataFrame for SHAP plot labels.
        if not isinstance(X_sample, pd.DataFrame):
            X_sample = pd.DataFrame(X_sample, columns=feature_names)

        # Beeswarm summary plot.
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(save_path / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Bar plot of mean |SHAP|.
        shap.summary_plot(
            shap_values, X_sample, plot_type="bar", show=False,
        )
        plt.savefig(save_path / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()

    return df


# ── Combined Ranking ─────────────────────────────────────────────────────────


def combined_feature_ranking(
    iv_df: pd.DataFrame,
    gain_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """Merge three importance metrics and produce an averaged ranking.

    Parameters
    ----------
    iv_df : pd.DataFrame
        Output of :func:`compute_information_value`.
    gain_df : pd.DataFrame
        Output of :func:`compute_gain_importance`.
    shap_df : pd.DataFrame
        Output of :func:`compute_shap_values`.
    top_n : int
        Number of top-ranked features to return.

    Returns
    -------
    pd.DataFrame
        Columns: feature, iv, iv_rank, gain_pct, gain_rank,
        mean_abs_shap, shap_rank, average_rank.
        Sorted by average_rank ascending, limited to *top_n* rows.
    """
    merged = (
        iv_df[["feature", "iv"]]
        .merge(gain_df[["feature", "gain_pct"]], on="feature", how="outer")
        .merge(shap_df[["feature", "mean_abs_shap"]], on="feature", how="outer")
    )

    # Fill missing values with 0 for features absent from one method.
    merged = merged.fillna(0.0)

    merged["iv_rank"] = merged["iv"].rank(ascending=False, method="min").astype(int)
    merged["gain_rank"] = (
        merged["gain_pct"].rank(ascending=False, method="min").astype(int)
    )
    merged["shap_rank"] = (
        merged["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    )

    merged["average_rank"] = (
        merged[["iv_rank", "gain_rank", "shap_rank"]].mean(axis=1)
    )

    merged = merged.sort_values("average_rank").reset_index(drop=True)

    columns = [
        "feature",
        "iv",
        "iv_rank",
        "gain_pct",
        "gain_rank",
        "mean_abs_shap",
        "shap_rank",
        "average_rank",
    ]
    return merged[columns].head(top_n)
