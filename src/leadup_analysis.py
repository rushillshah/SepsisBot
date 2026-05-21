"""Per-lead-time feature importance analysis.

Answers: at each clinical lead-time bin (e.g. 12-24h before onset), which
features carry the most predictive signal? Two complementary metrics:

* **IV per bin** — model-independent. Positives are rows in the bin,
  negatives are sampled never-sepsis rows.
* **SHAP per bin** — model-dependent. Mean |SHAP| of the rows in each bin,
  computed from a single XGBoost fitted on the full feature matrix.

Bins are right-exclusive intervals defined by ``LEADUP_BIN_EDGES``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    LEADUP_BIN_EDGES,
    LEADUP_BIN_LABELS,
    LEADUP_NEVER_LABEL,
    LEADUP_OUT_OF_RANGE_LABEL,
    RANDOM_STATE,
)


# ── Lead-time bookkeeping ────────────────────────────────────────────────────


def compute_hours_to_onset(
    patient_ids: np.ndarray,
    iculos: np.ndarray,
    eval_labels: np.ndarray,
) -> np.ndarray:
    """Hours until each row's patient first hits ``SepsisLabel == 1``.

    Returns ``NaN`` for rows belonging to never-sepsis patients and for
    rows at-or-after onset (post-onset is excluded from the leadup).
    """
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "iculos": iculos.astype(float),
        "label": eval_labels,
    })
    onset = (
        df[df["label"] == 1]
        .groupby("patient_id")["iculos"]
        .min()
        .rename("onset")
    )
    df = df.merge(onset, left_on="patient_id", right_index=True, how="left")
    hours = df["onset"] - df["iculos"]
    # Negative or zero => post-onset; mark as NaN (excluded from leadup)
    hours = hours.where(hours > 0, other=np.nan)
    return hours.to_numpy(dtype=float)


def bin_lead_time(
    hours_to_onset: np.ndarray,
    edges: list[int] | None = None,
    labels: list[str] | None = None,
) -> np.ndarray:
    """Map hours-to-onset to clinical bin labels.

    NaN inputs (never-sepsis) become ``LEADUP_NEVER_LABEL``.
    Values >= the largest edge become ``LEADUP_OUT_OF_RANGE_LABEL``.
    Right-exclusive: an exact value at an edge falls in the upper bin.
    """
    edges = edges or LEADUP_BIN_EDGES
    labels = labels or LEADUP_BIN_LABELS

    hours = np.asarray(hours_to_onset, dtype=float)
    out = np.full(hours.shape, LEADUP_NEVER_LABEL, dtype=object)

    in_range = ~np.isnan(hours)
    if in_range.any():
        binned = pd.cut(
            hours[in_range],
            bins=edges,
            labels=labels,
            right=False,
            include_lowest=True,
        )
        codes = pd.Series(binned).astype(object)
        codes = codes.where(codes.notna(), other=LEADUP_OUT_OF_RANGE_LABEL)
        out[in_range] = codes.to_numpy()

    return out


# ── Importance per bin ───────────────────────────────────────────────────────


def iv_per_lead_time(
    X: pd.DataFrame,
    lead_bins: np.ndarray,
    labels: list[str] | None = None,
    neg_sample_ratio: int = 3,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """IV for every (feature, lead-time bin).

    Per bin: positives are the rows in that bin, negatives are sampled from
    never-sepsis rows at ``neg_sample_ratio:1``. Returns a DataFrame indexed
    by feature with one column per bin label.
    """
    from src.feature_importance import compute_information_value  # heavy import

    labels = labels or LEADUP_BIN_LABELS
    rng = np.random.default_rng(random_state)
    never_idx = np.where(lead_bins == LEADUP_NEVER_LABEL)[0]

    out: dict[str, pd.Series] = {}
    for bin_label in labels:
        pos_idx = np.where(lead_bins == bin_label)[0]
        if len(pos_idx) == 0 or len(never_idx) == 0:
            out[bin_label] = pd.Series(0.0, index=X.columns)
            continue

        n_neg = min(len(never_idx), len(pos_idx) * neg_sample_ratio)
        neg_idx = rng.choice(never_idx, size=n_neg, replace=False)
        all_idx = np.concatenate([pos_idx, neg_idx])
        y_bin = pd.Series(
            np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))]).astype(int)
        )
        X_sub = X.iloc[all_idx].reset_index(drop=True)

        iv_df = compute_information_value(X_sub, y_bin)
        out[bin_label] = iv_df.set_index("feature")["iv"]

    result = pd.DataFrame(out).reindex(X.columns).fillna(0.0)
    result.index.name = "feature"
    return result


def shap_per_lead_time(
    model,
    X: pd.DataFrame,
    lead_bins: np.ndarray,
    feature_names: list[str],
    labels: list[str] | None = None,
    sample_per_bin: int = 2000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Mean |SHAP| per (feature, lead-time bin).

    Samples up to ``sample_per_bin`` rows from each lead-time bin (and from
    the ``never`` bin so the explainer sees a balanced background), runs
    SHAP once on the union, then aggregates ``mean(|shap|)`` per bin.
    """
    labels = labels or LEADUP_BIN_LABELS
    rng = np.random.default_rng(random_state)

    sampled_idx_parts: list[np.ndarray] = []
    sampled_bin_parts: list[np.ndarray] = []
    for bin_label in [*labels, LEADUP_NEVER_LABEL]:
        idx = np.where(lead_bins == bin_label)[0]
        if len(idx) == 0:
            continue
        n = min(len(idx), sample_per_bin)
        chosen = rng.choice(idx, size=n, replace=False)
        sampled_idx_parts.append(chosen)
        sampled_bin_parts.append(np.full(n, bin_label, dtype=object))

    if not sampled_idx_parts:
        return pd.DataFrame(0.0, index=feature_names, columns=labels)

    sampled_idx = np.concatenate(sampled_idx_parts)
    sampled_bins = np.concatenate(sampled_bin_parts)
    X_sample = X.iloc[sampled_idx]

    import shap  # heavy import — defer until needed

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    out: dict[str, pd.Series] = {}
    for bin_label in labels:
        mask = sampled_bins == bin_label
        if not mask.any():
            out[bin_label] = pd.Series(0.0, index=feature_names)
            continue
        mean_abs = np.abs(shap_values[mask]).mean(axis=0)
        out[bin_label] = pd.Series(mean_abs, index=feature_names)

    result = pd.DataFrame(out).reindex(feature_names).fillna(0.0)
    result.index.name = "feature"
    return result


def top_features_per_bin(
    importance_df: pd.DataFrame,
    n: int = 15,
) -> dict[str, list[tuple[str, float]]]:
    """Return the top-n features per bin as ``{bin: [(feature, score), ...]}``."""
    out: dict[str, list[tuple[str, float]]] = {}
    for col in importance_df.columns:
        top = importance_df[col].nlargest(n)
        out[col] = list(zip(top.index.tolist(), top.values.tolist()))
    return out


# ── Pipeline entry point ─────────────────────────────────────────────────────


_DEFAULT_XGB_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 150,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "min_child_weight": 10,
    "gamma": 0.5,
}


def run_leadup_analysis(
    X: pd.DataFrame,
    y_early: np.ndarray | pd.Series,
    eval_labels: np.ndarray,
    patient_ids: np.ndarray,
    iculos: np.ndarray,
    output_dir: str | Path,
    best_params: dict | None = None,
    skip_if_exists: bool = True,
) -> dict:
    """Compute IV + SHAP per lead-time bin and write CSVs.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (rows aligned with the other arrays).
    y_early : array-like
        Early-window training labels (used only to fit the SHAP model).
    eval_labels : np.ndarray
        Original ``SepsisLabel`` per row (used to find onset).
    patient_ids : np.ndarray
    iculos : np.ndarray
    output_dir : path-like
        Directory where ``iv_by_lead_time.csv`` and ``shap_by_lead_time.csv``
        will be written.
    best_params : dict, optional
        XGBoost hyperparameters for the SHAP fit. Falls back to a sensible
        regularized default if omitted.
    skip_if_exists : bool
        If True and both output CSVs already exist, the analysis is skipped
        and the cached results are loaded.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iv_path = output_dir / "iv_by_lead_time.csv"
    shap_path = output_dir / "shap_by_lead_time.csv"

    if skip_if_exists and iv_path.exists() and shap_path.exists():
        print(f"  [leadup] cached outputs found in {output_dir} — skipping recompute")
        iv_df = pd.read_csv(iv_path, index_col=0)
        shap_df = pd.read_csv(shap_path, index_col=0)
        return {"iv": iv_df, "shap": shap_df, "lead_bins": None}

    print("  [leadup] computing hours_to_onset and lead-time bins ...")
    hours_to_onset = compute_hours_to_onset(patient_ids, iculos, eval_labels)
    lead_bins = bin_lead_time(hours_to_onset)

    counts = pd.Series(lead_bins).value_counts()
    print(f"    Lead-time bin counts: {counts.to_dict()}")

    if skip_if_exists and iv_path.exists():
        print(f"    IV cache found at {iv_path} — loading")
        iv_df = pd.read_csv(iv_path, index_col=0)
    else:
        print("  [leadup] computing IV per lead-time bin ...")
        iv_df = iv_per_lead_time(X, lead_bins)
        iv_df.to_csv(iv_path)
        print(f"    Saved {iv_path} ({iv_df.shape[0]} features × {iv_df.shape[1]} bins)")

    shap_df: pd.DataFrame | None = None
    try:
        print("  [leadup] fitting XGBoost for SHAP analysis ...")
        from xgboost import XGBClassifier  # local import to keep test collection light
        params = best_params or _DEFAULT_XGB_PARAMS
        y_arr = np.asarray(y_early)
        n_pos = float((y_arr == 1).sum())
        n_neg = float((y_arr == 0).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        model = XGBClassifier(
            **params,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
        )
        model.fit(X, y_arr)

        print("  [leadup] computing SHAP per lead-time bin ...")
        shap_df = shap_per_lead_time(model, X, lead_bins, list(X.columns))
        shap_df.to_csv(shap_path)
        print(f"    Saved {shap_path} ({shap_df.shape[0]} features × {shap_df.shape[1]} bins)")
    except ImportError as exc:
        # SHAP can't be loaded in this environment; IV-only is still useful.
        print(f"  [leadup] SHAP step skipped: {exc}")

    return {"iv": iv_df, "shap": shap_df, "lead_bins": lead_bins}
