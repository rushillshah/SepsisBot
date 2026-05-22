"""Feature engineering for the sepsis prediction pipeline.

Adds temporal features (rolling window statistics, hour-over-hour deltas)
to the imputed per-hour snapshot data, then assembles the final feature
matrix for model training.

Expected input: a DataFrame that has already passed through
``imputation.impute`` — forward-filled values with ``{col}_measured``
and ``{col}_hours_since`` columns present.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    AGE_BINS,
    AGE_BIN_LABELS,
    ALL_FEATURE_COLS,
    CLINICAL_NORMAL_RANGES,
    CLINICAL_SCORE_COLS,
    CUSUM_SLACK,
    CUSUM_THRESHOLD,
    DEMOGRAPHIC_COLS,
    DYNAMIC_BASELINE_FEATURES,
    EARLY_LABEL_COL,
    EARLY_LABEL_EXTRA_HOURS,
    EXCLUDED_FEATURES,
    LABEL_COL,
    LAB_COLS,
    NORMAL_RANGE_COLS,
    ROLLING_COLS,
    ROLLING_STAT_SUFFIXES,
    ROLLING_STATS,
    ROLLING_WINDOW_HOURS,
    TIME_COL,
    VITAL_COLS,
)

# Deduplicate ROLLING_COLS (MAP appears in both VITAL_COLS and the extras).
_ROLLING_COLS_UNIQUE: list[str] = list(dict.fromkeys(ROLLING_COLS))

# Columns that are metadata / identifiers, never features.
_DROP_COLS = {"patient_id", "hospital", LABEL_COL} | set(EXCLUDED_FEATURES)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute backward-looking rolling window statistics per patient.

    For each column in ``ROLLING_COLS`` and each statistic in
    ``ROLLING_STATS``, a new column ``{col}_roll_{stat}`` is added.
    Rolling windows are anchored to the right (no future data leakage)
    and use ``min_periods=1`` so early hours still receive values.

    Parameters
    ----------
    df : pd.DataFrame
        Imputed hourly data with a ``patient_id`` column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with rolling-feature columns appended.
    """
    result = df.copy()
    grouped = result.groupby("patient_id")

    for col in _ROLLING_COLS_UNIQUE:
        if col not in result.columns:
            continue
        rolling = grouped[col].rolling(
            window=ROLLING_WINDOW_HOURS,
            min_periods=1,
        )
        for stat in ROLLING_STATS:
            suffix = ROLLING_STAT_SUFFIXES[stat]
            col_name = f"{col}_{suffix}"
            # .agg / getattr dispatches to the correct rolling method.
            computed = getattr(rolling, stat)()
            # rolling().stat() produces a MultiIndex (patient_id, row);
            # droplevel + sort restores alignment with the original index.
            values = computed.droplevel("patient_id").sort_index()
            # std with fewer than 2 observations returns NaN; a single
            # observation has zero variability, so 0.0 is correct.
            if stat == "std":
                values = values.fillna(0.0)
            result[col_name] = values

    return result


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hour-over-hour deltas for vital-sign columns per patient.

    For each column in ``VITAL_COLS``, a new column ``{col}_hourly_change``
    holds the difference from the previous hour within the same patient.  The
    first hour of each patient stay receives ``NaN`` (filled to ``0.0``
    so downstream models can consume the column directly).

    Parameters
    ----------
    df : pd.DataFrame
        Imputed hourly data with a ``patient_id`` column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with delta columns appended.
    """
    result = df.copy()

    for col in VITAL_COLS:
        result[f"{col}_hourly_change"] = (
            result.groupby("patient_id")[col]
            .diff()
            .fillna(0.0)
        )

    return result


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of feature column names in the matrix.

    Includes every column in *df* except ``patient_id``, ``hospital``,
    and ``SepsisLabel``.  Useful for model interpretation and SHAP
    explanations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that has been through the full feature-engineering
        pipeline (rolling + trend features added).

    Returns
    -------
    list[str]
        Column names in the same order they appear in *df*.
    """
    return [c for c in df.columns if c not in _DROP_COLS]


def _cusum_changepoint(values: np.ndarray, k: float = CUSUM_SLACK, h: float = CUSUM_THRESHOLD) -> int:
    """Find the CUSUM changepoint index in a 1D array of non-NaN values.

    Returns the index of the first value where the cumulative sum
    exceeds the threshold h. Returns len(values) if no change detected.
    """
    if len(values) < 6:
        return len(values)  # not enough data for baseline

    mu = np.mean(values[:6])
    sigma = np.std(values[:6])
    if sigma == 0:
        sigma = 1.0

    s_high = 0.0
    s_low = 0.0
    for i in range(6, len(values)):
        z = (values[i] - mu) / sigma
        s_high = max(0.0, s_high + z - k)
        s_low = max(0.0, s_low - z - k)
        if s_high > h or s_low > h:
            return i
    return len(values)


def _causal_cusum_baselines(
    values: np.ndarray,
    k: float = CUSUM_SLACK,
    h: float = CUSUM_THRESHOLD,
) -> np.ndarray:
    """Per-hour causal baseline via online CUSUM.

    For each index t, returns the baseline that would be available *using
    only values[:t+1]* — i.e., what a real-time bedside system could
    compute. No future data leaks into earlier hours.

    Algorithm:
      * Hours 0..5: baseline is the running mean of values seen so far.
        (CUSUM needs ≥6 points to seed μ, σ.)
      * Hour ≥6 with no changepoint detected yet: baseline is the running
        mean of all values observed up to and including hour t.
      * Hour ≥6 with a changepoint detected at hour cp ≤ t: baseline is
        frozen at mean(values[:cp]) — the pre-deterioration mean.

    Reference μ, σ (used by the CUSUM normalizer) are seeded from the
    first 6 values and never updated — matching the original
    ``_cusum_changepoint``'s reference semantics.
    """
    n = len(values)
    if n == 0:
        return np.zeros(0)

    baselines = np.zeros(n)

    # Hours 0..min(n,6)-1: running mean only.
    running_sum = 0.0
    for t in range(min(n, 6)):
        running_sum += values[t]
        baselines[t] = running_sum / (t + 1)

    if n < 6:
        return baselines

    # Reference μ, σ frozen from first 6 values.
    mu = float(np.mean(values[:6]))
    sigma = float(np.std(values[:6]))
    if sigma == 0.0:
        sigma = 1.0

    s_high = 0.0
    s_low = 0.0
    frozen_baseline: float | None = None

    for t in range(6, n):
        if frozen_baseline is None:
            z = (values[t] - mu) / sigma
            s_high = max(0.0, s_high + z - k)
            s_low = max(0.0, s_low - z - k)
            if s_high > h or s_low > h:
                # Changepoint detected at index t. Baseline frozen at mean
                # of strictly pre-change values, i.e. values[:t]. running_sum
                # currently holds sum(values[:t]) (we have not added values[t]
                # yet on this iteration), so divide by t.
                frozen_baseline = running_sum / t if t > 0 else 0.0
                baselines[t] = frozen_baseline
            else:
                running_sum += values[t]
                baselines[t] = running_sum / (t + 1)
        else:
            baselines[t] = frozen_baseline

    return baselines


def add_dynamic_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(patient, hour) deviation from a *causal* CUSUM baseline.

    For each feature in DYNAMIC_BASELINE_FEATURES, computes ``feature -
    causal_baseline(t)`` where ``causal_baseline(t)`` is what a real-time
    system would have established using only data up to and including hour
    ``t``. See :func:`_causal_cusum_baselines` for the algorithm.

    Replaces the previous (leaky) global per-patient CUSUM, which computed
    a single changepoint over the entire stay and then subtracted that
    baseline from every row — peeking into future data at early hours.
    """
    result = df.copy()
    sorted_df = result.sort_values(["patient_id", TIME_COL])

    for feature in DYNAMIC_BASELINE_FEATURES:
        if feature not in result.columns:
            continue

        deviations = pd.Series(0.0, index=result.index, dtype=float)
        for pid, group in sorted_df.groupby("patient_id"):
            values = group[feature].to_numpy(dtype=float, na_value=np.nan)
            mask_valid = ~np.isnan(values)
            if not mask_valid.any():
                continue
            filled = np.where(mask_valid, values, 0.0)
            baselines = _causal_cusum_baselines(filled)
            dev = filled - baselines
            dev[~mask_valid] = 0.0
            deviations.loc[group.index] = dev

        result[f"{feature}_baseline_dev"] = deviations.fillna(0.0)

    return result






def _mews_hr(hr: pd.Series) -> pd.Series:
    """MEWS heart rate component: 0-3."""
    return pd.Series(np.select(
        [hr >= 130, (hr >= 111) & (hr <= 129), (hr >= 101) & (hr <= 110),
         (hr >= 51) & (hr <= 100), (hr >= 41) & (hr <= 50), hr <= 40],
        [3, 2, 1, 0, 1, 2],
        default=0,
    ), index=hr.index, dtype=float)


def _mews_sbp(sbp: pd.Series) -> pd.Series:
    """MEWS systolic BP component: 0-3."""
    return pd.Series(np.select(
        [sbp <= 70, (sbp >= 71) & (sbp <= 80), (sbp >= 81) & (sbp <= 100),
         (sbp >= 101) & (sbp <= 199), sbp >= 200],
        [3, 2, 1, 0, 2],
        default=0,
    ), index=sbp.index, dtype=float)


def _mews_resp(resp: pd.Series) -> pd.Series:
    """MEWS respiratory rate component: 0-3."""
    return pd.Series(np.select(
        [resp <= 8, (resp >= 9) & (resp <= 14), (resp >= 15) & (resp <= 20),
         (resp >= 21) & (resp <= 29), resp >= 30],
        [3, 0, 1, 2, 3],
        default=0,
    ), index=resp.index, dtype=float)


def _mews_temp(temp: pd.Series) -> pd.Series:
    """MEWS temperature component: 0 or 2."""
    return pd.Series(
        np.where((temp >= 38.5) | (temp < 35), 2.0, 0.0),
        index=temp.index, dtype=float,
    )


def add_clinical_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute validated clinical scoring features from existing vitals/labs.

    Adds 5 columns: inflammation_score, sepsis_screen_score, shock_index,
    early_warning_score, lactate_bp_ratio. Computed from raw (imputed) values — call BEFORE
    rolling/trend features so these scores also get temporal stats.

    Returns a new DataFrame (no mutation).
    """
    result = df.copy()

    hr = result.get("HR", pd.Series(dtype=float))
    temp = result.get("Temp", pd.Series(dtype=float))
    resp = result.get("Resp", pd.Series(dtype=float))
    sbp = result.get("SBP", pd.Series(dtype=float))
    wbc = result.get("WBC", pd.Series(dtype=float))
    map_col = result.get("MAP", pd.Series(dtype=float))
    lactate = result.get("Lactate", pd.Series(dtype=float))

    # SIRS Score (0-4): 4 binary criteria summed
    sirs = (
        ((temp > 38) | (temp < 36)).astype(float)
        + (hr > 90).astype(float)
        + (resp > 20).astype(float)
        + ((wbc > 12) | (wbc < 4)).astype(float)
    )
    result["inflammation_score"] = sirs.fillna(0)

    # Modified qSOFA (0-2): 2 of 3 components (missing GCS)
    qsofa = (
        (resp >= 22).astype(float)
        + (sbp <= 100).astype(float)
    )
    result["sepsis_screen_score"] = qsofa.fillna(0)

    # Shock Index: HR / SBP
    with np.errstate(divide="ignore", invalid="ignore"):
        si = hr / sbp.replace(0, np.nan)
    result["shock_index"] = si.clip(0, 5).fillna(0)

    # Early Warning Score (MEWS: 0-8ish)
    result["early_warning_score"] = (
        _mews_hr(hr) + _mews_sbp(sbp) + _mews_resp(resp) + _mews_temp(temp)
    ).fillna(0)

    # Lactate / blood pressure ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = lactate / map_col.replace(0, np.nan)
    result["lactate_bp_ratio"] = lm.clip(0, 10).fillna(0)

    return result


def add_normal_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add age/gender-stratified clinical normal range features.

    For each column in NORMAL_RANGE_COLS, computes:
      - {col}_above_normal  (binary): value exceeds age/gender upper bound
      - {col}_below_normal  (binary): value below age/gender lower bound
      - {col}_deviation_from_normal (float): signed distance from range midpoint,
        normalized by half-range width (z-score-like: ±1.0 = at boundary)
      - {col}_abs_deviation_from_normal (float): magnitude of abnormality
        (|deviation_from_normal|). 0 when at midpoint of normal, grows as
        the value moves away from normal in either direction.
      - {col}_drift_from_normal (float): per-patient 1-hour change in
        ``abs_deviation_from_normal``. POSITIVE = moving away from normal
        (worsening); NEGATIVE = trending back toward normal (improving).
      - {col}_drift_from_normal_6h (float): per-patient 6-hour change in
        ``abs_deviation_from_normal`` (t minus t-6). Captures slower
        trends — e.g. a lactate that fails to normalize over 6 hours.
        Same sign convention as the 1h drift.

    Uses clinically established reference ranges from CLINICAL_NORMAL_RANGES.
    No data leakage — only uses current-row value + static demographics, and
    drift is a backward-looking diff within the same patient.
    """
    age_bin = pd.cut(
        df["Age"],
        bins=AGE_BINS,
        labels=AGE_BIN_LABELS,
        right=False,
    ).fillna("18-40")

    gender = df["Gender"].fillna(1).astype(int)
    pid = df["patient_id"]

    new_cols: dict[str, np.ndarray | pd.Series] = {}

    for col in NORMAL_RANGE_COLS:
        if col not in df.columns:
            continue

        ranges = CLINICAL_NORMAL_RANGES.get(col)
        if ranges is None:
            continue

        default = ranges.get(("18-40", 1), (0.0, 0.0))
        keys = list(zip(age_bin.astype(str), gender))
        low = np.array([ranges.get(k, default)[0] for k in keys], dtype=np.float64)
        high = np.array([ranges.get(k, default)[1] for k in keys], dtype=np.float64)

        values = df[col].values.astype(np.float64)
        midpoint = (low + high) / 2.0
        half_range = (high - low) / 2.0
        half_range[half_range == 0] = 1.0

        deviation = (values - midpoint) / half_range
        abs_deviation = np.abs(deviation)

        new_cols[f"{col}_above_normal"] = (values > high).astype(np.int8)
        new_cols[f"{col}_below_normal"] = (values < low).astype(np.int8)
        new_cols[f"{col}_deviation_from_normal"] = deviation
        new_cols[f"{col}_abs_deviation_from_normal"] = abs_deviation

        abs_dev_series = pd.Series(abs_deviation, index=df.index)
        grouped = abs_dev_series.groupby(pid)
        drift_1h = grouped.diff().fillna(0.0)
        drift_6h = grouped.diff(periods=6).fillna(0.0)
        new_cols[f"{col}_drift_from_normal"] = drift_1h.values
        new_cols[f"{col}_drift_from_normal_6h"] = drift_6h.values

    if not new_cols:
        return df.copy()

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def create_early_label(df: pd.DataFrame, extra_hours: int = EARLY_LABEL_EXTRA_HOURS) -> pd.DataFrame:
    """Create extended positive label window for early detection training.

    For each sepsis patient, sets early_label=1 starting `extra_hours`
    before the first SepsisLabel=1 hour. Non-sepsis patients get 0.
    """
    result = df.copy()
    result[EARLY_LABEL_COL] = 0
    # Find onset hour per patient (first SepsisLabel=1)
    onset = result[result[LABEL_COL] == 1].groupby("patient_id")[TIME_COL].min().rename("_onset_hour")
    result = result.merge(onset, on="patient_id", how="left")
    # Set early_label=1 where ICULOS >= onset - extra_hours
    mask = result["_onset_hour"].notna() & (result[TIME_COL] >= result["_onset_hour"] - extra_hours)
    result.loc[mask, EARLY_LABEL_COL] = 1
    result = result.drop(columns=["_onset_hour"])
    return result


def select_leading_indicators(
    columns: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Partition feature columns into leading indicators vs. drop-list.

    Drops symptom / static-cohort-marker / clinician-action features that
    separate "already-sick" patients without providing genuine lead time,
    keeping only trajectory signals (vital trends, drift-from-baseline,
    rate-of-change, vital-derived deterioration scores).

    Returns
    -------
    tuple[list[str], dict[str, str]]
        ``(keep, drop_reasons)`` where ``keep`` is the retained column list
        and ``drop_reasons`` maps each dropped column to its rationale tag.
    """
    from src.config import (
        LEADING_DROP_FEATURE_PREFIXES,
        LEADING_DROP_LAB_LEVEL_SUFFIXES,
        LEADING_DROP_LEAKAGE_SUFFIXES,
        LEADING_KEEP_TRAJ_SUFFIXES,
    )

    labs = set(LAB_COLS)
    all_suffixes = sorted(
        LEADING_DROP_LAB_LEVEL_SUFFIXES
        + LEADING_KEEP_TRAJ_SUFFIXES
        + LEADING_DROP_LEAKAGE_SUFFIXES,
        key=len,
        reverse=True,
    )

    def split_base(col: str) -> tuple[str, str]:
        for suf in all_suffixes:
            if col.endswith(suf):
                return col[: -len(suf)], suf
        return col, ""

    keep: list[str] = []
    drop_reasons: dict[str, str] = {}
    for col in columns:
        base, suffix = split_base(col)
        if suffix in LEADING_DROP_LEAKAGE_SUFFIXES:
            drop_reasons[col] = "leakage:testing-frequency"
        elif any(col == p or col.startswith(p) for p in LEADING_DROP_FEATURE_PREFIXES):
            drop_reasons[col] = "symptom/intervention"
        elif base in labs and (suffix in LEADING_DROP_LAB_LEVEL_SUFFIXES or suffix == ""):
            drop_reasons[col] = "static-cohort-marker"
        else:
            keep.append(col)
    return keep, drop_reasons


def build_feature_matrix(
    df: pd.DataFrame,
    use_early_label: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble the final feature matrix and label vector.

    Runs the full feature-engineering pipeline (rolling stats + trend
    deltas) and then separates features from labels.

    The returned ``X`` contains:
    * Raw vital and lab values (``ALL_FEATURE_COLS``)
    * Demographic columns (``DEMOGRAPHIC_COLS``)
    * ICU length-of-stay (``ICULOS``)
    * Missingness flags (``{col}_measured`` for each lab column)
    * Time-since-measured (``{col}_hours_since`` for each lab column)
    * Rolling statistics (``{col}_{avg_6h|std_6h|min_6h|max_6h}``)
    * Trend features (``{col}_hourly_change``)

    Parameters
    ----------
    df : pd.DataFrame
        Imputed hourly data with ``patient_id``, ``hospital``, and
        ``SepsisLabel`` columns present.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` where ``X`` is the feature matrix and ``y`` is the
        binary sepsis label.

    Raises
    ------
    KeyError
        If ``SepsisLabel`` is missing from *df*.
    """
    if LABEL_COL not in df.columns:
        raise KeyError(
            f"Label column '{LABEL_COL}' not found in DataFrame. "
            "Ensure the raw data has been loaded correctly."
        )

    enriched = add_clinical_scores(df)
    enriched = add_normal_range_features(enriched)
    enriched = add_rolling_features(enriched)
    enriched = add_trend_features(enriched)

    # Use early_label for training if requested and available
    label_col = EARLY_LABEL_COL if (use_early_label and EARLY_LABEL_COL in enriched.columns) else LABEL_COL
    y = enriched[label_col].copy()

    drop_cols = _DROP_COLS | {EARLY_LABEL_COL}
    X = enriched.drop(columns=[c for c in drop_cols if c in enriched.columns])

    # Fill any remaining NaN in derived features (rolling stats on sparse
    # labs, hours_since sentinel values, etc.) so all models can consume X.
    X = X.fillna(0.0)

    return X, y


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale features using StandardScaler fit on training data only."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    return X_train_scaled, X_val_scaled, scaler
