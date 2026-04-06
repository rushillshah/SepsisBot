"""Early warning timing analysis for sepsis predictions.

Analyzes WHEN the model flags sepsis relative to actual onset:
- Hourly risk trajectories
- Daily max risk aggregation
- Early warning timing statistics
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import LABEL_COL, TIME_COL


def hourly_risk_trajectory(df: pd.DataFrame, y_prob: np.ndarray) -> pd.DataFrame:
    """Compute risk scores relative to sepsis onset for each sepsis patient."""
    work = df[["patient_id", TIME_COL, LABEL_COL]].copy()
    work["risk_score"] = np.asarray(y_prob)

    rows = []
    for pid, group in work.groupby("patient_id"):
        group = group.sort_values(TIME_COL)
        if group[LABEL_COL].max() == 0:
            continue
        onset_hour = group.loc[group[LABEL_COL] == 1, TIME_COL].iloc[0]
        group = group.copy()
        group["hours_before_onset"] = onset_hour - group[TIME_COL]
        before_onset = group[group["hours_before_onset"] >= 0]
        for _, row in before_onset.iterrows():
            rows.append({
                "patient_id": pid,
                "hours_before_onset": int(row["hours_before_onset"]),
                "risk_score": float(row["risk_score"]),
            })

    return pd.DataFrame(rows)


def daily_max_risk(df: pd.DataFrame, y_prob: np.ndarray) -> pd.DataFrame:
    """Aggregate hourly predictions into daily max risk scores."""
    work = df[["patient_id", TIME_COL, LABEL_COL]].copy()
    work["risk_score"] = np.asarray(y_prob)
    work["icu_day"] = work[TIME_COL].apply(lambda x: max(1, math.ceil(x / 24)))

    patient_sepsis = work.groupby("patient_id")[LABEL_COL].max().reset_index()
    patient_sepsis.columns = ["patient_id", "sepsis_actual"]

    onset_days = (
        work[work[LABEL_COL] == 1]
        .groupby("patient_id")["icu_day"]
        .min()
        .reset_index()
    )
    onset_days.columns = ["patient_id", "sepsis_onset_day"]

    daily = (
        work.groupby(["patient_id", "icu_day"])["risk_score"]
        .max()
        .reset_index()
    )
    daily.columns = ["patient_id", "icu_day", "max_daily_risk"]
    daily = daily.merge(patient_sepsis, on="patient_id")
    daily = daily.merge(onset_days, on="patient_id", how="left")

    return daily


def early_warning_summary(
    df: pd.DataFrame, y_prob: np.ndarray, threshold: float = 0.5,
) -> dict:
    """Summary statistics for how early the model flags sepsis."""
    work = df[["patient_id", TIME_COL, LABEL_COL]].copy()
    work["risk_score"] = np.asarray(y_prob)

    results = []
    for pid, group in work.groupby("patient_id"):
        group = group.sort_values(TIME_COL)
        if group[LABEL_COL].max() == 0:
            continue
        onset_hour = group.loc[group[LABEL_COL] == 1, TIME_COL].iloc[0]
        alerts = group[group["risk_score"] >= threshold]
        if len(alerts) == 0:
            results.append({"patient_id": pid, "hours_before": np.nan, "caught": False})
        else:
            first_alert_hour = alerts[TIME_COL].iloc[0]
            hours_before = onset_hour - first_alert_hour
            results.append({"patient_id": pid, "hours_before": float(hours_before), "caught": True})

    rdf = pd.DataFrame(results)
    total = len(rdf)
    caught = rdf[rdf["caught"]]
    not_caught = total - len(caught)
    hours = caught["hours_before"].dropna()

    return {
        "total_sepsis_patients": total,
        "median_hours_before_onset": float(hours.median()) if len(hours) > 0 else 0.0,
        "mean_hours_before_onset": float(hours.mean()) if len(hours) > 0 else 0.0,
        "pct_caught_6h_before": float((hours >= 6).sum() / total * 100) if total > 0 else 0.0,
        "pct_caught_12h_before": float((hours >= 12).sum() / total * 100) if total > 0 else 0.0,
        "pct_caught_24h_before": float((hours >= 24).sum() / total * 100) if total > 0 else 0.0,
        "pct_never_caught": float(not_caught / total * 100) if total > 0 else 0.0,
        "threshold_used": threshold,
    }


def plot_average_risk_trajectory(
    trajectory_df: pd.DataFrame, save_path: str | None = None,
) -> None:
    """Plot average risk score over hours-before-onset for sepsis patients."""
    grouped = trajectory_df.groupby("hours_before_onset")["risk_score"]
    means = grouped.mean()
    stds = grouped.std().fillna(0)

    hours = means.index.values
    idx = np.argsort(-hours)
    hours_sorted = hours[idx]
    means_sorted = means.values[idx]
    stds_sorted = stds.values[idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(-hours_sorted, means_sorted, color="#d62728", linewidth=2, label="Mean Risk Score")
    ax.fill_between(
        -hours_sorted,
        np.clip(means_sorted - stds_sorted, 0, 1),
        np.clip(means_sorted + stds_sorted, 0, 1),
        alpha=0.2, color="#d62728", label="+/- 1 Std",
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Threshold (0.5)")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax.set_xlabel("Hours Relative to Sepsis Onset (0 = onset)")
    ax.set_ylabel("Risk Score")
    ax.set_title("Average Risk Trajectory for Sepsis Patients")
    ax.legend()
    ax.set_xlim(-min(72, hours_sorted.max()), 5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_daily_risk_comparison(
    daily_df: pd.DataFrame, save_path: str | None = None,
) -> None:
    """Box plot comparing daily max risk for sepsis vs non-sepsis patients."""
    plot_df = daily_df[daily_df["icu_day"] <= 14].copy()
    plot_df["group"] = plot_df["sepsis_actual"].map({True: "Sepsis", False: "No Sepsis", 1: "Sepsis", 0: "No Sepsis"})

    fig, ax = plt.subplots(figsize=(14, 5))
    days = sorted(plot_df["icu_day"].unique())

    sepsis_data = [plot_df[(plot_df["icu_day"] == d) & (plot_df["group"] == "Sepsis")]["max_daily_risk"].values for d in days]
    no_sepsis_data = [plot_df[(plot_df["icu_day"] == d) & (plot_df["group"] == "No Sepsis")]["max_daily_risk"].values for d in days]

    positions_s = [d - 0.2 for d in days]
    positions_n = [d + 0.2 for d in days]

    bp1 = ax.boxplot(sepsis_data, positions=positions_s, widths=0.35, patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(no_sepsis_data, positions=positions_n, widths=0.35, patch_artist=True, showfliers=False)

    for patch in bp1["boxes"]:
        patch.set_facecolor("#d62728")
        patch.set_alpha(0.6)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.6)

    ax.set_xticks(days)
    ax.set_xticklabels([str(d) for d in days])
    ax.set_xlabel("ICU Day")
    ax.set_ylabel("Max Daily Risk Score")
    ax.set_title("Daily Max Risk Score: Sepsis vs Non-Sepsis Patients")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Sepsis", "No Sepsis"])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
