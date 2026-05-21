"""Generate a feature dictionary CSV.

Produces docs/feature_dictionary.csv with two sections:
  1) All 41 raw PhysioNet CinC 2019 columns with descriptions.
  2) Top 100 engineered features by Information Value (IV) with descriptions
     parsed from the project's naming conventions.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IV_PATH = ROOT / "data" / "processed" / "feature_analysis" / "iv_ranking.csv"
OUT_PATH = ROOT / "docs" / "feature_dictionary.csv"


# PhysioNet CinC 2019 raw column descriptions
# Source: https://physionet.org/content/challenge-2019/1.0.0/ (challenge data dictionary)
RAW_COLS: dict[str, tuple[str, str, str]] = {
    # name: (category, units, description)
    "HR": ("Vital", "beats/min", "Heart rate"),
    "O2Sat": ("Vital", "%", "Pulse oximetry (peripheral oxygen saturation)"),
    "Temp": ("Vital", "deg C", "Body temperature"),
    "SBP": ("Vital", "mm Hg", "Systolic blood pressure"),
    "MAP": ("Vital", "mm Hg", "Mean arterial pressure"),
    "DBP": ("Vital", "mm Hg", "Diastolic blood pressure"),
    "Resp": ("Vital", "breaths/min", "Respiration rate"),
    "EtCO2": ("Vital", "mm Hg", "End-tidal carbon dioxide"),
    "BaseExcess": ("Lab (blood gas)", "mmol/L", "Excess bicarbonate; deviation of buffer base from normal"),
    "HCO3": ("Lab (blood gas)", "mmol/L", "Bicarbonate"),
    "FiO2": ("Lab (blood gas)", "fraction", "Fraction of inspired oxygen"),
    "pH": ("Lab (blood gas)", "—", "Arterial pH"),
    "PaCO2": ("Lab (blood gas)", "mm Hg", "Partial pressure of CO2 from arterial blood"),
    "SaO2": ("Lab (blood gas)", "%", "Oxygen saturation from arterial blood"),
    "AST": ("Lab (chemistry)", "IU/L", "Aspartate aminotransferase (liver enzyme)"),
    "BUN": ("Lab (chemistry)", "mg/dL", "Blood urea nitrogen (kidney function)"),
    "Alkalinephos": ("Lab (chemistry)", "IU/L", "Alkaline phosphatase"),
    "Calcium": ("Lab (chemistry)", "mg/dL", "Serum calcium"),
    "Chloride": ("Lab (chemistry)", "mmol/L", "Serum chloride"),
    "Creatinine": ("Lab (chemistry)", "mg/dL", "Serum creatinine (kidney function)"),
    "Bilirubin_direct": ("Lab (chemistry)", "mg/dL", "Direct (conjugated) bilirubin"),
    "Glucose": ("Lab (chemistry)", "mg/dL", "Serum glucose"),
    "Lactate": ("Lab (chemistry)", "mmol/L", "Lactic acid; tissue hypoperfusion marker, classic sepsis biomarker"),
    "Magnesium": ("Lab (chemistry)", "mmol/dL", "Serum magnesium"),
    "Phosphate": ("Lab (chemistry)", "mg/dL", "Serum phosphate"),
    "Potassium": ("Lab (chemistry)", "mmol/L", "Serum potassium"),
    "Bilirubin_total": ("Lab (chemistry)", "mg/dL", "Total bilirubin"),
    "TroponinI": ("Lab (cardiac)", "ng/mL", "Troponin I (cardiac injury marker)"),
    "Hct": ("Lab (CBC)", "%", "Hematocrit"),
    "Hgb": ("Lab (CBC)", "g/dL", "Hemoglobin"),
    "PTT": ("Lab (coag)", "seconds", "Partial thromboplastin time (clotting)"),
    "WBC": ("Lab (CBC)", "count*1000/uL", "White blood cell count (immune response)"),
    "Fibrinogen": ("Lab (coag)", "mg/dL", "Fibrinogen"),
    "Platelets": ("Lab (CBC)", "count*1000/uL", "Platelet count"),
    "Age": ("Demographic", "years", "Patient age (>=100 truncated to 100 in source)"),
    "Gender": ("Demographic", "0/1", "Female=0, Male=1"),
    "Unit1": ("Demographic", "0/1", "Admin ID for ICU unit (MICU). EXCLUDED from features (site confounder)."),
    "Unit2": ("Demographic", "0/1", "Admin ID for ICU unit (SICU). EXCLUDED from features (site confounder)."),
    "HospAdmTime": ("Demographic", "hours", "Hours between hospital admit and ICU admit. EXCLUDED (site confounder)."),
    "ICULOS": ("Time", "hours", "ICU length-of-stay so far. EXCLUDED from features (circular/leaky)."),
    "SepsisLabel": ("Outcome", "0/1", "Target. Flips to 1 starting 6h before Sepsis-3 clinical onset (t_sepsis - 6); 0 for non-sepsis patients."),
}


# Curated descriptions for engineered base concepts and clinical scores.
SPECIAL_FEATURES: dict[str, str] = {
    "early_warning_score": (
        "MEWS (Modified Early Warning Score). Project-engineered, NOT in PhysioNet raw data. "
        "Sum of 4 components: HR (0-3), SBP (0-3), Resp (0-3), Temp (0 or 2). Higher = more deteriorated. "
        "Standard ward-vitals deterioration score; range ~0-11."
    ),
    "inflammation_score": (
        "SIRS criteria count (0-4). Project-engineered. Sums 4 binary flags: "
        "Temp >38 or <36C; HR>90; Resp>20; WBC>12 or <4."
    ),
    "sepsis_screen_score": (
        "Modified qSOFA (0-2). Project-engineered. Sums Resp>=22 and SBP<=100. "
        "(GCS component omitted — not in dataset.)"
    ),
    "shock_index": (
        "HR / SBP. Project-engineered. Elevated values suggest hemodynamic instability."
    ),
    "lactate_bp_ratio": (
        "Lactate / MAP. Project-engineered. High lactate with low MAP indicates shock."
    ),
}


def describe_feature(name: str) -> str:
    """Decode an engineered feature name into a human description."""
    if name in SPECIAL_FEATURES:
        return SPECIAL_FEATURES[name]

    base = name
    suffix_chain: list[str] = []

    # Peel suffixes in the order they are applied by the pipeline.
    # Order: rolling stat applied last -> peel first.
    for stat_suffix, label in [
        ("_avg_6h", "6h rolling mean"),
        ("_max_6h", "6h rolling max"),
        ("_min_6h", "6h rolling min"),
        ("_std_6h", "6h rolling stdev"),
    ]:
        if base.endswith(stat_suffix):
            base = base[: -len(stat_suffix)]
            suffix_chain.append(label)
            break

    if base.endswith("_baseline_dev"):
        base = base[: -len("_baseline_dev")]
        suffix_chain.append(
            "deviation from causal CUSUM dynamic baseline (real-time changepoint detection over the patient's history so far)"
        )
    elif base.endswith("_deviation_from_normal"):
        base = base[: -len("_deviation_from_normal")]
        suffix_chain.append("signed distance from age/gender-stratified clinical normal-range midpoint")
    elif base.endswith("_hours_since"):
        base = base[: -len("_hours_since")]
        suffix_chain.append("hours since last measurement of this variable (-1 before first draw)")
    elif base.endswith("_measured"):
        base = base[: -len("_measured")]
        suffix_chain.append("binary missingness flag (1 if measured this hour, else 0)")
    elif base.endswith("_hourly_change"):
        base = base[: -len("_hourly_change")]
        suffix_chain.append("hour-over-hour delta")

    raw_desc = (
        RAW_COLS[base][2]
        if base in RAW_COLS
        else SPECIAL_FEATURES.get(base, f"engineered base '{base}'")
    )

    if not suffix_chain:
        return raw_desc

    # Build a left-to-right English chain, e.g. "6h rolling max of [deviation from baseline of [Lactate]]".
    # We applied peeling outer->inner, so reverse to read inner->outer.
    desc = raw_desc
    for transform in reversed(suffix_chain):
        desc = f"{transform} of [{desc}]"
    return desc


def category_for_engineered(name: str) -> str:
    if name in SPECIAL_FEATURES:
        return "Clinical score (engineered)"
    if name.endswith(("_avg_6h", "_max_6h", "_min_6h", "_std_6h")):
        return "Rolling 6h statistic (engineered)"
    if "_baseline_dev" in name:
        return "Dynamic baseline deviation (engineered, causal CUSUM)"
    if name.endswith("_deviation_from_normal"):
        return "Distance from normal range (engineered)"
    if name.endswith("_hours_since"):
        return "Time-since-measurement (engineered)"
    if name.endswith("_measured"):
        return "Missingness flag (engineered)"
    if name.endswith("_hourly_change"):
        return "Hourly delta (engineered)"
    return "Raw" if name in RAW_COLS else "Engineered"


def main() -> None:
    iv_df = pd.read_csv(IV_PATH).head(100)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "rank", "name", "category", "iv", "iv_strength", "units", "description"])

        # Section 1: raw columns
        for name, (category, units, desc) in RAW_COLS.items():
            writer.writerow(["raw_data_column", "", name, category, "", "", units, desc])

        # Section 2: top 100 IV features
        for rank, row in enumerate(iv_df.itertuples(index=False), start=1):
            name = row.feature
            iv = f"{row.iv:.6f}"
            strength = row.iv_strength
            writer.writerow([
                "top_100_by_iv",
                rank,
                name,
                category_for_engineered(name),
                iv,
                strength,
                "",
                describe_feature(name),
            ])

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
