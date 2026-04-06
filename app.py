"""Sepsis Early Warning — Proof of Concept Dashboard.

A dual-audience dashboard explaining the sepsis prediction model
to both data scientists and medical professionals.

Run with: streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import (
    ALL_FEATURE_COLS,
    DATA_PROCESSED,
    DEMOGRAPHIC_COLS,
    LABEL_COL,
    LAB_COLS,
    TIME_COL,
    VITAL_COLS,
)

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sepsis Early Warning System — PoC",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-left: 4px solid #1f77b4;
    }
    .metric-card-warn {
        border-left-color: #ff7f0e;
    }
    .metric-card-bad {
        border-left-color: #d62728;
    }
    .metric-card-good {
        border-left-color: #2ca02c;
    }
    .explanation-box {
        background: #1a2332;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        border: 1px solid #2a4a6b;
        color: #e0e0e0;
    }
    .clinical-box {
        background: #2a2010;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        border: 1px solid #5a4a20;
        color: #e0d8c8;
    }
    .tech-box {
        background: #102a10;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        border: 1px solid #1a5a1a;
        color: #c8e0c8;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_parquet(name: str) -> pd.DataFrame | None:
    path = DATA_PROCESSED / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_json(name: str) -> dict | None:
    path = DATA_PROCESSED / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_raw_data() -> pd.DataFrame | None:
    return load_parquet("raw_data")


def get_model_metrics() -> dict | None:
    return load_json("model_metrics")


# ── Helpers ───────────────────────────────────────────────────────────────────

def audience_toggle() -> str:
    return st.sidebar.radio(
        "Explain for:",
        ["Clinical / Non-Technical", "Data Science / Technical"],
        index=0,
        key="audience",
    )


def explain(clinical: str, technical: str, audience: str) -> None:
    if audience == "Clinical / Non-Technical":
        st.markdown(f'<div class="clinical-box">{clinical}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="tech-box">{technical}</div>', unsafe_allow_html=True)


def no_data_warning():
    st.error(
        "No processed data found. Run `python run_pipeline.py` first to "
        "load data, train models, and generate results."
    )


# ── Navigation ────────────────────────────────────────────────────────────────

PAGES = [
    "Executive Summary",
    "The Data",
    "How the Model Works",
    "Model Performance",
    "Patient Deep Dive",
    "What Drives Predictions",
    "Limitations & Next Steps",
]

st.sidebar.markdown("### Sepsis Early Warning PoC")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
audience = audience_toggle()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Executive Summary
# ══════════════════════════════════════════════════════════════════════════════

def page_executive_summary():
    st.markdown('<p class="main-header">Sepsis Early Warning System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Proof of Concept — Using PhysioNet CinC 2019 Open Data</p>', unsafe_allow_html=True)
    st.markdown("---")

    explain(
        clinical=(
            "<b>What is this?</b> We built a computer model that looks at a patient's "
            "vital signs (heart rate, blood pressure, temperature, etc.) and lab results "
            "(white blood cell count, lactate, bilirubin, etc.) collected every hour in the ICU, "
            "and tries to predict whether that patient will develop sepsis — <b>up to 6 hours "
            "before</b> doctors would typically diagnose it."
            "<br><br>"
            "<b>Why does this matter?</b> Every hour of delayed sepsis treatment increases "
            "mortality by 4-8%. An early warning system could alert clinicians to intervene sooner."
        ),
        technical=(
            "<b>Binary classification model</b> trained on the PhysioNet/CinC 2019 Challenge dataset "
            "(40K ICU patients, 1.5M hourly observations). Per-hour snapshot approach with 148 engineered "
            "features (rolling stats, missingness indicators, trend deltas). "
            "XGBoost + Logistic Regression baseline. "
            "Train on Hospital A, validate on Hospital B for cross-site generalization."
        ),
        audience=audience,
    )

    st.markdown("")

    # Key numbers
    df = get_raw_data()
    metrics = get_model_metrics()

    if df is not None:
        total_patients = df["patient_id"].nunique()
        sepsis_patients = int(df.groupby("patient_id")[LABEL_COL].max().sum())
        total_hours = len(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients Studied", f"{total_patients:,}")
        c2.metric("Hourly Observations", f"{total_hours:,}")
        c3.metric("Sepsis Cases", f"{sepsis_patients:,}")
        c4.metric("Sepsis Rate", f"{sepsis_patients/total_patients:.1%}")

    # Class imbalance indicator
    if df is not None:
        st.markdown("### Class Imbalance")
        patient_labels = df.groupby("patient_id")[LABEL_COL].max()
        n_sepsis_p = int(patient_labels.sum())
        n_no_sepsis_p = len(patient_labels) - n_sepsis_p

        hour_sepsis = int(df[LABEL_COL].sum())
        hour_no_sepsis = len(df) - hour_sepsis

        c1, c2 = st.columns(2)
        with c1:
            fig_pat = go.Figure(go.Pie(
                labels=["No Sepsis", "Sepsis"],
                values=[n_no_sepsis_p, n_sepsis_p],
                marker_colors=["#1f77b4", "#d62728"],
                textinfo="label+percent",
                hole=0.4,
            ))
            fig_pat.update_layout(title="Patient-Level", height=300, showlegend=False, margin=dict(t=40, b=10))
            st.plotly_chart(fig_pat, use_container_width=True)

        with c2:
            fig_hr = go.Figure(go.Pie(
                labels=["Non-Sepsis Hours", "Sepsis Hours"],
                values=[hour_no_sepsis, hour_sepsis],
                marker_colors=["#1f77b4", "#d62728"],
                textinfo="label+percent",
                hole=0.4,
            ))
            fig_hr.update_layout(title="Hourly Observation-Level", height=300, showlegend=False, margin=dict(t=40, b=10))
            st.plotly_chart(fig_hr, use_container_width=True)

        imbalance_ratio = hour_no_sepsis / hour_sepsis if hour_sepsis > 0 else 0
        explain(
            clinical=(
                f"Sepsis is rare — only <b>{n_sepsis_p:,} out of {total_patients:,} patients</b> ({n_sepsis_p/total_patients:.1%}) "
                f"actually develop it. At the hourly level, only <b>{hour_sepsis/len(df):.1%}</b> of observations are sepsis-positive. "
                f"This <b>{imbalance_ratio:.0f}:1 imbalance</b> makes prediction hard — the model can look "
                f"accurate by just predicting 'no sepsis' every time."
            ),
            technical=(
                f"Class ratio: {imbalance_ratio:.1f}:1 (negative:positive) at the observation level. "
                f"Patient-level prevalence: {n_sepsis_p/total_patients:.1%}. "
                f"Handled via scale_pos_weight in XGBoost and class_weight='balanced' in LR. "
                f"PR-AUC is more informative than AUROC at this prevalence — random baseline PR-AUC ≈ {n_sepsis_p/total_patients:.3f}."
            ),
            audience=audience,
        )

    if metrics is not None:
        st.markdown("### Current Model Performance (Patient-Level CV)")

        c1, c2, c3 = st.columns(3)
        auroc = metrics.get("cv_xgb_auroc", metrics.get("auroc", 0))
        sens = metrics.get("cv_xgb_sensitivity", metrics.get("sensitivity", 0))
        spec = metrics.get("cv_xgb_specificity", metrics.get("specificity", 0))
        auroc_std = metrics.get("cv_xgb_auroc_std", 0)

        c1.metric(
            "AUROC (XGBoost CV)",
            f"{auroc:.3f} +/- {auroc_std:.3f}",
            delta="Above 0.80 target" if auroc >= 0.80 else "Below 0.80 target",
            delta_color="normal" if auroc >= 0.80 else "inverse",
        )
        c2.metric(
            "Sepsis Detection Rate",
            f"{sens:.0%}",
            help="Of patients who actually had sepsis, how many did the model flag? (3-fold CV average)",
        )
        c3.metric(
            "Specificity",
            f"{spec:.0%}",
            help="Of patients who did NOT have sepsis, how many were correctly left alone?",
        )

        explain(
            clinical=(
                f"<b>In plain English:</b> The model correctly identifies about <b>{sens:.0%} of sepsis cases</b> "
                f"with an overall discriminative ability (AUROC) of <b>{auroc:.2f}</b>. "
                f"These results come from 3-fold patient-level cross-validation on both hospitals combined — "
                f"no patient appears in both training and testing."
            ),
            technical=(
                f"<b>CV AUROC: {auroc:.3f} +/- {auroc_std:.3f}</b> (3-fold patient-level stratified CV). "
                f"Site confounders (Unit1, Unit2, HospAdmTime) removed. "
                f"XGBoost with V2 regularization, Platt calibration, StandardScaler. "
                f"Clinical scoring features (SIRS, qSOFA, Shock Index, MEWS, Lactate/MAP) included."
            ),
            audience=audience,
        )

        st.markdown("### Bottom Line")
        st.info(
            "**This PoC proves the concept is viable.** The data pipeline works, the model trains, "
            "and clinically relevant features (lactate, temperature, creatinine) emerge as important predictors. "
            "With real patient data and cross-hospital training, we expect significant improvement."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: The Data
# ══════════════════════════════════════════════════════════════════════════════

def page_the_data():
    st.markdown("## The Data")

    df = get_raw_data()
    if df is None:
        no_data_warning()
        return

    explain(
        clinical=(
            "We used a publicly available dataset from <b>PhysioNet</b> — a trusted medical research database. "
            "It contains de-identified records from <b>40,111 ICU patients</b> across <b>two different hospitals</b>. "
            "For each patient, we have hourly readings of vital signs and periodic lab results, "
            "plus a label indicating whether and when they developed sepsis."
        ),
        technical=(
            "PhysioNet/CinC Challenge 2019. 40,111 patients (20,229 Hospital A + 19,882 Hospital B). "
            "1,543,363 hourly observations. 41 columns: 8 vitals, 26 labs, 6 demographics, 1 label. "
            "PSV format, one file per patient. CC BY 4.0 license."
        ),
        audience=audience,
    )

    # Hospital breakdown
    st.markdown("### Patient Breakdown")
    col1, col2 = st.columns(2)

    for hosp, col in [("A", col1), ("B", col2)]:
        hdf = df[df["hospital"] == hosp]
        n_pat = hdf["patient_id"].nunique()
        n_sepsis = int(hdf.groupby("patient_id")[LABEL_COL].max().sum())
        prev = n_sepsis / n_pat if n_pat > 0 else 0
        avg_stay = hdf.groupby("patient_id")[TIME_COL].max().mean()

        with col:
            st.markdown(f"#### Hospital {hosp}")
            st.metric("Patients", f"{n_pat:,}")
            st.metric("Sepsis Cases", f"{n_sepsis:,} ({prev:.1%})")
            st.metric("Avg Stay", f"{avg_stay:.0f} hours")

    # What we measure
    st.markdown("### What We Measure")

    vital_tab, lab_tab, missing_tab = st.tabs(["Vital Signs", "Lab Values", "Missing Data Challenge"])

    with vital_tab:
        explain(
            clinical=(
                "These are the continuous bedside measurements that ICU monitors collect automatically, "
                "typically every hour or more frequently."
            ),
            technical="8 vital sign channels, low missingness (<10% for most).",
            audience=audience,
        )
        vital_data = []
        for v in VITAL_COLS:
            if v in df.columns:
                missing_pct = df[v].isna().mean() * 100
                vital_data.append({
                    "Measurement": v,
                    "Clinical Name": {
                        "HR": "Heart Rate", "O2Sat": "Oxygen Saturation (SpO2)",
                        "Temp": "Temperature", "SBP": "Systolic Blood Pressure",
                        "MAP": "Mean Arterial Pressure", "DBP": "Diastolic Blood Pressure",
                        "Resp": "Respiratory Rate", "EtCO2": "End-Tidal CO2",
                    }.get(v, v),
                    "Missing %": f"{missing_pct:.1f}%",
                    "Mean": f"{df[v].mean():.1f}" if not df[v].isna().all() else "N/A",
                })
        st.dataframe(pd.DataFrame(vital_data), use_container_width=True, hide_index=True)

    with lab_tab:
        explain(
            clinical=(
                "Lab values require a blood draw and take time to process. They're checked "
                "every few hours (not continuously), which is why there's so much missing data. "
                "<b>This is normal in clinical practice</b> — sicker patients get tested more often."
            ),
            technical=(
                "26 lab channels with 70-99% NaN rates. Imputed via forward-fill + missingness "
                "flags + time-since-measured features. The missingness pattern itself is predictive "
                "(higher measurement frequency correlates with acuity)."
            ),
            audience=audience,
        )
        lab_data = []
        for lab in LAB_COLS:
            if lab in df.columns:
                missing_pct = df[lab].isna().mean() * 100
                lab_data.append({
                    "Lab Test": lab,
                    "Missing %": f"{missing_pct:.1f}%",
                    "Measured Mean": f"{df[lab].mean():.2f}" if not df[lab].isna().all() else "N/A",
                })
        lab_df = pd.DataFrame(lab_data).sort_values("Missing %", ascending=False)
        st.dataframe(lab_df, use_container_width=True, hide_index=True)

    with missing_tab:
        st.markdown("### The Missing Data Problem")
        explain(
            clinical=(
                "In real ICU care, not every test is run every hour. A doctor orders a lactate "
                "level when they're concerned, not on a fixed schedule. So our data has lots of gaps. "
                "<br><br>"
                "<b>How we handle it:</b> We carry the last known value forward (if your WBC was 12 "
                "at 8am and wasn't re-checked until 2pm, we assume it stayed around 12). We also track "
                "<i>how long since the last test</i> as its own signal — because frequent testing "
                "often means the clinical team is worried."
            ),
            technical=(
                "Imputation strategy: (1) compute missingness flags and time-since-measured BEFORE "
                "forward-fill so they reflect actual draw times, (2) forward-fill per patient, "
                "(3) fill remaining NaN with column medians. "
                "The missingness metadata adds ~52 features (26 flags + 26 time-since counters)."
            ),
            audience=audience,
        )

        feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
        missing_pct = (df[feature_cols].isna().mean() * 100).sort_values(ascending=True)
        fig = px.bar(
            x=missing_pct.values,
            y=missing_pct.index,
            orientation="h",
            labels={"x": "Missing (%)", "y": ""},
            title="Missing Data Rate by Variable",
            color=missing_pct.values,
            color_continuous_scale=["#2ca02c", "#ff7f0e", "#d62728"],
        )
        fig.update_layout(height=700, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: How the Model Works
# ══════════════════════════════════════════════════════════════════════════════

def page_how_model_works():
    st.markdown("## How the Model Works")

    explain(
        clinical=(
            "Think of the model as a very fast, very consistent junior resident who looks at "
            "a patient's chart every single hour and asks: <i>\"Based on everything I see right now — "
            "vital signs, lab trends, how often labs are being drawn — is this patient heading toward "
            "sepsis in the next 6 hours?\"</i>"
            "<br><br>"
            "It doesn't replace clinical judgment. It flags patients who might need a closer look."
        ),
        technical=(
            "Per-hour binary classification. Each hourly row is an independent training example. "
            "The target label (SepsisLabel) flips to 1 at t_sepsis - 6h using Sepsis-3 criteria. "
            "This creates a 6-hour early prediction window."
        ),
        audience=audience,
    )

    st.markdown("### The 148 Features")
    st.markdown("For each hour, the model looks at:")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Raw Measurements")
        st.markdown("""
        - Heart rate, BP, SpO2, temp, respiratory rate
        - WBC, lactate, creatinine, bilirubin, platelets
        - 34 total raw values
        """)

    with c2:
        st.markdown("#### Trends Over Time")
        st.markdown("""
        - 6-hour rolling averages, min, max, variability
        - Hour-over-hour changes (is HR rising or falling?)
        - ~68 temporal features
        """)

    with c3:
        st.markdown("#### Testing Patterns")
        st.markdown("""
        - Which labs were drawn this hour?
        - How long since each lab was last checked?
        - ~52 missingness features
        """)

    st.markdown("### Two Models Compared")

    explain(
        clinical=(
            "We trained two different types of models to compare:<br>"
            "<b>Logistic Regression</b> — A simple, transparent model. Like a weighted checklist: "
            "each measurement gets a score, and the scores add up to a risk level.<br>"
            "<b>XGBoost</b> — A more powerful model that can find complex patterns "
            "(e.g., 'high lactate AND rising heart rate AND falling blood pressure = high risk'). "
            "Harder to interpret but usually more accurate."
        ),
        technical=(
            "<b>Logistic Regression:</b> L2 regularization, class_weight='balanced', max_iter=1000. "
            "Convergence warning at 1000 iters — features need scaling.<br>"
            "<b>XGBoost:</b> RandomizedSearchCV (20 iter, 5-fold stratified CV). "
            "Best params: max_depth=7, lr=0.1, n_estimators=500, subsample=0.7, colsample=0.8. "
            "scale_pos_weight computed from class ratio (~13:1)."
        ),
        audience=audience,
    )

    # Visual: training approach
    st.markdown("### Train/Validation Split")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Hospital A (Train)"], y=[20229], name="Hospital A",
        marker_color="#1f77b4", text=["20,229 patients"], textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=["Hospital B (Validate)"], y=[19882], name="Hospital B",
        marker_color="#ff7f0e", text=["19,882 patients"], textposition="inside",
    ))
    fig.update_layout(
        title="Cross-Hospital Validation: Train on one hospital, test on another",
        yaxis_title="Patients",
        showlegend=False,
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    explain(
        clinical=(
            "We deliberately trained the model on data from Hospital A and tested it on Hospital B. "
            "This is harder than testing on the same hospital, but it tells us whether the model "
            "learns <i>real sepsis patterns</i> vs. just memorizing one hospital's quirks."
        ),
        technical=(
            "Cross-hospital holdout is the harshest realistic validation. "
            "Different hospitals have different EHR systems, measurement frequencies, patient "
            "populations, and clinical protocols. Any model that only works on training-site data "
            "won't survive deployment."
        ),
        audience=audience,
    )

    # ── Model Parameters ────────────────────────────────────────────────────
    metrics = get_model_metrics()
    if metrics is not None:
        st.markdown("### Model Parameters")

        xgb_params = metrics.get("xgb_best_params", {})
        lr_params = metrics.get("lr_params", {})

        lstm_params = metrics.get("lstm_params", {})

        col_lr, col_xgb, col_lstm = st.columns(3)

        with col_lr:
            st.markdown("#### Logistic Regression")
            if lr_params:
                param_rows = [
                    {"Parameter": k, "Value": str(v)}
                    for k, v in lr_params.items()
                ]
                st.dataframe(
                    pd.DataFrame(param_rows),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("Parameters not available")

        with col_xgb:
            st.markdown("#### XGBoost (Best from Tuning)")
            if xgb_params:
                param_rows = [
                    {"Parameter": k, "Value": str(v)}
                    for k, v in xgb_params.items()
                ]
                st.dataframe(
                    pd.DataFrame(param_rows),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("Parameters not available")

        with col_lstm:
            st.markdown("#### LSTM (Sequence Model)")
            if lstm_params:
                param_rows = [
                    {"Parameter": k, "Value": str(v)}
                    for k, v in lstm_params.items()
                ]
                st.dataframe(
                    pd.DataFrame(param_rows),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("Run pipeline with LSTM to see parameters")

        explain(
            clinical=(
                "These are the 'settings' for each model — they control how the model learns. "
                "<b>Logistic Regression</b> is set to give equal attention to sepsis and non-sepsis "
                "cases (class_weight='balanced') so it doesn't just ignore the rare sepsis events. "
                "<b>XGBoost</b> was tested with 20 different setting combinations and the best one "
                "was chosen automatically."
            ),
            technical=(
                f"XGBoost tuned via RandomizedSearchCV (20 iterations, 5-fold stratified CV, scoring=roc_auc). "
                f"Best CV AUROC: 0.993 (Hospital A only — overfitting). "
                f"n_features={metrics.get('n_features', 'N/A')}. "
                f"scale_pos_weight={xgb_params.get('scale_pos_weight', 'auto')} "
                f"compensates for ~13:1 class imbalance."
            ),
            audience=audience,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Model Performance
# ══════════════════════════════════════════════════════════════════════════════

def page_model_performance():
    st.markdown("## Model Performance")

    metrics = get_model_metrics()
    if metrics is None:
        no_data_warning()
        return

    def _fmt(val):
        if val is None or val == "N/A":
            return "N/A"
        try:
            return f"{float(val):.4f}"
        except (ValueError, TypeError):
            return str(val)

    # ── Cross-Validation Results ─────────────────────────────────────
    cv_xgb_auroc = metrics.get("cv_xgb_auroc")
    if cv_xgb_auroc is not None:
        st.markdown("### Cross-Validation Results (Hour-Level Metrics)")
        st.caption("3-fold stratified CV on both hospitals — metrics computed per hourly observation")

        cv_table = pd.DataFrame({
            "Metric": ["AUROC", "Gini", "Sensitivity (Recall)", "Specificity", "Precision", "F1", "PR-AUC"],
            "XGBoost (CV)": [
                _fmt(metrics.get("cv_xgb_auroc")),
                _fmt(metrics.get("cv_xgb_gini")),
                _fmt(metrics.get("cv_xgb_sensitivity")),
                _fmt(metrics.get("cv_xgb_specificity")),
                _fmt(metrics.get("cv_xgb_precision")),
                _fmt(metrics.get("cv_xgb_f1")),
                _fmt(metrics.get("cv_xgb_pr_auc")),
            ],
            "Logistic Reg (CV)": [
                _fmt(metrics.get("cv_lr_auroc")),
                _fmt(metrics.get("cv_lr_gini")),
                _fmt(metrics.get("cv_lr_sensitivity")),
                _fmt(metrics.get("cv_lr_specificity")),
                _fmt(metrics.get("cv_lr_precision")),
                _fmt(metrics.get("cv_lr_f1")),
                _fmt(metrics.get("cv_lr_pr_auc")),
            ],
            "Ideal": ["1.0000", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
        })
        st.dataframe(cv_table, use_container_width=True, hide_index=True)

        # CV metric cards
        cv_auroc_std = metrics.get("cv_xgb_auroc_std", 0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CV AUROC (XGBoost)", f"{cv_xgb_auroc:.3f} +/- {cv_auroc_std:.3f}")
        c2.metric("CV Sensitivity", _fmt(metrics.get("cv_xgb_sensitivity")))
        c3.metric("CV Specificity", _fmt(metrics.get("cv_xgb_specificity")))
        c4.metric("CV Precision", _fmt(metrics.get("cv_xgb_precision")))

        explain(
            clinical=(
                "These are the <b>primary results</b> — the model was tested on patients it never "
                "saw during training, with both hospitals mixed in every fold. Site-specific confounders "
                "(hospital ward type, admission timing) have been removed. This is a much "
                "fairer test than the cross-hospital results below."
            ),
            technical=(
                f"5-fold patient-level stratified CV (StratifiedGroupKFold). "
                f"XGBoost with Platt calibration via CalibratedClassifierCV. "
                f"AUROC {cv_xgb_auroc:.3f} +/- {cv_auroc_std:.3f}. "
                f"Features scaled via StandardScaler (fit on train fold only). "
                f"Site confounders (Unit1, Unit2, HospAdmTime) removed from feature set. "
                f"Stronger regularization: max_depth 3-5, min_child_weight 5-20, L1/L2 reg."
            ),
            audience=audience,
        )

        # ── Overfit Check ────────────────────────────────────────────────
        overfit_table = metrics.get("cv_overfit_table")
        if overfit_table:
            st.markdown("### Overfit Check (Train vs Validation AUROC)")
            ot_df = pd.DataFrame(overfit_table)
            # Format for display
            display_df = pd.DataFrame({
                "Fold": ot_df["fold"],
                "XGB Train": ot_df["xgb_train_auroc"].map(lambda x: f"{x:.4f}"),
                "XGB Val": ot_df["xgb_val_auroc"].map(lambda x: f"{x:.4f}"),
                "XGB Gap": ot_df["xgb_gap"].map(lambda x: f"{x:.4f}"),
                "LR Train": ot_df["lr_train_auroc"].map(lambda x: f"{x:.4f}"),
                "LR Val": ot_df["lr_val_auroc"].map(lambda x: f"{x:.4f}"),
                "LR Gap": ot_df["lr_gap"].map(lambda x: f"{x:.4f}"),
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            avg_xgb_gap = ot_df["xgb_gap"].mean()
            avg_lr_gap = ot_df["lr_gap"].mean()

            explain(
                clinical=(
                    f"This table compares how well the model performs on data it <b>trained on</b> vs "
                    f"data it <b>never saw</b>. A large gap means the model memorized training data "
                    f"instead of learning general patterns. "
                    f"<br><br>Average gap: XGBoost {avg_xgb_gap:.3f}, LR {avg_lr_gap:.3f}. "
                    f"{'Gaps are reasonable — the model generalizes well.' if avg_xgb_gap < 0.15 else 'Gaps are large — some overfitting remains.'}"
                ),
                technical=(
                    f"Per-fold train vs val AUROC. XGBoost avg gap: {avg_xgb_gap:.4f}, "
                    f"LR avg gap: {avg_lr_gap:.4f}. "
                    f"Gaps < 0.10 indicate good generalization. "
                    f"Gaps > 0.15 suggest overfitting despite regularization."
                ),
                audience=audience,
            )

    # ── Full CV Metrics Table ────────────────────────────────────────────────
    if cv_xgb_auroc is None:
        st.info("No CV data available. Run `python run_pipeline.py` to generate.")
        return

    st.markdown("### Full CV Metrics — XGBoost vs Logistic Regression")

    cv_metric_rows = ["AUROC", "Sensitivity (Recall)", "Specificity", "Precision", "F1", "Gini", "PR-AUC"]
    xgb_cv = [
        metrics.get("cv_xgb_auroc"), metrics.get("cv_xgb_sensitivity"),
        metrics.get("cv_xgb_specificity"), metrics.get("cv_xgb_precision"),
        metrics.get("cv_xgb_f1"), metrics.get("cv_xgb_gini"), metrics.get("cv_xgb_pr_auc"),
    ]
    lr_cv = [
        metrics.get("cv_lr_auroc"), metrics.get("cv_lr_sensitivity"),
        metrics.get("cv_lr_specificity"), metrics.get("cv_lr_precision"),
        metrics.get("cv_lr_f1"), metrics.get("cv_lr_gini"), metrics.get("cv_lr_pr_auc"),
    ]

    cv_table = pd.DataFrame({
        "Metric": cv_metric_rows,
        "XGBoost (CV)": [_fmt(v) for v in xgb_cv],
        "Logistic Reg (CV)": [_fmt(v) for v in lr_cv],
        "Ideal": ["1.0000"] * 7,
    })
    st.dataframe(cv_table, use_container_width=True, hide_index=True)

    explain(
        clinical=(
            "<b>Reading this table:</b><br>"
            "- <b>AUROC</b> — Overall ability to distinguish sepsis from non-sepsis (1.0 = perfect, 0.5 = coin flip)<br>"
            "- <b>Sensitivity / Recall</b> — Of all actual sepsis cases, what % did we catch?<br>"
            "- <b>Specificity</b> — Of all non-sepsis patients, what % did we correctly leave alone?<br>"
            "- <b>Precision</b> — Of all patients we flagged as sepsis, what % actually had it?<br>"
            "- <b>F1 Score</b> — Balance between precision and recall (higher is better)<br>"
            "- <b>Gini</b> — Another way to express AUC: Gini = 2 x AUC - 1<br>"
            "- <b>PR-AUC</b> — Like AUC but focused on the rare sepsis class"
        ),
        technical=(
            f"3-fold patient-level stratified CV. "
            f"XGBoost AUROC {_fmt(metrics.get('cv_xgb_auroc'))} vs LR {_fmt(metrics.get('cv_lr_auroc'))}. "
            f"Platt calibration applied to XGBoost. Features scaled via StandardScaler. "
            f"Site confounders removed."
        ),
        audience=audience,
    )

    # ── ROC Curve ──────────────────────────────────────────────────────────
    st.markdown("### ROC Curve (from last CV fold)")

    fpr = metrics.get("fpr", [])
    tpr = metrics.get("tpr", [])
    lr_fpr = metrics.get("lr_fpr", [])
    lr_tpr = metrics.get("lr_tpr", [])

    if fpr and tpr:
        fig = go.Figure()
        xgb_auroc_val = metrics.get("cv_xgb_auroc", 0)
        xgb_gini_val = metrics.get("cv_xgb_gini", 0)

        # LSTM curve if available
        lstm_fpr = metrics.get("lstm_fpr", [])
        lstm_tpr = metrics.get("lstm_tpr", [])
        if lstm_fpr and lstm_tpr:
            lstm_auroc = metrics.get("lstm_auroc", 0)
            lstm_gini_val_l = metrics.get("lstm_gini", 0)
            fig.add_trace(go.Scatter(
                x=lstm_fpr, y=lstm_tpr,
                mode="lines",
                name=f"LSTM (AUC = {lstm_auroc:.3f}, Gini = {lstm_gini_val_l:.3f})",
                    line=dict(color="#d62728", width=2.5),
                ))
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"XGBoost CV (AUC = {xgb_auroc_val:.3f})",
            line=dict(color="#1f77b4", width=2.5),
        ))
        if lr_fpr and lr_tpr:
            lr_auroc_val = metrics.get("cv_lr_auroc", 0)
            fig.add_trace(go.Scatter(
                x=lr_fpr, y=lr_tpr,
                mode="lines",
                name=f"Logistic Reg CV (AUC = {lr_auroc_val:.3f})",
                line=dict(color="#2ca02c", width=2),
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random Guess (AUC = 0.500)",
            line=dict(color="gray", dash="dash", width=1),
        ))
        fig.add_shape(
            type="rect", x0=0, x1=0.2, y0=0.8, y1=1.0,
            line=dict(color="green", dash="dot"),
            fillcolor="rgba(0,255,0,0.05)",
        )
        fig.add_annotation(
            x=0.1, y=0.9, text="Clinical<br>Target Zone",
            showarrow=False, font=dict(size=10, color="green"),
        )
        fig.update_layout(
            xaxis_title="False Positive Rate (false alarms)",
            yaxis_title="True Positive Rate (sepsis detected)",
            height=500,
            legend=dict(x=0.4, y=0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Precision-recall curve data not stored in CV-only pipeline yet
    pr_prec = metrics.get("pr_precision", [])
    pr_rec = metrics.get("pr_recall", [])
    if pr_prec and pr_rec:
        st.markdown("### Precision-Recall Curve")
        fig_pr = go.Figure()
        pr_auc_val = metrics.get("cv_xgb_pr_auc", 0)
        fig_pr.add_trace(go.Scatter(
            x=pr_rec, y=pr_prec,
            mode="lines",
            name=f"XGBoost (PR-AUC = {pr_auc_val:.3f})",
            line=dict(color="#1f77b4", width=2.5),
        ))
        fig_pr.add_trace(go.Scatter(
            x=[0, 1], y=[0.073, 0.073],
            mode="lines",
            name="Random Baseline (~7.3% prevalence)",
            line=dict(color="gray", dash="dash", width=1),
        ))
        fig_pr.update_layout(
            xaxis_title="Recall (sensitivity — sepsis cases caught)",
            yaxis_title="Precision (% of alerts that are real)",
            height=500,
            legend=dict(x=0.4, y=0.95),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    explain(
        clinical=(
            "The ROC curve shows the tradeoff between <b>catching sepsis cases</b> (y-axis) and "
            "<b>false alarms</b> (x-axis). The higher and more to the left, the better. "
            "The green box is where a clinically useful model would live."
        ),
        technical=(
            "ROC curve from the last CV fold. Both XGBoost and LR curves shown. "
            "The closer the curve hugs the top-left corner, the better the discrimination."
        ),
        audience=audience,
    )

    # Patient-level metrics (the ones clinicians care about)
    p_sens = metrics.get("patient_sensitivity")
    p_spec = metrics.get("patient_specificity")
    p_prec = metrics.get("patient_precision")
    if p_sens is not None:
        st.markdown("### Patient-Level Metrics (What Clinicians Care About)")
        st.caption("Did the model alert on this patient at any point? Yes/No — averaged across CV folds")
        c1, c2, c3 = st.columns(3)
        c1.metric("Patient Sensitivity", f"{p_sens:.1%}", help="% of sepsis patients the model flagged at least once")
        c2.metric("Patient Specificity", f"{p_spec:.1%}", help="% of non-sepsis patients correctly left alone")
        c3.metric("Patient Precision", f"{p_prec:.1%}", help="Of patients flagged, % that actually had sepsis")

        explain(
            clinical=(
                f"At the patient level: the model catches <b>{p_sens:.0%} of sepsis patients</b> "
                f"but also false-alarms on <b>{1-p_spec:.0%} of non-sepsis patients</b>. "
                f"Of everyone flagged, <b>{p_prec:.0%}</b> actually had sepsis."
            ),
            technical=(
                f"Patient-level: sens={p_sens:.3f}, spec={p_spec:.3f}, prec={p_prec:.3f}. "
                f"Computed by taking max(prob) per patient across all their hours and comparing to threshold. "
                f"Averaged across CV folds. These differ significantly from the hour-level metrics above."
            ),
            audience=audience,
        )

    # Confusion matrix (real, computed from actual predictions)
    st.markdown("### What Happens in Practice (Patient-Level Confusion Matrix)")

    cm = metrics.get("confusion_matrix")
    if cm:
        tp = cm["tp"]
        fn = cm["fn"]
        fp = cm["fp"]
        tn = cm["tn"]
        n_total = cm["total_patients"]
        n_actual = cm["actual_sepsis"]
        n_no_sepsis = cm["actual_no_sepsis"]

        fig = go.Figure(data=go.Heatmap(
            z=[[tp, fn], [fp, tn]],
            x=["Predicted Sepsis", "Predicted No Sepsis"],
            y=["Actually Had Sepsis", "Actually No Sepsis"],
            text=[
                [f"Caught: {tp:,}", f"Missed: {fn:,}"],
                [f"False Alarm: {fp:,}", f"Correct: {tn:,}"],
            ],
            texttemplate="%{text}",
            colorscale=[[0, "#fee"], [0.5, "#fca"], [1, "#c33"]],
            showscale=False,
            hoverinfo="text",
        ))
        fig.update_layout(
            title=f"Patient-Level Outcomes — CV ({n_total:,} patients)",
            height=350,
            xaxis_title="Model Prediction",
            yaxis_title="Reality",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        explain(
            clinical=(
                f"Out of <b>{n_actual:,} actual sepsis patients</b>:<br>"
                f"- <b>{tp:,}</b> were correctly flagged (caught)<br>"
                f"- <b>{fn:,}</b> were missed<br><br>"
                f"Out of <b>{n_no_sepsis:,} non-sepsis patients</b>:<br>"
                f"- <b>{fp:,}</b> received a false alarm<br>"
                f"- <b>{tn:,}</b> were correctly left alone<br><br>"
                f"The false alarm count ({fp:,}) is still high relative to true catches ({tp:,}) — "
                f"this is inherent to the 7% prevalence. Feature selection and threshold tuning can improve this."
            ),
            technical=(
                f"TP={tp:,}, FN={fn:,}, FP={fp:,}, TN={tn:,}. "
                f"PPV={ppv:.3f}, NPV={npv:.3f}. "
                f"Derived from CV sensitivity ({cm['tp']/n_actual:.3f}) and specificity ({cm['tn']/n_no_sepsis:.3f}) "
                f"applied to full population (40,111 patients). "
                f"Low PPV is expected at 7% prevalence — even good classifiers produce many FP."
            ),
            audience=audience,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Patient Deep Dive
# ══════════════════════════════════════════════════════════════════════════════

def page_patient_deep_dive():
    st.markdown("## Patient Deep Dive")

    df = get_raw_data()
    if df is None:
        no_data_warning()
        return

    explain(
        clinical=(
            "Select a patient below to see their full ICU timeline — vital signs, lab values, "
            "and when sepsis occurred. The <b>red shading</b> marks the period when the sepsis "
            "label is active (starting 6 hours before clinical diagnosis)."
        ),
        technical=(
            "Raw (pre-imputation) data plotted. Lab values shown as dots at actual draw times. "
            "SepsisLabel shading starts at t_sepsis - 6h per the CinC 2019 labeling scheme."
        ),
        audience=audience,
    )

    # Patient selection
    sepsis_ids = sorted(
        df[df[LABEL_COL] == 1]["patient_id"].unique().tolist()
    )
    non_sepsis_ids = sorted(
        df[df[LABEL_COL] == 0].groupby("patient_id").filter(
            lambda x: x[LABEL_COL].max() == 0
        )["patient_id"].unique().tolist()
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        show_sepsis = st.radio("Show:", ["Sepsis Patients", "Non-Sepsis Patients"])

    patient_list = sepsis_ids if show_sepsis == "Sepsis Patients" else non_sepsis_ids[:500]
    selected = st.selectbox(
        f"Select Patient ({len(patient_list):,} available)",
        patient_list,
    )

    if selected:
        patient_df = df[df["patient_id"] == selected].sort_values(TIME_COL)
        stay_hrs = patient_df[TIME_COL].max()
        has_sepsis = patient_df[LABEL_COL].max() == 1
        onset_hr = patient_df[patient_df[LABEL_COL] == 1][TIME_COL].min() if has_sepsis else None

        # Info bar
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patient", selected)
        c2.metric("Stay Length", f"{stay_hrs:.0f} hrs")
        c3.metric("Sepsis", "Yes" if has_sepsis else "No")
        c4.metric("Onset Hour", f"{onset_hr:.0f}" if onset_hr else "N/A")

        # Vital signs
        st.markdown("### Vital Signs Over Time")
        available_vitals = [v for v in VITAL_COLS if v in patient_df.columns and patient_df[v].notna().any()]

        fig = make_subplots(
            rows=len(available_vitals), cols=1,
            shared_xaxes=True,
            subplot_titles=available_vitals,
            vertical_spacing=0.03,
        )

        for i, vital in enumerate(available_vitals, 1):
            fig.add_trace(
                go.Scatter(
                    x=patient_df[TIME_COL], y=patient_df[vital],
                    mode="lines+markers", name=vital,
                    marker=dict(size=3), line=dict(width=1.5),
                    showlegend=False,
                ),
                row=i, col=1,
            )

            # Add sepsis shading
            if has_sepsis:
                sepsis_hours = patient_df[patient_df[LABEL_COL] == 1][TIME_COL]
                fig.add_vrect(
                    x0=sepsis_hours.min() - 0.5,
                    x1=sepsis_hours.max() + 0.5,
                    fillcolor="red", opacity=0.08,
                    layer="below", line_width=0,
                    row=i, col=1,
                )

        fig.update_layout(
            height=250 * len(available_vitals),
            title_text="Vital Signs Timeline (red = sepsis window)",
        )
        fig.update_xaxes(title_text="ICU Hours", row=len(available_vitals), col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Key labs
        st.markdown("### Key Lab Values (measured timepoints only)")
        key_labs = ["Lactate", "WBC", "Creatinine", "Platelets", "Bilirubin_total"]
        available_labs = [l for l in key_labs if l in patient_df.columns and patient_df[l].notna().any()]

        if available_labs:
            fig_labs = make_subplots(
                rows=len(available_labs), cols=1,
                shared_xaxes=True,
                subplot_titles=available_labs,
                vertical_spacing=0.05,
            )
            for i, lab in enumerate(available_labs, 1):
                measured = patient_df[patient_df[lab].notna()]
                fig_labs.add_trace(
                    go.Scatter(
                        x=measured[TIME_COL], y=measured[lab],
                        mode="markers+lines", name=lab,
                        marker=dict(size=8), line=dict(width=1, dash="dot"),
                        showlegend=False,
                    ),
                    row=i, col=1,
                )
                if has_sepsis:
                    sepsis_hours = patient_df[patient_df[LABEL_COL] == 1][TIME_COL]
                    fig_labs.add_vrect(
                        x0=sepsis_hours.min() - 0.5,
                        x1=sepsis_hours.max() + 0.5,
                        fillcolor="red", opacity=0.08,
                        layer="below", line_width=0,
                        row=i, col=1,
                    )

            fig_labs.update_layout(height=220 * len(available_labs))
            fig_labs.update_xaxes(title_text="ICU Hours", row=len(available_labs), col=1)
            st.plotly_chart(fig_labs, use_container_width=True)
        else:
            st.info("No key lab values measured for this patient.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: What Drives Predictions
# ══════════════════════════════════════════════════════════════════════════════

def page_feature_importance():
    st.markdown("## What Drives Predictions")

    metrics = get_model_metrics()
    if metrics is None:
        no_data_warning()
        return

    explain(
        clinical=(
            "Which measurements matter most to the model? This shows the top factors the model "
            "uses when deciding if a patient is at risk. Features that appear at the top have "
            "the biggest influence on the prediction."
        ),
        technical=(
            "XGBoost gain-based feature importance from the best CV model. "
            "Note: feature importance reflects what the model learned, not necessarily "
            "what's clinically causal. Site-specific features (Unit2, ICULOS) dominating "
            "is a red flag for overfitting."
        ),
        audience=audience,
    )

    feat_imp = metrics.get("feature_importance", {})
    if not feat_imp:
        st.warning("No feature importance data available.")
        return

    # Categorize features
    def categorize(name):
        if any(v in name for v in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]):
            return "Vital Sign"
        if "measured" in name or "hours_since" in name:
            return "Testing Pattern"
        if "roll" in name or "delta" in name:
            return "Trend / Variability"
        if name in ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]:
            return "Demographics / Time"
        return "Lab Value"

    imp_df = (
        pd.DataFrame({"Feature": list(feat_imp.keys()), "Importance": list(feat_imp.values())})
        .sort_values("Importance", ascending=False)
        .head(25)
    )
    imp_df["Category"] = imp_df["Feature"].apply(categorize)

    color_map = {
        "Vital Sign": "#1f77b4",
        "Lab Value": "#2ca02c",
        "Trend / Variability": "#ff7f0e",
        "Testing Pattern": "#9467bd",
        "Demographics / Time": "#d62728",
    }

    fig = px.bar(
        imp_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        color="Category",
        orientation="h",
        title="Top 25 Most Important Features",
        color_discrete_map=color_map,
    )
    fig.update_layout(height=700, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    explain(
        clinical=(
            "<b>Key findings:</b><br>"
            "- <b>Lactate trends</b> (variability, time since measurement, rolling min) are among the top predictors — "
            "this aligns with clinical knowledge that rising lactate is an early sepsis marker<br>"
            "- <b>Temperature max</b> over 6 hours is important — fever is a classic sepsis sign<br>"
            "- <b>Creatinine trends</b> (rolling max) appear — kidney function decline can indicate organ dysfunction<br>"
            "- <b>Testing frequency</b> (hours since FiO2/BUN measurement) is predictive — sicker patients get tested more<br><br>"
            "<b>Concern:</b> Hospital unit type (Unit2) and time in ICU (ICULOS) are top features. "
            "This means the model may be learning <i>which hospital unit you're in</i> rather than "
            "<i>your actual clinical trajectory</i>. This is a fixable problem."
        ),
        technical=(
            "Unit2 and ICULOS dominating feature importance confirms site-specific overfitting. "
            "These are confounders, not causal features. Remediation: (1) drop Unit1/Unit2 from features, "
            "(2) train on both hospitals with CV, (3) add time-aware features that normalize ICULOS. "
            "The clinical signal IS present (Lactate_roll_std, Temp_roll_max, Creatinine_roll_max) "
            "but is being drowned out by the site-specific features."
        ),
        audience=audience,
    )

    # ── Combined Feature Ranking ─────────────────────────────────────
    feature_ranking = metrics.get("feature_ranking")
    if feature_ranking:
        st.markdown("### Feature Ranking — Three Methods Compared")
        rank_df = pd.DataFrame(feature_ranking)
        display_cols = ["feature", "iv", "gain_pct", "mean_abs_shap", "average_rank"]
        available_cols = [c for c in display_cols if c in rank_df.columns]
        if available_cols:
            st.dataframe(rank_df[available_cols].head(20), use_container_width=True, hide_index=True)

    # ── SHAP Plots ───────────────────────────────────────────────────
    # Prefer top-50 plots if available, fall back to original
    shap_summary_path = DATA_PROCESSED / "feature_analysis" / "shap_top50_beeswarm.png"
    if not shap_summary_path.exists():
        shap_summary_path = DATA_PROCESSED / "shap_summary.png"
    shap_bar_path = DATA_PROCESSED / "feature_analysis" / "shap_top50_bar.png"
    if not shap_bar_path.exists():
        shap_bar_path = DATA_PROCESSED / "shap_bar.png"

    if shap_summary_path.exists():
        st.markdown("### SHAP Summary Plot (Beeswarm)")
        st.image(str(shap_summary_path), use_container_width=True)
        explain(
            clinical=(
                "Each dot is one hourly observation. The horizontal position shows how much that "
                "feature pushed the prediction toward sepsis (right) or away from it (left). "
                "The color shows whether the feature value was high (red) or low (blue). "
                "Features at the top have the most influence overall."
            ),
            technical=(
                "SHAP TreeExplainer values on 10K sampled observations from the XGBoost model. "
                "Each dot = one sample's SHAP value for that feature. "
                "Color = feature value (red=high, blue=low). "
                "X-axis = SHAP value (impact on model output in log-odds space)."
            ),
            audience=audience,
        )

    if shap_bar_path.exists():
        st.markdown("### SHAP Feature Importance (Mean |SHAP|)")
        st.image(str(shap_bar_path), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: Limitations & Next Steps
# ══════════════════════════════════════════════════════════════════════════════

def page_limitations():
    st.markdown("## Limitations & Next Steps")

    st.markdown("### Current Limitations")

    explain(
        clinical=(
            "<b>What this model CANNOT do yet:</b><br>"
            "1. It cannot reliably predict sepsis across different hospitals (it learned one hospital's patterns too well)<br>"
            "2. It misses 44% of true sepsis cases<br>"
            "3. It raises too many false alarms (37% of healthy patients get flagged)<br>"
            "4. It doesn't account for medications, fluids, or antibiotics already being given<br>"
            "5. It's missing key biomarkers: procalcitonin, CRP, urine output, albumin<br>"
            "6. It only works on ICU patients, not general ward or emergency department"
        ),
        technical=(
            "<b>Technical issues identified:</b><br>"
            "1. <b>Overfitting to Hospital A</b> — 0.993 CV AUROC vs 0.624 validation AUROC<br>"
            "2. <b>Site-specific confounders</b> — Unit2, ICULOS dominate feature importance<br>"
            "3. <b>Poor probability calibration</b> — optimal threshold at 0.018 (should be ~0.5)<br>"
            "4. <b>Logistic regression outperforms XGBoost</b> — tree model memorized noise<br>"
            "5. <b>7 missing clinical variables</b> — no urine output, procalcitonin, CRP, INR, vasopressors, vent status, albumin<br>"
            "6. <b>No feature scaling</b> — logistic regression convergence warning"
        ),
        audience=audience,
    )

    st.markdown("### Improvement Roadmap")

    st.markdown("#### Quick Wins (same data, better model)")
    improvements = pd.DataFrame([
        {"Fix": "Train on both hospitals with stratified CV", "Expected Impact": "High", "Effort": "Low",
         "Why": "Eliminates cross-hospital overfitting"},
        {"Fix": "Drop Unit1/Unit2 from features", "Expected Impact": "Medium", "Effort": "Trivial",
         "Why": "Removes site-specific confounders"},
        {"Fix": "Add feature scaling (StandardScaler)", "Expected Impact": "Medium", "Effort": "Low",
         "Why": "Fixes logistic regression convergence"},
        {"Fix": "Reduce XGBoost max_depth to 3-4", "Expected Impact": "Medium", "Effort": "Low",
         "Why": "Reduces overfitting to noise"},
        {"Fix": "Feature selection (drop low-importance features)", "Expected Impact": "Medium", "Effort": "Low",
         "Why": "Removes noise features"},
    ])
    st.dataframe(improvements, use_container_width=True, hide_index=True)

    st.markdown("#### Medium-Term (better data)")
    st.markdown("""
    - **MIMIC-IV integration** — adds urine output, vasopressors, ventilator status, albumin, INR
    - **Multi-center training** — train on 3+ hospitals for robust generalization
    - **Deep learning** — LSTM/GRU models that understand time-series sequences natively
    """)

    st.markdown("#### End Goal (real patient data)")
    st.markdown("""
    - **Institutional data** — train on your hospital's actual EHR data
    - **Prospective validation** — run alongside clinicians in shadow mode
    - **Clinical integration** — alerts in the EHR with explainable risk factors
    """)

    st.markdown("### The Key Takeaway")
    st.success(
        "**The proof of concept demonstrates that sepsis prediction from hourly vitals and labs is feasible.** "
        "Clinically relevant features (lactate, temperature, creatinine, testing frequency) emerge as "
        "important predictors. The model needs architectural improvements and better data to reach "
        "clinical-grade accuracy, but the foundation is solid."
    )


# ── Router ────────────────────────────────────────────────────────────────────

PAGE_DISPATCH = {
    "Executive Summary": page_executive_summary,
    "The Data": page_the_data,
    "How the Model Works": page_how_model_works,
    "Model Performance": page_model_performance,
    "Patient Deep Dive": page_patient_deep_dive,
    "What Drives Predictions": page_feature_importance,
    "Limitations & Next Steps": page_limitations,
}

PAGE_DISPATCH[page]()
