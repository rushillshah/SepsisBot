"""Sepsis Early Warning — Proof of Concept Dashboard.

Streamlined dashboard for the sepsis prediction model.
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
    DATA_PROCESSED,
    LABEL_COL,
    LAB_COLS,
    TIME_COL,
    VITAL_COLS,
)

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sepsis Early Warning System — PoC",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-explanation {
        background: #1a2332;
        border-radius: 8px;
        padding: 14px;
        margin: 8px 0;
        border: 1px solid #2a4a6b;
        color: #e0e0e0;
        font-size: 0.9rem;
    }
    .formula-box {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        border: 1px solid #444;
        font-family: monospace;
        text-align: center;
        font-size: 1.1rem;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_parquet(name: str) -> pd.DataFrame | None:
    path = DATA_PROCESSED / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data
def load_json(name: str) -> dict | None:
    path = DATA_PROCESSED / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_woe_data() -> dict | None:
    path = DATA_PROCESSED / "feature_analysis" / "woe_buckets.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def no_data_warning():
    st.error("No processed data found. Run `python run_pipeline.py` first.")


# ── Navigation ───────────────────────────────────────────────────────────────

PAGES = ["Overview", "Performance", "Feature Analysis", "Patient Explorer"]

st.sidebar.markdown("### Sepsis Early Warning PoC")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Overview
# ══════════════════════════════════════════════════════════════════════════════

def page_overview():
    st.markdown("# Sepsis Early Warning System")
    st.caption("Proof of Concept — PhysioNet CinC 2019 Open Data (40,111 ICU patients)")
    st.markdown("---")

    metrics = load_json("model_metrics")
    df = load_parquet("raw_data")

    if metrics is None:
        no_data_warning()
        return

    # ── Overall Score ────────────────────────────────────────────────
    auroc = metrics.get("cv_xgb_auroc", 0)
    auroc_std = metrics.get("cv_xgb_auroc_std", 0)
    gini = 2 * auroc - 1

    st.markdown("### Model Score")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        grade = "Excellent" if auroc >= 0.90 else "Good" if auroc >= 0.80 else "Fair" if auroc >= 0.70 else "Poor"
        color = "#2ca02c" if auroc >= 0.90 else "#1f77b4" if auroc >= 0.80 else "#ff7f0e" if auroc >= 0.70 else "#d62728"
        st.metric("AUROC", f"{auroc:.3f} ± {auroc_std:.3f}",
                  help="Area Under the ROC Curve. Measures how well the model ranks sepsis patients above non-sepsis. 1.0 = perfect, 0.5 = coin flip. Above 0.80 is considered good for clinical models.")
        st.markdown(f'<div class="metric-explanation">'
                    f'<b>Rating: <span style="color:{color}">{grade}</span></b><br>'
                    f'AUROC measures how well the model separates sepsis from non-sepsis patients. '
                    f'1.0 = perfect separation, 0.5 = random coin flip.</div>',
                    unsafe_allow_html=True)
    with c2:
        st.metric("Gini Coefficient", f"{gini:.3f}",
                  help="Gini = 2 x AUROC - 1. Ranges from 0 (no discrimination) to 1 (perfect). Common in credit/risk modeling as an alternative way to express AUROC.")
        st.markdown(f'<div class="metric-explanation">'
                    f'Another way to express discriminative power.<br>'
                    f'<b>Gini = 2 × AUROC − 1</b></div>',
                    unsafe_allow_html=True)
    with c3:
        pr_auc = metrics.get("cv_xgb_pr_auc", 0)
        st.metric("PR-AUC", f"{pr_auc:.3f}" if pr_auc else "N/A",
                  help="Precision-Recall AUC. Unlike AUROC, PR-AUC is sensitive to class imbalance. With only 7.3% sepsis prevalence, this is a stricter measure of how well the model identifies the rare positive class.")
        st.markdown(f'<div class="metric-explanation">'
                    f'Like AUROC but focused on the rare sepsis class. '
                    f'More informative when prevalence is low (7.3%).</div>',
                    unsafe_allow_html=True)

    st.markdown(
        '<div class="formula-box">'
        'AUROC = P(model scores a sepsis patient higher than a non-sepsis patient)'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Key Metrics with Explanations ────────────────────────────────
    st.markdown("### What This Means in Practice")

    threshold = metrics.get("default_threshold", 0.30)
    p_sens = metrics.get("patient_sensitivity", 0)
    p_spec = metrics.get("patient_specificity", 0)
    p_prec = metrics.get("patient_precision", 0)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Sensitivity", f"{p_sens:.0%}",
                  help="Also called Recall or True Positive Rate. Of all patients who actually developed sepsis, what percentage did the model flag? A sensitivity of 90% means 10% of sepsis cases were missed.")
        st.markdown(f'<div class="metric-explanation">'
                    f'Of patients who <b>actually had sepsis</b>, how many did the model catch?<br><br>'
                    f'Higher = fewer missed cases.</div>',
                    unsafe_allow_html=True)
    with c2:
        st.metric("Specificity", f"{p_spec:.0%}",
                  help="Also called True Negative Rate. Of all patients who did NOT develop sepsis, what percentage were correctly left alone? Low specificity means too many false alarms, causing alert fatigue.")
        st.markdown(f'<div class="metric-explanation">'
                    f'Of patients who <b>did NOT have sepsis</b>, how many were correctly left alone?<br><br>'
                    f'Higher = fewer false alarms.</div>',
                    unsafe_allow_html=True)
    with c3:
        st.metric("Precision", f"{p_prec:.0%}",
                  help="Also called Positive Predictive Value (PPV). When the model raises an alert, how often is the patient actually septic? Low precision means clinicians waste time investigating false alarms.")
        st.markdown(f'<div class="metric-explanation">'
                    f'When the model <b>raises an alert</b>, how often is it actually right?<br><br>'
                    f'Higher = more trustworthy alerts.</div>',
                    unsafe_allow_html=True)
    with c4:
        f1 = metrics.get("cv_xgb_f1", 0)
        st.metric("F1 Score", f"{f1:.3f}" if f1 else "N/A",
                  help="Harmonic mean of Precision and Sensitivity. F1 = 2 x (Precision x Sensitivity) / (Precision + Sensitivity). Balances catching sepsis cases vs. not overwhelming clinicians with false alarms.")
        st.markdown(f'<div class="metric-explanation">'
                    f'Balance between sensitivity and precision.<br>'
                    f'<b>F1 = 2 × (Prec × Sens) / (Prec + Sens)</b></div>',
                    unsafe_allow_html=True)

    st.caption(f"All metrics at threshold = {threshold}")

    # ── Data Summary ─────────────────────────────────────────────────
    if df is not None:
        st.markdown("### Data Summary")
        total_patients = df["patient_id"].nunique()
        sepsis_patients = int(df.groupby("patient_id")[LABEL_COL].max().sum())
        total_hours = len(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients", f"{total_patients:,}",
                  help="Total unique ICU patients across both hospitals in the PhysioNet CinC 2019 dataset.")
        c2.metric("Hourly Observations", f"{total_hours:,}",
                   help="Total hourly time-steps across all patients. Each row is one hour of one patient's ICU stay with vitals and labs.")
        c3.metric("Sepsis Cases", f"{sepsis_patients:,}",
                   help="Patients who developed sepsis during their ICU stay (SepsisLabel = 1 at any point). Label is set 6 hours before clinical onset.")
        c4.metric("Sepsis Rate", f"{sepsis_patients/total_patients:.1%}",
                   help="Percentage of patients who developed sepsis. Low prevalence (7.3%) makes this a class-imbalanced problem, which is why we use oversampling and PR-AUC.")

    # ── Clinical Baselines ───────────────────────────────────────────
    st.markdown("### How We Define \"Abnormal\"")
    st.markdown(
        '<div class="metric-explanation">'
        '<b>Age/Gender-Stratified Normal Ranges</b> — Each vital and lab value is compared '
        'against clinically established reference ranges adjusted for the patient\'s age group '
        '(18-40, 40-60, 60-80, 80+) and sex. For example, a temperature of 37.5\u00b0C is flagged '
        'as abnormal in an 80-year-old (normal ceiling: 37.2\u00b0C) but not in a 25-year-old '
        '(ceiling: 37.8\u00b0C). Creatinine of 1.15 mg/dL is abnormal for a young woman '
        '(range: 0.5\u20131.0) but normal for a man (range: 0.7\u20131.2).'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Current Limitations & What Real Hospital Data Would Add"):
        st.markdown("""
**What we have now (PhysioNet open data):**
- Age and Gender only — the sole demographic variables available
- No medication history, no comorbidities, no admitting diagnosis

**What this misses:**
- A patient on **beta-blockers** has a naturally lower HR — their HR of 55 bpm is normal, not bradycardic
- A patient with **chronic kidney disease** has elevated baseline creatinine — a value of 2.0 may be their normal
- A **COPD patient** may have a baseline SpO2 of 88-90% — not the 95%+ we'd expect in a healthy adult
- **Diabetic patients** have different glucose thresholds; **heart failure patients** tolerate different BP ranges

**With real EMR data, the model could use:**
- Admission medication list to adjust vital sign baselines
- Problem list / ICD codes to set condition-specific thresholds
- Pre-ICU outpatient vitals as the patient's true personal baseline
- Vasopressor/ventilator status to contextualize hemodynamic readings

This is a key reason to pursue real hospital data — the model architecture is ready to incorporate these richer features.
""")

    # ── Bottom Line ──────────────────────────────────────────────────
    st.markdown("---")
    st.info(
        "**This PoC proves the concept is viable.** Clinically relevant features "
        "(lactate, temperature, creatinine, testing frequency) emerge as the top predictors. "
        "With real patient data (medications, comorbidities, pre-ICU baselines), "
        "we expect significant improvement in personalized risk assessment."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Performance
# ══════════════════════════════════════════════════════════════════════════════

def page_performance():
    st.markdown("## Model Performance")

    metrics = load_json("model_metrics")
    if metrics is None:
        no_data_warning()
        return

    excl = metrics.get("exclusion_note")
    if excl:
        st.info(
            f"**Exclusions applied:** {excl['early_onset_patients_excluded']} sepsis patients "
            f"with onset ≤ 6h ICU excluded from all metrics (already septic on admission — unpredictable). "
            f"First 6 hours of all other patients are included in training."
        )

    # ── Model Comparison (309 vs Top 100 features) ────────────────
    comparison = metrics.get("model_comparison")
    if comparison:
        st.markdown("### Feature Selection: 309 vs Top 100")
        st.markdown(
            '<div class="metric-explanation">'
            'Comparing the full 309-feature model against a leaner model using only the '
            'top 100 features ranked by Information Value. Fewer features can reduce overfitting '
            'and improve generalization.</div>',
            unsafe_allow_html=True,
        )

        full = comparison["full_309"]
        slim = comparison["top_100"]

        comp_data = {
            "Metric": ["AUROC", "Gini", "PR-AUC", "Overfit Gap", "Patient Sensitivity", "Patient Specificity", "Patient Precision", "Features"],
            f"Full ({full['n_features']} features)": [
                f"{full['xgb_auroc']:.4f}", f"{full['xgb_gini']:.4f}", f"{full['xgb_pr_auc']:.4f}",
                f"{full['overfit_gap']:.4f}", f"{full['patient_sensitivity']:.1%}",
                f"{full['patient_specificity']:.1%}", f"{full['patient_precision']:.1%}",
                str(full['n_features']),
            ],
            f"Top {slim['n_features']} features": [
                f"{slim['xgb_auroc']:.4f}", f"{slim['xgb_gini']:.4f}", f"{slim['xgb_pr_auc']:.4f}",
                f"{slim['overfit_gap']:.4f}", f"{slim['patient_sensitivity']:.1%}",
                f"{slim['patient_specificity']:.1%}", f"{slim['patient_precision']:.1%}",
                str(slim['n_features']),
            ],
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        # Overlay ROC curves if available
        slim_fpr = slim.get("fpr", [])
        slim_tpr = slim.get("tpr", [])
        if slim_fpr and slim_tpr:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=metrics.get("fpr", []), y=metrics.get("tpr", []), mode="lines",
                name=f"309 features (AUROC={full['xgb_auroc']:.3f})",
                line=dict(color="#1f77b4", width=2.5),
            ))
            fig_comp.add_trace(go.Scatter(
                x=slim_fpr, y=slim_tpr, mode="lines",
                name=f"Top {slim['n_features']} (AUROC={slim['xgb_auroc']:.3f})",
                line=dict(color="#ff7f0e", width=2.5),
            ))
            fig_comp.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Random (0.500)", line=dict(color="gray", dash="dash", width=1),
            ))
            fig_comp.update_layout(
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                height=400, legend=dict(x=0.4, y=0.15),
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # ── ROC Curve ────────────────────────────────────────────────────
    fpr = metrics.get("fpr", [])
    tpr = metrics.get("tpr", [])

    if fpr and tpr:
        st.markdown("### ROC Curve")
        fig = go.Figure()

        xgb_auroc = metrics.get("cv_xgb_auroc", 0)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"XGBoost (AUROC = {xgb_auroc:.3f})",
            line=dict(color="#1f77b4", width=2.5),
        ))

        lr_fpr = metrics.get("lr_fpr", [])
        lr_tpr = metrics.get("lr_tpr", [])
        if lr_fpr and lr_tpr:
            lr_auroc = metrics.get("cv_lr_auroc", 0)
            fig.add_trace(go.Scatter(
                x=lr_fpr, y=lr_tpr, mode="lines",
                name=f"Logistic Regression (AUROC = {lr_auroc:.3f})",
                line=dict(color="#2ca02c", width=2),
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Guess (0.500)",
            line=dict(color="gray", dash="dash", width=1),
        ))
        fig.update_layout(
            xaxis_title="False Positive Rate (false alarms)",
            yaxis_title="True Positive Rate (sepsis caught)",
            height=450,
            legend=dict(x=0.4, y=0.15),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="metric-explanation">'
            'The ROC curve shows the tradeoff: catching more sepsis cases (up) vs triggering '
            'more false alarms (right). A perfect model hugs the top-left corner. '
            'The diagonal = random guessing.</div>',
            unsafe_allow_html=True,
        )

    # ── Confusion Matrix ─────────────────────────────────────────────
    cm = metrics.get("confusion_matrix")
    if cm:
        st.markdown("### What Happens in Practice")

        tp, fn, fp, tn = cm["tp"], cm["fn"], cm["fp"], cm["tn"]
        n_actual = cm["actual_sepsis"]
        n_no_sepsis = cm.get("actual_no_sepsis", cm["total_patients"] - n_actual)

        fig = go.Figure(data=go.Heatmap(
            z=[[tp, fn], [fp, tn]],
            x=["Predicted Sepsis", "Predicted No Sepsis"],
            y=["Actually Had Sepsis", "Actually No Sepsis"],
            text=[
                [f"Caught: {tp:,}", f"Missed: {fn:,}"],
                [f"False Alarm: {fp:,}", f"Correct: {tn:,}"],
            ],
            texttemplate="%{text}",
            textfont=dict(size=16, color="white"),
            colorscale=[[0, "#1a3a2a"], [0.5, "#8b4513"], [1, "#b22222"]],
            showscale=False,
        ))
        fig.update_layout(
            title=f"Patient-Level Outcomes ({cm['total_patients']:,} patients)",
            height=320,
            yaxis=dict(autorange="reversed"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="metric-explanation">'
                f'<b>Sepsis patients ({n_actual:,}):</b><br>'
                f'Caught: {tp:,} | Missed: {fn:,}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-explanation">'
                f'<b>Non-sepsis patients ({n_no_sepsis:,}):</b><br>'
                f'Correct: {tn:,} | False alarm: {fp:,}</div>',
                unsafe_allow_html=True,
            )

    # ── Threshold Analysis ───────────────────────────────────────────
    threshold_data = metrics.get("threshold_analysis")
    if threshold_data:
        st.markdown("### Threshold Tuning")
        st.markdown(
            '<div class="metric-explanation">'
            'The <b>threshold</b> controls how strict the model is. '
            'Lower = catches more sepsis but more false alarms. '
            'Higher = fewer false alarms but misses more cases. '
            'Pick the row that matches your clinical tolerance.</div>',
            unsafe_allow_html=True,
        )

        t_df = pd.DataFrame(threshold_data)
        display_cols = ["threshold", "patient_sensitivity", "patient_specificity",
                        "patient_precision", "total_flagged"]
        available = [c for c in display_cols if c in t_df.columns]
        if available:
            fmt_df = t_df[available].copy()
            for col in ["patient_sensitivity", "patient_specificity", "patient_precision"]:
                if col in fmt_df.columns:
                    fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.1%}")
            rename = {
                "threshold": "Threshold",
                "patient_sensitivity": "Sensitivity",
                "patient_specificity": "Specificity",
                "patient_precision": "Precision",
                "total_flagged": "Patients Flagged",
            }
            fmt_df = fmt_df.rename(columns=rename)
            st.dataframe(fmt_df, use_container_width=True, hide_index=True)

    # ── Sustained Alert Analysis ────────────────────────────────────
    consec_data = metrics.get("consecutive_alert_analysis")
    if consec_data:
        st.markdown("### Sustained Alert Mode")
        st.markdown(
            '<div class="metric-explanation">'
            '<b>Problem:</b> The default "any single hour above threshold" rule generates too many '
            'false alarms — one noisy spike in a 5-day stay flags the whole patient.<br><br>'
            '<b>Solution:</b> Require multiple <i>consecutive</i> hours above threshold before alerting. '
            'This filters out transient spikes while catching real deterioration trends.</div>',
            unsafe_allow_html=True,
        )

        ca_df = pd.DataFrame(consec_data)

        # Let user pick min_consecutive to explore
        min_c_options = sorted(ca_df["min_consecutive"].unique())
        selected_min_c = st.select_slider(
            "Minimum consecutive hours required",
            options=min_c_options,
            value=3,
            help="How many hours in a row must the risk score stay above threshold to trigger an alert?",
        )

        filtered = ca_df[ca_df["min_consecutive"] == selected_min_c].copy()
        for col in ["sensitivity", "specificity", "precision"]:
            filtered[col] = filtered[col].map(lambda x: f"{x:.1%}")

        display = filtered[["threshold", "sensitivity", "specificity", "precision", "flagged"]].copy()
        if "median_early_warning_hours" in filtered.columns:
            display["median_early_warning_hours"] = filtered["median_early_warning_hours"].map(
                lambda x: f"{x:.1f}h" if pd.notna(x) and x != "None" else "N/A"
            )
        display = display.rename(columns={
            "threshold": "Threshold",
            "sensitivity": "Sensitivity",
            "specificity": "Specificity",
            "precision": "Precision",
            "flagged": "Patients Flagged",
            "median_early_warning_hours": "Median Early Warning",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

        # Compare best consecutive vs best any-hour at similar sensitivity
        if threshold_data:
            any_hour_30 = next((r for r in threshold_data if abs(r["threshold"] - 0.30) < 0.001), None)
            best_consec = ca_df[(ca_df["min_consecutive"] == selected_min_c)].sort_values("precision", ascending=False)
            # Find row with sensitivity >= 0.75
            viable = best_consec[best_consec["sensitivity"].apply(lambda x: float(x) if isinstance(x, str) else x) >= 0.75]
            if not viable.empty and any_hour_30:
                best = viable.iloc[0]
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f'<div class="metric-explanation">'
                        f'<b>Any-Hour Alert (t=0.30):</b><br>'
                        f'Precision: {any_hour_30["patient_precision"]:.1%} | '
                        f'Flagged: {any_hour_30["total_flagged"]:,}</div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f'<div class="metric-explanation">'
                        f'<b>Sustained Alert (t={best["threshold"]}, {selected_min_c}h):</b><br>'
                        f'Precision: {best["precision"]:.1%} | '
                        f'Flagged: {best["flagged"]:,}</div>',
                        unsafe_allow_html=True,
                    )

    # ── Overfit Check ────────────────────────────────────────────────
    overfit_table = metrics.get("cv_overfit_table")
    if overfit_table:
        with st.expander("Overfit Check (Train vs Validation)"):
            ot_df = pd.DataFrame(overfit_table)
            avg_gap = ot_df["xgb_gap"].mean()
            st.markdown(
                f'<div class="metric-explanation">'
                f'Average train-val AUROC gap: <b>{avg_gap:.3f}</b>. '
                f'{"Good — model generalizes well." if avg_gap < 0.15 else "Large gap — some overfitting remains."}'
                f'<br>A large gap means the model memorized training data instead of learning general patterns.</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(ot_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Feature Analysis
# ══════════════════════════════════════════════════════════════════════════════

def page_feature_analysis():
    st.markdown("## Feature Analysis")
    st.markdown(
        '<div class="metric-explanation">'
        'Which measurements matter most when predicting sepsis? '
        'Features at the top have the biggest influence on the model\'s predictions.</div>',
        unsafe_allow_html=True,
    )

    metrics = load_json("model_metrics")
    if metrics is None:
        no_data_warning()
        return

    # ── Feature Importance Bar Chart ─────────────────────────────────
    feat_imp = metrics.get("feature_importance", {})
    if feat_imp:
        st.markdown("### Top Predictive Features")

        def categorize(name):
            if any(v in name for v in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]):
                return "Vital Sign"
            if "measured" in name or "hours_since" in name:
                return "Testing Pattern"
            if "_6h" in name or "hourly_change" in name:
                return "Trend / Variability"
            if name in ["Age", "Gender"]:
                return "Demographics"
            return "Lab Value"

        imp_df = (
            pd.DataFrame({"Feature": list(feat_imp.keys()), "Importance": list(feat_imp.values())})
            .sort_values("Importance", ascending=False)
            .head(20)
        )
        imp_df["Category"] = imp_df["Feature"].apply(categorize)

        fig = px.bar(
            imp_df.sort_values("Importance"),
            x="Importance", y="Feature", color="Category",
            orientation="h",
            color_discrete_map={
                "Vital Sign": "#1f77b4", "Lab Value": "#2ca02c",
                "Trend / Variability": "#ff7f0e", "Testing Pattern": "#9467bd",
                "Demographics": "#d62728",
            },
        )
        fig.update_layout(
            height=550, yaxis_title="",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Information Value Table ───────────────────────────────────────
    iv_data = metrics.get("iv_top20")
    if iv_data:
        st.markdown("### Information Value (IV)")
        st.markdown(
            '<div class="metric-explanation">'
            '<b>IV measures how well a feature separates sepsis from non-sepsis.</b><br>'
            '< 0.02 = useless | 0.02–0.1 = weak | 0.1–0.3 = medium | 0.3–0.5 = strong | > 0.5 = suspicious (possible leakage)'
            '</div>',
            unsafe_allow_html=True,
        )
        iv_df = pd.DataFrame(iv_data)
        display_cols = [c for c in ["feature", "iv", "iv_strength"] if c in iv_df.columns]
        if display_cols:
            fmt = iv_df[display_cols].copy()
            if "iv" in fmt.columns:
                fmt["iv"] = fmt["iv"].map(lambda x: f"{x:.4f}")
            fmt = fmt.rename(columns={"feature": "Feature", "iv": "IV Score", "iv_strength": "Strength"})
            st.dataframe(fmt, use_container_width=True, hide_index=True)

    # ── WOE Buckets ──────────────────────────────────────────────────
    woe_data = load_woe_data()
    if woe_data:
        st.markdown("### WOE Buckets (Weight of Evidence)")
        st.markdown(
            '<div class="metric-explanation">'
            '<b>WOE shows how each value range of a feature relates to sepsis risk.</b><br>'
            'Positive WOE = higher sepsis risk in that range. '
            'Negative WOE = lower risk. '
            'The further from zero, the stronger the signal.</div>',
            unsafe_allow_html=True,
        )

        feature_list = list(woe_data.keys())
        selected_feature = st.selectbox("Select feature", feature_list)

        if selected_feature and selected_feature in woe_data:
            bucket_df = pd.DataFrame(woe_data[selected_feature])

            # WOE bar chart
            fig = go.Figure()
            colors = ["#2ca02c" if w < 0 else "#d62728" for w in bucket_df["woe"]]
            fig.add_trace(go.Bar(
                x=bucket_df["bucket"],
                y=bucket_df["woe"],
                marker_color=colors,
                text=bucket_df["woe"].map(lambda x: f"{x:.2f}"),
                textposition="outside",
            ))
            fig.update_layout(
                title=f"WOE by Bucket — {selected_feature}",
                xaxis_title="Value Range",
                yaxis_title="Weight of Evidence",
                height=400,
                xaxis_tickangle=-45,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detail table
            detail = bucket_df.copy()
            detail["event_rate"] = detail["event_rate"].map(lambda x: f"{x:.1%}")
            detail = detail.rename(columns={
                "bucket": "Range", "count": "Observations", "events": "Sepsis Cases",
                "event_rate": "Sepsis Rate", "woe": "WOE", "iv_contribution": "IV Contribution",
            })
            st.dataframe(detail, use_container_width=True, hide_index=True)

            st.markdown(
                '<div class="metric-explanation">'
                f'<b>IV Formula:</b> IV = Σ (% of sepsis in bucket − % of non-sepsis in bucket) × WOE<br>'
                f'<b>WOE Formula:</b> WOE = ln(% of sepsis in bucket / % of non-sepsis in bucket)'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── SHAP Plots ───────────────────────────────────────────────────
    shap_path = DATA_PROCESSED / "feature_analysis" / "shap_top50_beeswarm.png"
    if not shap_path.exists():
        shap_path = DATA_PROCESSED / "shap_summary.png"

    if shap_path.exists():
        st.markdown("### SHAP Analysis")
        st.markdown(
            '<div class="metric-explanation">'
            'Each dot = one hourly observation. Position shows how much that feature '
            'pushed the prediction toward sepsis (right) or away (left). '
            'Color = feature value (red = high, blue = low).</div>',
            unsafe_allow_html=True,
        )
        st.image(str(shap_path), use_container_width=True)

    bar_path = DATA_PROCESSED / "feature_analysis" / "shap_top50_bar.png"
    if not bar_path.exists():
        bar_path = DATA_PROCESSED / "shap_bar.png"
    if bar_path.exists():
        st.image(str(bar_path), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Patient Explorer
# ══════════════════════════════════════════════════════════════════════════════

def page_patient_explorer():
    st.markdown("## Patient Explorer")

    df = load_parquet("raw_data")
    if df is None:
        no_data_warning()
        return

    st.markdown(
        '<div class="metric-explanation">'
        'Browse individual patient timelines. Red shading marks the sepsis window '
        '(starting 6 hours before clinical diagnosis).</div>',
        unsafe_allow_html=True,
    )

    # Patient selection
    sepsis_ids = sorted(df[df[LABEL_COL] == 1]["patient_id"].unique().tolist())
    non_sepsis_ids = sorted(
        df[df[LABEL_COL] == 0]
        .groupby("patient_id")
        .filter(lambda x: x[LABEL_COL].max() == 0)["patient_id"]
        .unique().tolist()
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        show_sepsis = st.radio("Show:", ["Sepsis Patients", "Non-Sepsis Patients"])

    patient_list = sepsis_ids if show_sepsis == "Sepsis Patients" else non_sepsis_ids[:500]
    selected = st.selectbox(f"Select Patient ({len(patient_list):,} available)", patient_list)

    if not selected:
        return

    patient_df = df[df["patient_id"] == selected].sort_values(TIME_COL)
    stay_hrs = patient_df[TIME_COL].max()
    has_sepsis = patient_df[LABEL_COL].max() == 1
    onset_hr = patient_df[patient_df[LABEL_COL] == 1][TIME_COL].min() if has_sepsis else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient", selected,
              help="Unique patient identifier from the PhysioNet dataset.")
    c2.metric("Stay Length", f"{stay_hrs:.0f} hrs",
              help="Total hours this patient spent in the ICU. Longer stays may indicate higher acuity.")
    c3.metric("Sepsis", "Yes" if has_sepsis else "No",
              help="Whether this patient developed sepsis during their ICU stay, per Sepsis-3 criteria.")
    c4.metric("Onset Hour", f"{onset_hr:.0f}" if onset_hr else "N/A",
              help="ICU hour when SepsisLabel first becomes 1 (6 hours before clinical sepsis onset). The model aims to flag patients before this point.")

    # Vital signs
    st.markdown("### Vital Signs")
    available_vitals = [v for v in VITAL_COLS if v in patient_df.columns and patient_df[v].notna().any()]

    fig = make_subplots(
        rows=len(available_vitals), cols=1,
        shared_xaxes=True, subplot_titles=available_vitals,
        vertical_spacing=0.03,
    )
    for i, vital in enumerate(available_vitals, 1):
        fig.add_trace(go.Scatter(
            x=patient_df[TIME_COL], y=patient_df[vital],
            mode="lines+markers", name=vital,
            marker=dict(size=3), line=dict(width=1.5),
            showlegend=False,
        ), row=i, col=1)

        if has_sepsis:
            sepsis_hours = patient_df[patient_df[LABEL_COL] == 1][TIME_COL]
            fig.add_vrect(
                x0=sepsis_hours.min() - 0.5, x1=sepsis_hours.max() + 0.5,
                fillcolor="red", opacity=0.08, layer="below", line_width=0,
                row=i, col=1,
            )

    fig.update_layout(
        height=220 * len(available_vitals),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title_text="ICU Hours", row=len(available_vitals), col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Key labs
    st.markdown("### Key Labs")
    key_labs = ["Lactate", "WBC", "Creatinine", "Platelets", "Bilirubin_total"]
    available_labs = [l for l in key_labs if l in patient_df.columns and patient_df[l].notna().any()]

    if available_labs:
        fig_labs = make_subplots(
            rows=len(available_labs), cols=1,
            shared_xaxes=True, subplot_titles=available_labs,
            vertical_spacing=0.05,
        )
        for i, lab in enumerate(available_labs, 1):
            measured = patient_df[patient_df[lab].notna()]
            fig_labs.add_trace(go.Scatter(
                x=measured[TIME_COL], y=measured[lab],
                mode="markers+lines", name=lab,
                marker=dict(size=8), line=dict(width=1, dash="dot"),
                showlegend=False,
            ), row=i, col=1)

            if has_sepsis:
                sepsis_hours = patient_df[patient_df[LABEL_COL] == 1][TIME_COL]
                fig_labs.add_vrect(
                    x0=sepsis_hours.min() - 0.5, x1=sepsis_hours.max() + 0.5,
                    fillcolor="red", opacity=0.08, layer="below", line_width=0,
                    row=i, col=1,
                )

        fig_labs.update_layout(
            height=200 * len(available_labs),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_labs.update_xaxes(title_text="ICU Hours", row=len(available_labs), col=1)
        st.plotly_chart(fig_labs, use_container_width=True)
    else:
        st.info("No key lab values measured for this patient.")


# ── Router ───────────────────────────────────────────────────────────────────

PAGE_DISPATCH = {
    "Overview": page_overview,
    "Performance": page_performance,
    "Feature Analysis": page_feature_analysis,
    "Patient Explorer": page_patient_explorer,
}

PAGE_DISPATCH[page]()
