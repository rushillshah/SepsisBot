# AGENTS.md — Sepsis Early Warning System

## On Session Start

Read these first:

1. `docs/sessions/` — development history and known pitfalls.
2. `docs/model_report.md` — current model documentation.
3. `docs/agents/README.md` — project-local specialist agents.
4. `docs/skills/README.md` — reusable project workflows.

## Current Status

Latest clean run: **2026-05-23**

- Model: XGBoost primary, Logistic Regression baseline.
- Evaluation: **3-fold patient-level stratified CV**.
- Feature policy: leading-indicator filter plus fold-local collinearity pruning.
- Collinearity rule: drop one feature from any pair with `|Pearson r| >= 0.8`, keeping the higher-IV feature.
- Final saved feature count: **119**.
- Max retained correlation: **0.773**.
- XGBoost CV AUROC: **0.7988**.
- XGBoost CV PR-AUC: **0.0656**.
- LR CV AUROC: **0.7480**.
- LR CV PR-AUC: **0.0438**.

At threshold `0.30`:

| Sensitivity | Specificity | Precision | Flagged |
|---:|---:|---:|---:|
| 91.4% | 52.5% | 10.3% | 19,704 |

## Critical Guardrails

- Do **not** claim severity-independent prediction. PhysioNet does not include APACHE/SOFA/comorbidity/medication severity adjustment.
- Do **not** use old plots/artifacts showing `ICULOS`, `FiO2`, lab-testing frequency, or 170/248 features as current.
- Do **not** headline old AUROC values (`0.955`, `0.854`, `0.812`) without explaining they came from earlier less-clean model states.
- Use the current clean headline: **XGBoost AUROC 0.7988 after leading-indicator filtering and collinearity pruning**.
- Treat the model as a **PoC early-risk model**, not deployable clinical validation.

## Current Feature Policy

Excluded from the current clinical model:

- `ICULOS`
- `Unit1`, `Unit2`, `HospAdmTime`
- raw `FiO2`
- static lab absolute-level encodings where configured as cohort markers
- measurement frequency / missingness features
- highly correlated lower-IV features at `|r| >= 0.8`

The suspicious early-warning-score trio was resolved:

- Kept: `early_warning_score_avg_6h`
- Dropped: `early_warning_score_max_6h`
- Dropped: `early_warning_score`

Current top features include:

1. `early_warning_score_avg_6h`
2. `Temp_max_6h`
3. `Resp_avg_6h`
4. `pH_drift_from_normal_6h`
5. `MAP_min_6h`
6. `HR_max_6h`

## Important Artifacts

- `data/processed/model_metrics.json`
- `data/processed/models/feature_names.pkl`
- `data/processed/feature_analysis/final_feature_list.csv`
- `data/processed/feature_analysis/collinearity_pruning_audit.csv`
- `data/processed/feature_analysis/retained_feature_correlation_audit.json`
- `data/processed/feature_analysis/shap_ranking.csv`
- `data/processed/feature_analysis/leadup/iv_by_lead_time.csv`

Note: SHAP PNG generation is currently blocked by a local Python `pyexpat/libexpat` issue. `shap_ranking.csv` is generated through XGBoost native contribution values.

## Running

```bash
source .venv/bin/activate
python run_pipeline.py
pytest tests/ -v
streamlit run app.py
```

Pipeline caches raw/imputed parquet files. A full clean training run takes roughly 15-45 minutes depending on local machine state.

## Specialist Agents

Before major work, pick the relevant project-local agent from `docs/agents/README.md`:

- CV Validation Agent
- Feature Selection Agent
- Clinical Claim Review Agent
- Artifact Audit Agent
- Deck / Narrative Agent
- Data Leakage Agent
- Sepsis PPT Generation Skill

When a new failure mode appears, add it to the relevant agent and skill docs so future sessions inherit the lesson.
