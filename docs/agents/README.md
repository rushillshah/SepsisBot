# Project Agent Roster

Use these as role prompts/checklists for future Codex sessions. They are project-local specs, not installed global agents.

## When To Use Which Agent

| Agent | Use When |
|---|---|
| [CV Validation Agent](cv-validation-agent.md) | Changing splits, labels, thresholds, metrics, or training logic |
| [Feature Selection Agent](feature-selection-agent.md) | Adding/dropping features, changing IV/correlation/SHAP logic |
| [Clinical Claim Review Agent](clinical-claim-review-agent.md) | Updating deck/report wording or answering clinician criticism |
| [Artifact Audit Agent](artifact-audit-agent.md) | Before presenting, committing, or trusting generated artifacts |
| [Deck Narrative Agent](deck-narrative-agent.md) | Translating results into slides |
| [Data Leakage Agent](data-leakage-agent.md) | Any time performance jumps, top features look suspicious, or labels change |

## Standing Rule

Every time the project discovers a new failure mode, update the relevant agent file and `docs/skills/`.
