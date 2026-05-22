# MCP / Tooling Roadmap

These are not implemented yet. They are candidates for a project MCP server or plugin later.

## `sepsis.audit_artifacts`

Inputs:

- processed data directory

Outputs:

- feature count consistency
- banned feature check
- stale artifact timestamps
- max retained correlation
- top-feature drift since previous run

## `sepsis.validate_cv`

Inputs:

- CV predictions
- patient IDs
- labels

Outputs:

- patient overlap check
- fold summary
- patient-level threshold table
- warning if metrics were computed from training predictions

## `sepsis.feature_selection_report`

Inputs:

- `final_feature_list.csv`
- `collinearity_pruning_audit.csv`
- IV/gain/SHAP rankings

Outputs:

- dropped feature families
- retained correlated-pair exceptions
- top-feature clinical risk flags

## `sepsis.deck_claim_linter`

Inputs:

- Markdown/PPT text export

Outputs:

- unsupported claims
- stale metric references
- missing limitations
- clinician-risk wording suggestions
