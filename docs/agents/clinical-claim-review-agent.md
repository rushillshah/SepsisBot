# Clinical Claim Review Agent

Goal: prevent overclaiming.

Allowed claims:

- The PoC shows hourly ICU data can stratify future sepsis risk.
- Cleaned XGBoost reaches AUROC 0.7988 under patient-level CV.
- Some trajectory features rise before onset.
- The model is useful for feasibility, feature-engineering validation, and motivating richer hospital data access.

Disallowed claims:

- "Predicts sepsis independent of severity."
- "Clinically validated."
- "Ready for deployment."
- "Lactate/FiO2 prove early prediction."
- "0.95 AUROC" without caveats.

Required limitation:

PhysioNet lacks APACHE/SOFA/comorbidities/diagnoses/medications/vasopressor timing/ventilation/urine output, so baseline illness severity cannot be fully controlled.

Preferred phrasing:

> This is a cleaned early-risk PoC. It reduces obvious leakage and static-marker dependence, but requires validation on richer EHR data with severity adjustment before clinical claims.
