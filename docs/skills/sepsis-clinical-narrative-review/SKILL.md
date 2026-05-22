---
name: sepsis-clinical-narrative-review
description: Use when updating reports, decks, answers to clinicians, or claims about sepsis prediction, early warning, symptoms, severity, leakage, or model validity in the SepsisDataModel project.
---

# Sepsis Clinical Narrative Review

Use conservative language.

Allowed:

- "early-risk PoC"
- "patient-level CV"
- "trajectory-oriented features"
- "requires richer EHR validation"

Avoid:

- "severity independent"
- "clinically validated"
- "deployment ready"
- stale AUROC values
- symptom features framed as genuine lead-time predictors

Required limitation:

PhysioNet lacks full severity adjustment variables such as APACHE/SOFA, diagnoses, comorbidities, medications, ventilation, vasopressors, and urine output.

Recommended claim:

> The cleaned model demonstrates feasibility of hourly ICU risk stratification, but clinical validity requires prospective testing on richer EHR data with severity adjustment.
