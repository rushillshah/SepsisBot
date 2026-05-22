"""Configuration constants for the sepsis prediction pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
TRAINING_A = DATA_RAW / "training_setA"
TRAINING_B = DATA_RAW / "training_setB"

# ── Column Groups ──────────────────────────────────────────────────────────────

VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

DEMOGRAPHIC_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

TIME_COL = "ICULOS"
LABEL_COL = "SepsisLabel"
EARLY_LABEL_COL = "early_label"
EARLY_LABEL_EXTRA_HOURS = 6

ALL_FEATURE_COLS = VITAL_COLS + LAB_COLS
ALL_INPUT_COLS = ALL_FEATURE_COLS + DEMOGRAPHIC_COLS + [TIME_COL]

# ── Feature Engineering ────────────────────────────────────────────────────────

ROLLING_WINDOW_HOURS = 6
ROLLING_STATS = ["mean", "min", "max", "std"]
ROLLING_STAT_SUFFIXES = {"mean": "avg_6h", "min": "min_6h", "max": "max_6h", "std": "std_6h"}
CLINICAL_SCORE_COLS = ["inflammation_score", "sepsis_screen_score", "shock_index", "early_warning_score", "lactate_bp_ratio"]

DYNAMIC_BASELINE_FEATURES = VITAL_COLS + LAB_COLS
DYNAMIC_DEVIATION_COLS = [f"{v}_baseline_dev" for v in DYNAMIC_BASELINE_FEATURES]
CUSUM_SLACK = 0.5
CUSUM_THRESHOLD = 4.0

KEY_LAB_COLS = ["Lactate", "WBC", "Creatinine", "Platelets"]
KEY_DEVIATION_COLS = [f"{v}_baseline_dev" for v in VITAL_COLS + KEY_LAB_COLS]
ROLLING_COLS = VITAL_COLS + KEY_LAB_COLS + CLINICAL_SCORE_COLS

# ── Features to Exclude (site-specific confounders) ────────────────────────

EXCLUDED_FEATURES = ["Unit1", "Unit2", "HospAdmTime", "ICULOS"]

# ── Model Hyperparameters ──────────────────────────────────────────────────────

XGBOOST_PARAM_GRID_V2 = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [50, 100, 150],
    "min_child_weight": [5, 10, 20],
    "gamma": [0.1, 0.5, 1.0],
    "reg_alpha": [0.1, 1.0],
    "reg_lambda": [1.0, 5.0],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
}

# ── Age/Gender-Stratified Clinical Normal Ranges ─────────────────────────────
# Reference: Bates' Guide to Physical Examination, Tietz Clinical Guide to
# Laboratory Tests, AHA/ACC guidelines, Mayo Clinic Lab references.

NORMAL_RANGE_COLS = VITAL_COLS + LAB_COLS

AGE_BINS = [0, 40, 60, 80, 200]
AGE_BIN_LABELS = ["18-40", "40-60", "60-80", "80+"]

# {col: {(age_bin_label, gender): (low, high)}}
# Gender: 1 = male, 0 = female (PhysioNet convention)
CLINICAL_NORMAL_RANGES: dict[str, dict[tuple[str, int], tuple[float, float]]] = {
    "HR": {
        ("18-40", 1): (60, 100), ("18-40", 0): (60, 100),
        ("40-60", 1): (60, 100), ("40-60", 0): (60, 100),
        ("60-80", 1): (60, 100), ("60-80", 0): (60, 100),
        ("80+",   1): (60, 100), ("80+",   0): (60, 100),
    },
    "Resp": {
        ("18-40", 1): (12, 20), ("18-40", 0): (12, 20),
        ("40-60", 1): (12, 20), ("40-60", 0): (12, 20),
        ("60-80", 1): (12, 20), ("60-80", 0): (14, 22),
        ("80+",   1): (14, 22), ("80+",   0): (14, 22),
    },
    "Temp": {  # Elderly run cooler (Gomolin et al., JAGS 2005)
        ("18-40", 1): (36.1, 37.8), ("18-40", 0): (36.1, 37.8),
        ("40-60", 1): (36.1, 37.8), ("40-60", 0): (36.1, 37.8),
        ("60-80", 1): (35.8, 37.5), ("60-80", 0): (35.8, 37.5),
        ("80+",   1): (35.6, 37.2), ("80+",   0): (35.6, 37.2),
    },
    "SBP": {  # BP norms increase with age (AHA/ACC)
        ("18-40", 1): (90, 130), ("18-40", 0): (90, 125),
        ("40-60", 1): (90, 140), ("40-60", 0): (90, 135),
        ("60-80", 1): (90, 150), ("60-80", 0): (90, 150),
        ("80+",   1): (100, 160), ("80+", 0): (100, 160),
    },
    "MAP": {
        ("18-40", 1): (70, 100), ("18-40", 0): (70, 100),
        ("40-60", 1): (70, 105), ("40-60", 0): (70, 105),
        ("60-80", 1): (70, 110), ("60-80", 0): (70, 110),
        ("80+",   1): (70, 110), ("80+",   0): (70, 110),
    },
    "O2Sat": {  # SpO2 declines with age (Crapo et al.)
        ("18-40", 1): (95, 100), ("18-40", 0): (95, 100),
        ("40-60", 1): (94, 100), ("40-60", 0): (94, 100),
        ("60-80", 1): (93, 100), ("60-80", 0): (93, 100),
        ("80+",   1): (92, 100), ("80+",   0): (92, 100),
    },
    "DBP": {
        ("18-40", 1): (60, 85), ("18-40", 0): (60, 80),
        ("40-60", 1): (60, 90), ("40-60", 0): (60, 85),
        ("60-80", 1): (60, 90), ("60-80", 0): (60, 90),
        ("80+",   1): (60, 90), ("80+",   0): (60, 90),
    },
    "WBC": {
        ("18-40", 1): (4.5, 11.0), ("18-40", 0): (4.5, 11.0),
        ("40-60", 1): (4.5, 11.0), ("40-60", 0): (4.5, 11.0),
        ("60-80", 1): (4.0, 10.5), ("60-80", 0): (4.0, 10.5),
        ("80+",   1): (3.5, 10.0), ("80+",   0): (3.5, 10.0),
    },
    "Creatinine": {  # Significant gender difference (muscle mass)
        ("18-40", 1): (0.7, 1.2), ("18-40", 0): (0.5, 1.0),
        ("40-60", 1): (0.7, 1.3), ("40-60", 0): (0.6, 1.1),
        ("60-80", 1): (0.8, 1.4), ("60-80", 0): (0.6, 1.2),
        ("80+",   1): (0.8, 1.5), ("80+",   0): (0.7, 1.3),
    },
    "Lactate": {  # Age/gender invariant
        ("18-40", 1): (0.5, 2.0), ("18-40", 0): (0.5, 2.0),
        ("40-60", 1): (0.5, 2.0), ("40-60", 0): (0.5, 2.0),
        ("60-80", 1): (0.5, 2.0), ("60-80", 0): (0.5, 2.0),
        ("80+",   1): (0.5, 2.0), ("80+",   0): (0.5, 2.0),
    },
    "Platelets": {
        ("18-40", 1): (150, 400), ("18-40", 0): (150, 400),
        ("40-60", 1): (150, 400), ("40-60", 0): (150, 400),
        ("60-80", 1): (140, 380), ("60-80", 0): (140, 380),
        ("80+",   1): (130, 350), ("80+",   0): (130, 350),
    },
    "BUN": {
        ("18-40", 1): (7, 20), ("18-40", 0): (7, 18),
        ("40-60", 1): (8, 23), ("40-60", 0): (7, 21),
        ("60-80", 1): (8, 26), ("60-80", 0): (8, 24),
        ("80+",   1): (10, 28), ("80+",   0): (9, 26),
    },
    "Glucose": {  # ICU (non-fasting) range
        ("18-40", 1): (70, 140), ("18-40", 0): (70, 140),
        ("40-60", 1): (70, 140), ("40-60", 0): (70, 140),
        ("60-80", 1): (70, 140), ("60-80", 0): (70, 140),
        ("80+",   1): (70, 140), ("80+",   0): (70, 140),
    },
    "Bilirubin_total": {
        ("18-40", 1): (0.1, 1.2), ("18-40", 0): (0.1, 1.0),
        ("40-60", 1): (0.1, 1.2), ("40-60", 0): (0.1, 1.0),
        ("60-80", 1): (0.1, 1.2), ("60-80", 0): (0.1, 1.0),
        ("80+",   1): (0.1, 1.2), ("80+",   0): (0.1, 1.0),
    },
    "FiO2": {  # Room air = 0.21; supplemental O2 means abnormal
        ("18-40", 1): (0.21, 0.30), ("18-40", 0): (0.21, 0.30),
        ("40-60", 1): (0.21, 0.30), ("40-60", 0): (0.21, 0.30),
        ("60-80", 1): (0.21, 0.30), ("60-80", 0): (0.21, 0.30),
        ("80+",   1): (0.21, 0.30), ("80+",   0): (0.21, 0.30),
    },
    "pH": {  # Arterial pH (Tietz); narrow physiological range
        ("18-40", 1): (7.35, 7.45), ("18-40", 0): (7.35, 7.45),
        ("40-60", 1): (7.35, 7.45), ("40-60", 0): (7.35, 7.45),
        ("60-80", 1): (7.35, 7.45), ("60-80", 0): (7.35, 7.45),
        ("80+",   1): (7.35, 7.45), ("80+",   0): (7.35, 7.45),
    },
    "PaCO2": {  # Arterial CO2, mmHg
        ("18-40", 1): (35, 45), ("18-40", 0): (35, 45),
        ("40-60", 1): (35, 45), ("40-60", 0): (35, 45),
        ("60-80", 1): (35, 45), ("60-80", 0): (35, 45),
        ("80+",   1): (35, 45), ("80+",   0): (35, 45),
    },
    "HCO3": {  # Bicarbonate, mEq/L
        ("18-40", 1): (22, 28), ("18-40", 0): (22, 28),
        ("40-60", 1): (22, 28), ("40-60", 0): (22, 28),
        ("60-80", 1): (22, 28), ("60-80", 0): (22, 28),
        ("80+",   1): (22, 28), ("80+",   0): (22, 28),
    },
    "EtCO2": {  # End-tidal CO2, mmHg
        ("18-40", 1): (35, 45), ("18-40", 0): (35, 45),
        ("40-60", 1): (35, 45), ("40-60", 0): (35, 45),
        ("60-80", 1): (35, 45), ("60-80", 0): (35, 45),
        ("80+",   1): (35, 45), ("80+",   0): (35, 45),
    },
    "BaseExcess": {  # mEq/L
        ("18-40", 1): (-2, 2), ("18-40", 0): (-2, 2),
        ("40-60", 1): (-2, 2), ("40-60", 0): (-2, 2),
        ("60-80", 1): (-2, 2), ("60-80", 0): (-2, 2),
        ("80+",   1): (-2, 2), ("80+",   0): (-2, 2),
    },
    "SaO2": {  # Arterial O2 saturation, %
        ("18-40", 1): (95, 100), ("18-40", 0): (95, 100),
        ("40-60", 1): (94, 100), ("40-60", 0): (94, 100),
        ("60-80", 1): (93, 100), ("60-80", 0): (93, 100),
        ("80+",   1): (92, 100), ("80+",   0): (92, 100),
    },
    "AST": {  # U/L (Tietz)
        ("18-40", 1): (10, 40), ("18-40", 0): (9, 32),
        ("40-60", 1): (10, 40), ("40-60", 0): (9, 32),
        ("60-80", 1): (10, 40), ("60-80", 0): (9, 32),
        ("80+",   1): (10, 40), ("80+",   0): (9, 32),
    },
    "Alkalinephos": {  # U/L
        ("18-40", 1): (44, 147), ("18-40", 0): (44, 147),
        ("40-60", 1): (44, 147), ("40-60", 0): (44, 147),
        ("60-80", 1): (44, 147), ("60-80", 0): (44, 147),
        ("80+",   1): (44, 147), ("80+",   0): (44, 147),
    },
    "Calcium": {  # mg/dL (total)
        ("18-40", 1): (8.5, 10.5), ("18-40", 0): (8.5, 10.5),
        ("40-60", 1): (8.5, 10.5), ("40-60", 0): (8.5, 10.5),
        ("60-80", 1): (8.5, 10.5), ("60-80", 0): (8.5, 10.5),
        ("80+",   1): (8.5, 10.5), ("80+",   0): (8.5, 10.5),
    },
    "Chloride": {  # mEq/L
        ("18-40", 1): (96, 106), ("18-40", 0): (96, 106),
        ("40-60", 1): (96, 106), ("40-60", 0): (96, 106),
        ("60-80", 1): (96, 106), ("60-80", 0): (96, 106),
        ("80+",   1): (96, 106), ("80+",   0): (96, 106),
    },
    "Bilirubin_direct": {  # mg/dL
        ("18-40", 1): (0.0, 0.3), ("18-40", 0): (0.0, 0.3),
        ("40-60", 1): (0.0, 0.3), ("40-60", 0): (0.0, 0.3),
        ("60-80", 1): (0.0, 0.3), ("60-80", 0): (0.0, 0.3),
        ("80+",   1): (0.0, 0.3), ("80+",   0): (0.0, 0.3),
    },
    "Magnesium": {  # mg/dL
        ("18-40", 1): (1.7, 2.2), ("18-40", 0): (1.7, 2.2),
        ("40-60", 1): (1.7, 2.2), ("40-60", 0): (1.7, 2.2),
        ("60-80", 1): (1.7, 2.2), ("60-80", 0): (1.7, 2.2),
        ("80+",   1): (1.7, 2.2), ("80+",   0): (1.7, 2.2),
    },
    "Phosphate": {  # mg/dL
        ("18-40", 1): (2.5, 4.5), ("18-40", 0): (2.5, 4.5),
        ("40-60", 1): (2.5, 4.5), ("40-60", 0): (2.5, 4.5),
        ("60-80", 1): (2.5, 4.5), ("60-80", 0): (2.5, 4.5),
        ("80+",   1): (2.5, 4.5), ("80+",   0): (2.5, 4.5),
    },
    "Potassium": {  # mEq/L
        ("18-40", 1): (3.5, 5.0), ("18-40", 0): (3.5, 5.0),
        ("40-60", 1): (3.5, 5.0), ("40-60", 0): (3.5, 5.0),
        ("60-80", 1): (3.5, 5.0), ("60-80", 0): (3.5, 5.0),
        ("80+",   1): (3.5, 5.0), ("80+",   0): (3.5, 5.0),
    },
    "TroponinI": {  # ng/mL (above 0.04 typically abnormal)
        ("18-40", 1): (0.0, 0.04), ("18-40", 0): (0.0, 0.04),
        ("40-60", 1): (0.0, 0.04), ("40-60", 0): (0.0, 0.04),
        ("60-80", 1): (0.0, 0.04), ("60-80", 0): (0.0, 0.04),
        ("80+",   1): (0.0, 0.04), ("80+",   0): (0.0, 0.04),
    },
    "Hct": {  # %
        ("18-40", 1): (40, 52), ("18-40", 0): (36, 48),
        ("40-60", 1): (40, 52), ("40-60", 0): (36, 48),
        ("60-80", 1): (38, 50), ("60-80", 0): (35, 47),
        ("80+",   1): (37, 49), ("80+",   0): (34, 46),
    },
    "Hgb": {  # g/dL
        ("18-40", 1): (13.5, 17.5), ("18-40", 0): (12.0, 16.0),
        ("40-60", 1): (13.5, 17.5), ("40-60", 0): (12.0, 16.0),
        ("60-80", 1): (13.0, 17.0), ("60-80", 0): (11.5, 15.5),
        ("80+",   1): (12.5, 16.5), ("80+",   0): (11.0, 15.0),
    },
    "PTT": {  # seconds
        ("18-40", 1): (25, 35), ("18-40", 0): (25, 35),
        ("40-60", 1): (25, 35), ("40-60", 0): (25, 35),
        ("60-80", 1): (25, 35), ("60-80", 0): (25, 35),
        ("80+",   1): (25, 35), ("80+",   0): (25, 35),
    },
    "Fibrinogen": {  # mg/dL
        ("18-40", 1): (200, 400), ("18-40", 0): (200, 400),
        ("40-60", 1): (200, 400), ("40-60", 0): (200, 400),
        ("60-80", 1): (200, 400), ("60-80", 0): (200, 400),
        ("80+",   1): (200, 400), ("80+",   0): (200, 400),
    },
}

NORMAL_RANGE_FEATURE_SUFFIXES = [
    "_above_normal", "_below_normal", "_deviation_from_normal",
    "_abs_deviation_from_normal", "_drift_from_normal",
]

# ── Lead-Time Importance Bins ─────────────────────────────────────────────────
# For per-feature importance analysis stratified by hours-before-onset.
# Edges are right-exclusive: [0, 3), [3, 6), [6, 12), [12, 24), [24, 48).

LEADUP_BIN_EDGES = [0, 3, 6, 12, 24, 48]
LEADUP_BIN_LABELS = ["0-3h", "3-6h", "6-12h", "12-24h", "24-48h"]
LEADUP_NEVER_LABEL = "never"  # never-sepsis rows (used as IV negatives)
LEADUP_OUT_OF_RANGE_LABEL = "out_of_range"  # >=48h before onset

# ── Leading-Indicator Feature Filter ─────────────────────────────────────────
# Symptom / static-cohort-marker / clinician-action features carry signal that
# separates "already-sick" patients but provide no genuine lead time (they are
# flat-but-elevated 48h before onset, or encode that the team already suspected
# sepsis). The filter below keeps only TRAJECTORY signals: vital trends, drift
# from baseline, rate-of-change, and vital-derived deterioration scores.
#
# Evidence: onset-aligned trajectory analysis — 23/34 raw features are static
# cohort markers, only Resp/Alkalinephos genuinely ramp. Ablating the static
# block wholesale drops AUROC ~0.05 (the inflated, non-leading portion).

USE_LEADING_INDICATORS_ONLY = True

# Lab-level encodings that represent the *absolute level* of a reactively-drawn
# lab → static cohort marker. Dropped (their drift/change versions are kept).
LEADING_DROP_LAB_LEVEL_SUFFIXES = [
    "_avg_6h", "_min_6h", "_max_6h", "_std_6h",
    "_above_normal", "_below_normal",
    "_deviation_from_normal", "_abs_deviation_from_normal",
]
# Trajectory encodings kept for every signal (the genuine leading indicators).
LEADING_KEEP_TRAJ_SUFFIXES = ["_drift_from_normal", "_drift_from_normal_6h", "_hourly_change"]
# Clinician-action leakage (testing frequency / missingness) — dropped for all.
LEADING_DROP_LEAKAGE_SUFFIXES = ["_measured", "_hours_since"]
# Feature families dropped entirely (intervention-driven or direct symptom).
LEADING_DROP_FEATURE_PREFIXES = [
    "FiO2",                  # intervention-driven (supplemental O2 decision)
    "lactate_bp_ratio",      # symptom composite (hyperlactatemia + hypotension = shock)
    "inflammation_score",    # screening score for the septic STATE
    "sepsis_screen_score",   # screening score for the septic STATE
]

# ── Alert Aggregation ─────────────────────────────────────────────────────────

MIN_CONSECUTIVE_HOURS = 3  # Require N sustained hours above threshold to flag a patient

CV_N_ITER = 5

CV_FOLDS = 3
INNER_CV_FOLDS = 3

# Drop one feature from any pair with |Pearson r| >= this value, keeping the
# higher-IV feature. Applied inside CV folds for honest validation metrics.
COLLINEARITY_THRESHOLD = 0.80
ENABLE_COLLINEARITY_PRUNING = True
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.30
