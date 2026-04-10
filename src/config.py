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

# Only roll deviations for vitals + key labs (not all 34 — rarely-measured lab
# deviations are mostly zeros, rolling stats on zeros is pure noise)
KEY_LAB_COLS = ["Lactate", "WBC", "Creatinine", "Platelets"]
KEY_DEVIATION_COLS = [f"{v}_baseline_dev" for v in VITAL_COLS + KEY_LAB_COLS]
ROLLING_COLS = VITAL_COLS + KEY_LAB_COLS + CLINICAL_SCORE_COLS + KEY_DEVIATION_COLS

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

NORMAL_RANGE_COLS = [
    "HR", "Resp", "Temp", "SBP", "MAP", "O2Sat", "DBP",
    "WBC", "Creatinine", "Lactate", "Platelets", "BUN", "Glucose", "Bilirubin_total",
]

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
}

NORMAL_RANGE_FEATURE_SUFFIXES = ["_above_normal", "_below_normal", "_deviation_from_normal"]

# ── Alert Aggregation ─────────────────────────────────────────────────────────

MIN_CONSECUTIVE_HOURS = 3  # Require N sustained hours above threshold to flag a patient

CV_N_ITER = 5

CV_FOLDS = 3
INNER_CV_FOLDS = 3
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.30
