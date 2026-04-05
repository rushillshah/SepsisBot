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

ALL_FEATURE_COLS = VITAL_COLS + LAB_COLS
ALL_INPUT_COLS = ALL_FEATURE_COLS + DEMOGRAPHIC_COLS + [TIME_COL]

# ── Feature Engineering ────────────────────────────────────────────────────────

ROLLING_WINDOW_HOURS = 6
ROLLING_STATS = ["mean", "min", "max", "std"]
ROLLING_COLS = VITAL_COLS + ["Lactate", "WBC", "Creatinine", "Platelets", "MAP"]

# ── Model Hyperparameters ──────────────────────────────────────────────────────

XGBOOST_PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
}

# ── Features to Exclude (site-specific confounders) ────────────────────────

EXCLUDED_FEATURES = ["Unit1", "Unit2", "HospAdmTime"]

# ── Improved Model Hyperparameters ─────────────────────────────────────────

XGBOOST_PARAM_GRID_V2 = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500],
    "min_child_weight": [5, 10, 20],
    "gamma": [0.1, 0.5, 1.0],
    "reg_alpha": [0.1, 1.0],
    "reg_lambda": [1.0, 5.0],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
}

CV_N_ITER = 30

CV_FOLDS = 5
RANDOM_STATE = 42
TARGET_AUROC = 0.80
