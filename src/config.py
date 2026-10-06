"""Configuration module for the Heart Disease Prediction project."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATA_PATH = DATA_DIR / "heart.csv"
SAMPLE_BATCH_PATH = DATA_DIR / "sample_test_batch.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics_summary.json"

# Features definitions
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
TARGET = "target"
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Feature UI labels and mappings
FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG Results",
    "thalach": "Max Heart Rate Achieved (bpm)",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "slope": "Slope of Peak Exercise ST Segment",
    "ca": "Number of Major Vessels Colored (0-3)",
    "thal": "Thallium Stress Test Result",
}

# Human-readable value maps
VALUE_MAPS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Chest Pain",
        3: "Asymptomatic",
    },
    "fbs": {0: "No (<= 120 mg/dl)", 1: "Yes (> 120 mg/dl)"},
    "restecg": {
        0: "Normal Sinus",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy (LVH)",
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "thal": {
        1: "Normal Perfusion",
        2: "Fixed Defect",
        3: "Reversible Defect",
    },
}

# Clinical risk thresholds
RISK_THRESHOLDS = {
    "low": 0.25,      # < 25%: Low Risk
    "moderate": 0.60, # 25% - 60%: Moderate Risk
    "high": 1.00,     # > 60%: High Risk
}

# Random seed for reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
