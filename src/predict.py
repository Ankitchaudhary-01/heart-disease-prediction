import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Union, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    BASELINE_MODEL_PATH,
    ALL_FEATURES,
    RISK_THRESHOLDS,
    FEATURE_LABELS,
    VALUE_MAPS,
)


class HeartDiseasePredictor:
    """Predictor class providing inference, risk stratification, and clinical guidance."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else BEST_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Please run `python -m src.train` first."
            )
        self.pipeline = joblib.load(self.model_path)

    def _determine_risk_tier(self, probability: float) -> Dict[str, Any]:
        """Classify probability into standardized clinical risk tier."""
        if probability < RISK_THRESHOLDS["low"]:
            return {
                "tier": "Low Risk",
                "color": "#10B981",  # Green
                "badge": "success",
                "recommendation": "Maintain heart-healthy lifestyle, regular exercise, and routine annual checkups.",
            }
        elif probability <= RISK_THRESHOLDS["moderate"]:
            return {
                "tier": "Moderate Risk",
                "color": "#F59E0B",  # Amber/Yellow
                "badge": "warning",
                "recommendation": "Schedule follow-up cardiology evaluation, monitor lipid profile & BP closely, and initiate lifestyle modifications.",
            }
        else:
            return {
                "tier": "High Risk",
                "color": "#EF4444",  # Red
                "badge": "danger",
                "recommendation": "Prompt comprehensive cardiological consultation, non-invasive imaging, and formal diagnostic workup recommended.",
            }

    def _generate_clinical_flags(self, patient_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify key clinical risk factors from input data."""
        flags = []

        # Age
        age = patient_data.get("age", 0)
        if age >= 60:
            flags.append({"factor": "Advanced Age", "detail": f"{age} years (Elevated baseline CAD prevalence)"})

        # Blood Pressure
        bp = patient_data.get("trestbps", 0)
        if bp >= 140:
            flags.append({"factor": "Stage 2 Hypertension", "detail": f"{bp} mm Hg (Accelerates arterial shear damage)"})
        elif bp >= 130:
            flags.append({"factor": "Stage 1 Hypertension", "detail": f"{bp} mm Hg (Elevated vascular stress)"})

        # Serum Cholesterol
        chol = patient_data.get("chol", 0)
        if chol >= 240:
            flags.append({"factor": "High Hypercholesterolemia", "detail": f"{chol} mg/dl (Substantially increases atheroma burden)"})
        elif chol >= 200:
            flags.append({"factor": "Borderline High Cholesterol", "detail": f"{chol} mg/dl (Above optimal range of <200 mg/dl)"})

        # Fasting Blood Sugar
        if patient_data.get("fbs", 0) == 1:
            flags.append({"factor": "Hyperglycemia (FBS > 120)", "detail": "Indicates impaired fasting glucose or diabetes"})

        # Exercise Induced Angina
        if patient_data.get("exang", 0) == 1:
            flags.append({"factor": "Exercise-Induced Angina", "detail": "Provocable ischemic symptom during physical exertion"})

        # ST Depression (Oldpeak)
        oldpeak = patient_data.get("oldpeak", 0.0)
        if oldpeak >= 2.0:
            flags.append({"factor": "Marked ST Depression", "detail": f"{oldpeak} mm (Strong indicator of myocardial ischemia)"})
        elif oldpeak >= 1.0:
            flags.append({"factor": "Moderate ST Depression", "detail": f"{oldpeak} mm (Abnormal repolarization under stress)"})

        # Major Vessels Colored by Fluoroscopy
        ca = patient_data.get("ca", 0)
        if ca >= 1:
            flags.append({"factor": "Coronary Vessel Narrowing", "detail": f"{ca} major vessel(s) with significant stenosis"})

        # Thallium Defect
        thal = patient_data.get("thal", 2)
        if thal == 3:
            flags.append({"factor": "Reversible Perfusion Defect", "detail": "Indicates inducible myocardial ischemia on nuclear scintigraphy"})
        elif thal == 1:
            flags.append({"factor": "Fixed Perfusion Defect", "detail": "Suggests prior myocardial infarction scar"})

        # Resting ECG
        restecg = patient_data.get("restecg", 0)
        if restecg == 2:
            flags.append({"factor": "Left Ventricular Hypertrophy (LVH)", "detail": "ECG shows cardiac muscle thickening"})
        elif restecg == 1:
            flags.append({"factor": "ECG ST-T Wave Abnormality", "detail": "Repolarization abnormality detected"})

        return flags

    def predict_single(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict heart disease diagnosis and risk assessment for a single patient record.
        """
        # Ensure all required features are present
        missing = [f for f in ALL_FEATURES if f not in patient_dict]
        if missing:
            raise ValueError(f"Missing required clinical features: {missing}")

        df_input = pd.DataFrame([patient_dict])[ALL_FEATURES]

        # Pipeline prediction
        prediction = int(self.pipeline.predict(df_input)[0])

        if hasattr(self.pipeline.named_steps["classifier"], "predict_proba"):
            probabilities = self.pipeline.predict_proba(df_input)[0]
            prob_disease = float(probabilities[1])
            confidence = float(np.max(probabilities))
        else:
            prob_disease = 1.0 if prediction == 1 else 0.0
            confidence = 1.0

        risk_info = self._determine_risk_tier(prob_disease)
        clinical_flags = self._generate_clinical_flags(patient_dict)

        return {
            "prediction": prediction,
            "diagnosis": "Heart Disease Indicated" if prediction == 1 else "No Heart Disease Detected",
            "probability": round(prob_disease, 4),
            "probability_percentage": f"{round(prob_disease * 100, 1)}%",
            "confidence": round(confidence, 4),
            "risk_tier": risk_info["tier"],
            "risk_color": risk_info["color"],
            "risk_badge": risk_info["badge"],
            "recommendation": risk_info["recommendation"],
            "clinical_flags": clinical_flags,
            "input_features": patient_dict,
        }

    def predict_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Predict heart disease for batch cohort dataframe and attach diagnosis columns.
        """
        df_copy = df_input.copy()
        missing = [f for f in ALL_FEATURES if f not in df_copy.columns]
        if missing:
            raise ValueError(f"Input dataframe missing required columns: {missing}")

        features_df = df_copy[ALL_FEATURES]
        preds = self.pipeline.predict(features_df)

        df_copy["predicted_target"] = preds
        df_copy["predicted_diagnosis"] = np.where(preds == 1, "Heart Disease Indicated", "Healthy / Low Risk")

        if hasattr(self.pipeline.named_steps["classifier"], "predict_proba"):
            probas = self.pipeline.predict_proba(features_df)[:, 1]
            df_copy["disease_probability"] = np.round(probas, 4)
            df_copy["probability_pct"] = np.round(probas * 100, 1)

            def get_tier(p):
                if p < RISK_THRESHOLDS["low"]:
                    return "Low Risk"
                elif p <= RISK_THRESHOLDS["moderate"]:
                    return "Moderate Risk"
                return "High Risk"

            df_copy["risk_tier"] = df_copy["disease_probability"].apply(get_tier)
        else:
            df_copy["disease_probability"] = preds.astype(float)
            df_copy["probability_pct"] = preds * 100.0
            df_copy["risk_tier"] = np.where(preds == 1, "High Risk", "Low Risk")

        return df_copy


if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    # High-risk profile test
    high_risk_patient = {
        "age": 63,
        "sex": 1,
        "cp": 0,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 1,
        "thal": 3,
    }
    result = predictor.predict_single(high_risk_patient)
    print("Prediction Result:")
    print(f"  Diagnosis: {result['diagnosis']}")
    print(f"  Probability: {result['probability_percentage']}")
    print(f"  Risk Tier: {result['risk_tier']}")
    print(f"  Clinical Flags: {len(result['clinical_flags'])}")
