"""Unit tests for Heart Disease Prediction data, pipeline, and inference modules."""

import unittest
import pandas as pd
import numpy as np

from src.config import ALL_FEATURES, RAW_DATA_PATH, BEST_MODEL_PATH
from src.data_loader import load_raw_data, clean_and_validate_data, split_data
from src.pipeline import create_preprocessor, create_model_pipeline
from src.predict import HeartDiseasePredictor
from sklearn.linear_model import LogisticRegression


def test_data_loader_and_cleaner():
    """Test loading and data cleaning validations."""
    df_raw = load_raw_data()
    assert isinstance(df_raw, pd.DataFrame)
    assert len(df_raw) > 0

    df_clean = clean_and_validate_data(df_raw)
    # Check no nulls remain in features
    assert df_clean[ALL_FEATURES].isnull().sum().sum() == 0
    # Check ca does not have invalid sentinel code 4
    assert (df_clean["ca"] == 4).sum() == 0
    # Check thal does not have invalid sentinel code 0
    assert (df_clean["thal"] == 0).sum() == 0
    # Check binary target
    assert set(df_clean["target"].unique()).issubset({0, 1})


def test_split_data():
    """Test strict train-test split."""
    df_raw = load_raw_data()
    df_clean = clean_and_validate_data(df_raw)
    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2, random_state=42)

    assert len(X_train) + len(X_test) == len(df_clean)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert list(X_train.columns) == ALL_FEATURES


def test_preprocessor_pipeline():
    """Test scikit-learn preprocessing ColumnTransformer."""
    preprocessor = create_preprocessor()
    df_raw = load_raw_data()
    df_clean = clean_and_validate_data(df_raw)
    X_train, _, _, _ = split_data(df_clean)

    transformed = preprocessor.fit_transform(X_train)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(X_train)
    # Numerical (6) + one-hot encoded categories
    assert transformed.shape[1] > len(ALL_FEATURES)


def test_predictor_single_and_batch():
    """Test HeartDiseasePredictor inference with high risk and low risk profiles."""
    predictor = HeartDiseasePredictor()

    patient_healthy = {
        "age": 32,
        "sex": 0,
        "cp": 2,
        "trestbps": 110,
        "chol": 175,
        "fbs": 0,
        "restecg": 0,
        "thalach": 180,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }

    patient_high_risk = {
        "age": 67,
        "sex": 1,
        "cp": 0,
        "trestbps": 160,
        "chol": 286,
        "fbs": 1,
        "restecg": 2,
        "thalach": 115,
        "exang": 1,
        "oldpeak": 3.2,
        "slope": 1,
        "ca": 2,
        "thal": 3,
    }

    res_healthy = predictor.predict_single(patient_healthy)
    res_high_risk = predictor.predict_single(patient_high_risk)

    assert res_healthy["prediction"] == 0
    assert res_healthy["risk_tier"] == "Low Risk"
    assert res_healthy["probability"] < 0.25

    assert res_high_risk["prediction"] == 1
    assert res_high_risk["risk_tier"] == "High Risk"
    assert res_high_risk["probability"] > 0.60
    assert len(res_high_risk["clinical_flags"]) > 0

    # Batch test
    batch_df = pd.DataFrame([patient_healthy, patient_high_risk])
    batch_results = predictor.predict_batch(batch_df)
    assert "predicted_diagnosis" in batch_results.columns
    assert "disease_probability" in batch_results.columns
    assert "risk_tier" in batch_results.columns
    assert len(batch_results) == 2


def test_database_persistence():
    """Test saving, retrieving, and calculating statistics from SQLite database."""
    from src.database import (
        save_patient_record,
        save_batch_records,
        get_all_records,
        get_database_stats,
        delete_record,
    )

    test_patient = {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 135,
        "chol": 245,
        "fbs": 0,
        "restecg": 1,
        "thalach": 145,
        "exang": 1,
        "oldpeak": 1.8,
        "slope": 1,
        "ca": 1,
        "thal": 3,
    }
    dummy_pred = {
        "prediction": 1,
        "diagnosis": "Heart Disease Indicated",
        "probability": 0.82,
        "risk_tier": "High Risk",
        "clinical_flags": [{"factor": "Hypertension", "detail": "135 mm Hg"}],
    }

    # Save single record
    rec_id = save_patient_record(test_patient, dummy_pred, patient_id="TEST-001", patient_name="Test Patient")
    assert rec_id > 0

    # Retrieve records
    df_records = get_all_records(limit=10)
    assert len(df_records) > 0
    assert "patient_id" in df_records.columns
    assert "TEST-001" in df_records["patient_id"].values

    # Test stats
    stats = get_database_stats()
    assert stats["total_records"] >= 1
    assert stats["high_risk_count"] >= 1

    # Cleanup test record
    deleted = delete_record(rec_id)
    assert deleted is True


if __name__ == "__main__":
    test_data_loader_and_cleaner()
    test_split_data()
    test_preprocessor_pipeline()
    test_predictor_single_and_batch()
    test_database_persistence()
    print("All unit tests passed successfully!")
