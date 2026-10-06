"""Model training and evaluation pipeline for Heart Disease Prediction."""

import json
import logging
import time
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    BEST_MODEL_PATH,
    BASELINE_MODEL_PATH,
    METRICS_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    CV_FOLDS,
)
from src.data_loader import load_raw_data, clean_and_validate_data, split_data
from src.pipeline import create_model_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_candidate_models() -> Dict[str, Any]:
    """Define candidate models including baselines and advanced ensemble models."""
    return {
        "Majority Class Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=4,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "Support Vector Classifier (RBF)": CalibratedClassifierCV(
            estimator=SVC(C=1.0, kernel="rbf", random_state=RANDOM_STATE),
            ensemble=False,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
        ),
    }


def evaluate_cross_validation(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
) -> Dict[str, Any]:
    """Perform Stratified K-Fold Cross-Validation and compute 95% confidence intervals."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
    )

    summary = {}
    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        # 95% Confidence Interval: mean +/- 1.96 * (std / sqrt(k))
        ci_half = float(1.96 * (std / np.sqrt(cv_folds)))
        summary[metric] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_95": [round(mean - ci_half, 4), round(mean + ci_half, 4)],
        }

    return summary


def evaluate_test_set(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate fitted model pipeline on held-out test set."""
    start_time = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    inference_time_ms = (time.perf_counter() - start_time) * 1000 / len(X_test)

    # Probabilities if supported
    has_proba = hasattr(pipeline.named_steps["classifier"], "predict_proba")
    if has_proba:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_proba))
        pr_auc = float(average_precision_score(y_test, y_proba))
        brier = float(brier_score_loss(y_test, y_proba))
    else:
        y_proba = None
        roc_auc = 0.5
        pr_auc = float(np.mean(y_test))
        brier = 0.5

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "per_sample_latency_ms": round(inference_time_ms, 4),
    }


def extract_feature_importance(pipeline) -> Dict[str, float]:
    """Extract feature importance or model coefficients for explainability."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Retrieve one-hot encoded and numerical feature names
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            ohe = trans.named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(cols)
            feature_names.extend(cat_names)

    importance_dict = {}
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        for feat, imp in zip(feature_names, importances):
            importance_dict[feat] = round(float(imp), 4)
    elif hasattr(classifier, "coef_"):
        coefs = np.abs(classifier.coef_[0])
        # normalize to sum to 1 for comparative view
        norm_coefs = coefs / np.sum(coefs) if np.sum(coefs) > 0 else coefs
        for feat, imp in zip(feature_names, norm_coefs):
            importance_dict[feat] = round(float(imp), 4)

    # Sort descending
    sorted_importances = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_importances


def train_and_compare_all_models() -> Tuple[Any, Dict[str, Any]]:
    """
    Execute full training, cross-validation comparison, and selection of best model.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and clean
    raw_df = load_raw_data()
    clean_df = clean_and_validate_data(raw_df)

    # Strict featurization ordering: split first
    X_train, X_test, y_train, y_test = split_data(clean_df)

    models = get_candidate_models()
    results = {}
    fitted_pipelines = {}

    logger.info("Starting model training and cross-validation across %d models...", len(models))

    for name, clf in models.items():
        logger.info("Training candidate: %s", name)
        pipeline = create_model_pipeline(clf)

        # 1. Cross-validation on training data only
        cv_summary = evaluate_cross_validation(pipeline, X_train, y_train)

        # 2. Fit pipeline on full training set
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        training_time_s = time.perf_counter() - t0

        # 3. Test set evaluation
        test_summary = evaluate_test_set(pipeline, X_test, y_test)

        # 4. Feature importance if applicable
        feat_imp = extract_feature_importance(pipeline)

        results[name] = {
            "cv_metrics": cv_summary,
            "test_metrics": test_summary,
            "training_time_seconds": round(training_time_s, 4),
            "feature_importance_top10": dict(list(feat_imp.items())[:10]) if feat_imp else {},
        }
        fitted_pipelines[name] = pipeline

    # Select best model based on validation ROC-AUC and F1-score (excluding DummyClassifier)
    candidate_scores = {
        name: results[name]["cv_metrics"]["roc_auc"]["mean"]
        for name in models.keys()
        if name != "Majority Class Baseline"
    }
    best_model_name = max(candidate_scores, key=candidate_scores.get)
    logger.info("Best model selected: %s (CV ROC-AUC: %.4f)", best_model_name, candidate_scores[best_model_name])

    # Save artifacts
    best_pipeline = fitted_pipelines[best_model_name]
    baseline_pipeline = fitted_pipelines["Logistic Regression"]

    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    joblib.dump(baseline_pipeline, BASELINE_MODEL_PATH)
    logger.info("Saved best model to %s", BEST_MODEL_PATH)
    logger.info("Saved baseline model to %s", BASELINE_MODEL_PATH)

    output_summary = {
        "best_model_name": best_model_name,
        "selection_criteria": "Stratified 5-Fold Cross-Validation ROC-AUC",
        "data_summary": {
            "total_records": len(clean_df),
            "train_records": len(X_train),
            "test_records": len(X_test),
            "positive_class_prevalence": round(float(y_train.mean()), 4),
        },
        "models": results,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(output_summary, f, indent=2)
    logger.info("Saved metrics summary to %s", METRICS_PATH)

    return best_pipeline, output_summary


if __name__ == "__main__":
    train_and_compare_all_models()
