"""Evaluation, slice analysis, and visualization generator for Heart Disease models."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix

from src.config import (
    BEST_MODEL_PATH,
    REPORTS_DIR,
    METRICS_PATH,
    TARGET,
    ALL_FEATURES,
)
from src.data_loader import load_raw_data, clean_and_validate_data, split_data


def generate_evaluation_visualizations():
    """Generate ROC curves, confusion matrix heatmap, and feature importance plots."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    raw_df = load_raw_data()
    clean_df = clean_and_validate_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(clean_df)

    # Load metrics
    with open(METRICS_PATH, "r") as f:
        metrics_data = json.load(f)

    # 1. Plot ROC Curves
    plt.figure(figsize=(9, 6))
    for model_name, info in metrics_data["models"].items():
        if model_name == "Majority Class Baseline":
            continue
        try:
            # We can plot test ROC from the stored models or predict
            pass
        except Exception:
            pass

    # Load best model pipeline
    best_pipeline = joblib.load(BEST_MODEL_PATH)
    y_probas = best_pipeline.predict_proba(X_test)[:, 1]
    y_preds = best_pipeline.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_probas)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"Best Model ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curve - Test Set", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    roc_path = REPORTS_DIR / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Held-out Test Cohort)", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Healthy (0)", "Disease (1)"], fontsize=10)
    plt.yticks(tick_marks, ["Healthy (0)", "Disease (1)"], fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )
    plt.ylabel("True Clinical Diagnosis", fontsize=11, fontweight="bold")
    plt.xlabel("Model Predicted Diagnosis", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 3. Feature Importance Plot
    classifier = best_pipeline.named_steps["classifier"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            ohe = trans.named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(cols)
            feature_names.extend(cat_names)

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        idx = np.argsort(importances)[::-1][:10]
        top_feats = [feature_names[i] for i in idx]
        top_imps = importances[idx]

        plt.figure(figsize=(9, 5))
        plt.barh(range(len(top_feats)), top_imps[::-1], color="#3B82F6")
        plt.yticks(range(len(top_feats)), top_feats[::-1], fontsize=10)
        plt.xlabel("Gini Feature Importance", fontsize=11, fontweight="bold")
        plt.title("Top 10 Most Predictive Clinical Features", fontsize=13, fontweight="bold")
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        fi_path = REPORTS_DIR / "feature_importance.png"
        plt.savefig(fi_path, dpi=300)
        plt.close()

    # 4. Slice-based demographic evaluation (Sex & Age slices)
    slice_results = {}
    test_eval_df = X_test.copy()
    test_eval_df["y_true"] = y_test.values
    test_eval_df["y_pred"] = y_preds

    # Subpopulation: Sex
    for sex_val, label in [(1, "Male"), (0, "Female")]:
        sub = test_eval_df[test_eval_df["sex"] == sex_val]
        if len(sub) > 0:
            sub_acc = float((sub["y_true"] == sub["y_pred"]).mean())
            slice_results[f"Sex_{label}"] = {
                "count": len(sub),
                "accuracy": round(sub_acc, 4),
                "positive_rate": round(float(sub["y_true"].mean()), 4),
            }

    # Subpopulation: Age (< 55 vs >= 55)
    for age_cond, label in [(test_eval_df["age"] < 55, "Age < 55"), (test_eval_df["age"] >= 55, "Age >= 55")]:
        sub = test_eval_df[age_cond]
        if len(sub) > 0:
            sub_acc = float((sub["y_true"] == sub["y_pred"]).mean())
            slice_results[label] = {
                "count": len(sub),
                "accuracy": round(sub_acc, 4),
                "positive_rate": round(float(sub["y_true"].mean()), 4),
            }

    slice_path = REPORTS_DIR / "demographic_slices.json"
    with open(slice_path, "w") as f:
        json.dump(slice_results, f, indent=2)

    print("Generated evaluation visualizations in reports/ directory.")
    print("Demographic slice results:\n", json.dumps(slice_results, indent=2))


if __name__ == "__main__":
    generate_evaluation_visualizations()
