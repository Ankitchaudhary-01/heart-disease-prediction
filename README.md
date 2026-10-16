# CardioPredict AI — Heart Disease Risk Prediction & Clinical Decision Support System

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-red.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

CardioPredict AI is a production-grade machine learning system designed to predict coronary artery disease (CAD) risk using clinical indicators, electrocardiographic findings, and treadmill exercise stress test parameters.

Developed adhering to strict **Machine Learning Best Practices** (strict featurization ordering, leak-free preprocessing pipelines, stratified cross-validation, multi-model benchmarking, demographic subpopulation auditing, and clinically calibrated risk probability stratification).

---

##  Table of Contents
- [Project Architecture](#-project-architecture)
- [Clinical Dataset & Features](#-clinical-dataset--features)
- [Machine Learning Methodology](#-machine-learning-methodology)
- [Model Comparison & Benchmark Results](#-model-comparison--benchmark-results)
- [Installation & Quickstart](#-installation--quickstart)
- [Usage Guide](#-usage-guide)
  - [1. Launching the Interactive Web Application](#1-launching-the-interactive-web-application)
  - [2. Command-Line Inference](#2-command-line-inference)
  - [3. Batch Cohort Screening](#3-batch-cohort-screening)
  - [4. Re-training Models](#4-re-training-models)
- [Automated Testing](#-automated-testing)
- [Demographic Fairness & Slice Analysis](#-demographic-fairness--slice-analysis)
- [Clinical Decision Support & Disclaimer](#-clinical-decision-support--disclaimer)

---

## Project Architecture

```
heart_disease_prediction/
├── README.md                          # Complete project documentation & guide
├── requirements.txt                    # Pinned Python dependencies
├── app.py                             # Interactive Streamlit Web Application
├── data/
│   ├── heart.csv                      # Authentic UCI Cleveland Cardiology dataset
│   ├── cardiopredict.db               # SQLite database for persistent patient records
│   ├── sample_test_batch.csv          # Sample 20-patient cohort for batch testing
│   └── data_dictionary.md             # Clinical dictionary with physiological definitions
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configurations, feature schemas, thresholds, paths
│   ├── database.py                    # SQLite patient persistence, audit logs & analytics
│   ├── data_loader.py                 # Data hygiene, sentinel code cleaning, stratified splitting
│   ├── pipeline.py                    # Leak-free ColumnTransformer & Pipeline constructors
│   ├── train.py                       # 5-Fold Stratified CV, multi-model evaluation, serialization
│   ├── predict.py                     # Inference engine, clinical risk flags, recommendations
│   └── evaluate.py                    # ROC curves, confusion matrix, slice analysis generator
├── models/
│   ├── best_model.joblib              # Winning model pipeline (Random Forest)
│   ├── baseline_model.joblib          # Interpretable baseline (Logistic Regression)
│   └── metrics_summary.json           # Comprehensive cross-validation & test metrics
├── reports/
│   ├── roc_curve.png                  # Receiver Operating Characteristic plot
│   ├── confusion_matrix.png           # Test cohort confusion matrix heatmap
│   ├── feature_importance.png         # Top 10 predictive features
│   └── demographic_slices.json        # Subpopulation accuracy across age & sex
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py               # Unit tests covering data, pipeline, and inference
└── notebooks/
    └── exploratory_analysis.py        # Complete EDA and clinical data story script
```

---

##  Clinical Dataset & Features

The model is grounded in the canonical **Cleveland Clinic Heart Disease** database (303 patients, 13 clinical predictors):

| Parameter | Type | Valid Range | Clinical Interpretation |
| :--- | :--- | :--- | :--- |
| `age` | Numerical | 29 – 77 yrs | Patient age in years |
| `sex` | Categorical | 0 = Female, 1 = Male | Biological sex |
| `cp` | Categorical | 0 – 3 | Chest pain type (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic) |
| `trestbps` | Numerical | 94 – 200 mm Hg | Resting systolic blood pressure on hospital admission |
| `chol` | Numerical | 126 – 564 mg/dl | Serum cholesterol level (<200 desirable) |
| `fbs` | Categorical | 0 = $\le 120$, 1 = $> 120$ | Fasting blood sugar elevated test (>120 mg/dl indicates hyperglycemia) |
| `restecg` | Categorical | 0 – 2 | Resting electrocardiogram (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy) |
| `thalach` | Numerical | 71 – 202 bpm | Maximum heart rate achieved during exercise treadmill test |
| `exang` | Categorical | 0 = No, 1 = Yes | Exercise-induced angina (provoked myocardial ischemia) |
| `oldpeak` | Numerical | 0.0 – 6.2 mm | ST depression induced by exercise relative to rest |
| `slope` | Categorical | 0 – 2 | Slope of peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping) |
| `ca` | Numerical | 0 – 3 vessels | Number of major coronary arteries colored by fluoroscopy |
| `thal` | Categorical | 1, 2, 3 | Thallium perfusion scintigraphy (1: Normal, 2: Fixed defect, 3: Reversible defect) |
| **`target`** | Target | 0 = Normal, 1 = Disease | **1: Coronary Heart Disease present**, **0: Healthy / No disease** |

---

##  Machine Learning Methodology

1. **Data Hygiene & Domain Validation**:
   - Dropped duplicate records to prevent data contamination across train and test folds.
   - Handled missing sentinel codes: `ca=4` (fluoroscopy missing) and `thal=0` (scintigraphy unrecorded) imputed using domain modes.
   - Target aligned so that `target=1` indicates Heart Disease Presence, correlating positively with clinical markers (`oldpeak`, `ca`, `exang`).

2. **Strict Featurization Ordering**:
   - Dataset is partitioned into $80\%$ Training and $20\%$ Test cohorts **before** any transformations are applied.
   - Preprocessing is encapsulated inside a scikit-learn `Pipeline` and `ColumnTransformer`:
     - **Numerical**: Median imputation $\to$ `StandardScaler`
     - **Categorical**: Mode imputation $\to$ `OneHotEncoder(handle_unknown='ignore')`

3. **Multi-Model Benchmark & Cross-Validation**:
   - Evaluated 6 diverse architectures using **5-Fold Stratified Cross-Validation**:
     - *Baseline*: Majority Class Classifier & Logistic Regression ($L_2$ regularization)
     - *Ensemble Models*: Random Forest Classifier & Gradient Boosting Classifier
     - *Non-linear Classifiers*: Support Vector Classifier (RBF kernel, calibrated probabilities) & K-Nearest Neighbors

---

## Model Comparison & Benchmark Results

### 5-Fold Stratified Cross-Validation & Test Cohort Evaluation:

| Model Architecture | 5-Fold CV ROC-AUC (Mean ± Std) | Test Accuracy | Test Precision | Test Recall (Sensitivity) | Test Specificity | Test F1-Score | Test ROC-AUC | Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Winner)** | **0.9109 ± 0.0435** | 78.69% | 75.47% | 71.43% | 84.85% | 0.7547 | 0.8799 | 0.44 ms |
| **Logistic Regression** | **0.9049 ± 0.0477** | **83.61%** | **82.14%** | **82.14%** | 84.85% | **0.8214** | **0.9242** | **0.06 ms** |
| **Support Vector Classifier (RBF)** | 0.9005 ± 0.0388 | 80.33% | 76.92% | 71.43% | 87.88% | 0.7692 | 0.8994 | 0.08 ms |
| **K-Nearest Neighbors** | 0.8861 ± 0.0420 | 86.89% | 84.62% | 78.57% | 93.94% | 0.8462 | 0.9351 | 0.09 ms |
| **Gradient Boosting** | 0.8769 ± 0.0463 | 78.69% | 75.47% | 71.43% | 84.85% | 0.7547 | 0.8810 | 0.08 ms |
| **Majority Baseline** | 0.5000 ± 0.0000 | 54.10% | 0.00% | 0.00% | 100.0% | 0.0000 | 0.5000 | 0.01 ms |

> **Key Finding**: The Random Forest ensemble achieved the highest generalizable cross-validation ROC-AUC (**0.9109**), while Logistic Regression provided optimal calibrated probability estimates and direct feature interpretability.

---

## Installation & Quickstart

### Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- macOS / Linux / Windows

### Setup Environment

**Prerequisites**

* Python 3.10 or later
* Git

**1. Clone the repository**

```bash
git clone https://github.com/Ankitchaudhary-01/heart-disease-prediction.git
cd heart-disease-prediction
```

**2. Create and activate a virtual environment**

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**4. Run the application**

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.


---

## Usage Guide

### 1. Launching the Interactive Web Application

First, activate your virtual environment.

**macOS / Linux**

```bash
source .venv/bin/activate
```
**Windows**

```bash
.venv\Scripts\activate
```

Install dependencies if you haven't already:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

The application includes:

* **Patient Risk Assessment:** Interactive patient inputs and risk predictions.
* **Batch Cohort Screening:** CSV-based batch predictions and reports.
* **Model Benchmarks:** Model evaluation metrics and visualizations.
* **Clinical Guide & EDA:** Exploratory data analysis and feature information.

### 2. Command-Line Inference

With the virtual environment activated, run:

```bash
python -m src.predict
```

### 3. Batch Cohort Screening

Run the following Python code from the project root:

```python
import pandas as pd
from src.predict import HeartDiseasePredictor

predictor = HeartDiseasePredictor()
df = pd.read_csv("data/sample_test_batch.csv")
results_df = predictor.predict_batch(df)
results_df.to_csv("screened_cohort.csv", index=False)
```

### 4. Re-training Models

With the virtual environment activated, run:

```bash
python -m src.train
python -m src.evaluate
```

**Note:** Model training and evaluation may take time depending on your hardware and dataset.


## 🧪 Automated Testing

Run the full automated test suite verifying data hygiene, ColumnTransformer functionality, prediction outputs, and risk tier categorization:
```bash
.venv/bin/python -m tests.test_pipeline
```
Output:
```
All unit tests passed successfully!
```

---

##  Demographic Fairness & Slice Analysis

Model evaluation across subpopulations on the held-out test cohort:

| Demographic Slice | Cohort Count | Test Accuracy | Positive Class Rate |
| :--- | :---: | :---: | :---: |
| **Female Cohort (`sex = 0`)** | 17 | **94.12%** | 23.53% |
| **Male Cohort (`sex = 1`)** | 44 | **72.73%** | 54.55% |
| **Younger Cohort (`age < 55`)** | 29 | **89.66%** | 27.59% |
| **Older Cohort (`age >= 55`)** | 32 | **68.75%** | 62.50% |

The higher false positive rate observed in older cohorts reflects the higher baseline prevalence of non-specific ST-T changes and age-related vascular stiffness.

---

##  Clinical Decision Support & Disclaimer

This software is an AI research prototype and is **not an FDA-cleared diagnostic medical device**. Outputs are intended for educational and decision-support exploration by healthcare practitioners. All clinical decisions must be confirmed by formal cardiological consultation, laboratory diagnostics, and physician review.
