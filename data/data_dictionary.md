# UCI Cleveland Heart Disease Dataset - Clinical Data Dictionary

## Overview
This dataset contains clinical and diagnostic measurements for patients evaluated for coronary artery disease (CAD). The goal is to predict whether heart disease is present (`target = 1`) or absent (`target = 0`).

---

## Features & Clinical Definitions

| Feature | Type | Range / Categories | Description | Clinical Context |
| :--- | :--- | :--- | :--- | :--- |
| **`age`** | Numerical | 29 – 77 years | Age of the patient in years | Cardiovascular risk increases with age due to vascular stiffening and plaque accumulation. |
| **`sex`** | Categorical (Binary) | 0 = Female<br>1 = Male | Biological sex of the patient | Men historically show higher early-onset CAD incidence, while post-menopausal women face equalized risk. |
| **`cp`** | Categorical | 0 = Typical angina<br>1 = Atypical angina<br>2 = Non-anginal pain<br>3 = Asymptomatic | Chest pain type reported by patient | **Typical angina**: retrosternal chest discomfort precipitated by exertion. **Atypical angina**: discomfort with atypical quality or duration. **Non-anginal**: musculoskeletal or gastrointestinal pain. **Asymptomatic**: silent ischemia. |
| **`trestbps`** | Numerical | 94 – 200 mm Hg | Resting blood pressure on hospital admission | Normal is < 120 mm Hg. Chronic hypertension causes mechanical damage to coronary endothelium. |
| **`chol`** | Numerical | 126 – 564 mg/dl | Serum cholesterol concentration | Desirable level is < 200 mg/dl; elevated levels promote atherosclerosis and arterial blockage. |
| **`fbs`** | Categorical (Binary) | 0 = $\le 120$ mg/dl<br>1 = $> 120$ mg/dl | Fasting blood sugar elevated test | Indicator of pre-diabetes or diabetes mellitus, a major independent risk factor for CAD. |
| **`restecg`** | Categorical | 0 = Normal<br>1 = ST-T wave abnormality<br>2 = Left ventricular hypertrophy (LVH) | Resting electrocardiogram findings | **0**: Normal sinus ECG.<br>**1**: T-wave inversion or ST elevation/depression > 0.05 mV indicating ischemia.<br>**2**: Definite LVH by Estes criteria indicating long-standing pressure overload. |
| **`thalach`** | Numerical | 71 – 202 bpm | Maximum heart rate achieved during exercise stress test | Submaximal or lower exercise capacity often indicates chronotropic incompetence or severe coronary insufficiency. |
| **`exang`** | Categorical (Binary) | 0 = No<br>1 = Yes | Exercise-induced angina | Angina provoked during physical exertion indicates inadequate myocardial oxygen supply under workload. |
| **`oldpeak`** | Numerical | 0.0 – 6.2 mm | ST depression induced by exercise relative to rest | Key electrocardiographic indicator during treadmill stress test; deeper ST depression correlates with multi-vessel disease. |
| **`slope`** | Categorical | 0 = Upsloping<br>1 = Flat<br>2 = Downsloping | Slope of peak exercise ST segment | **Downsloping** (2) and **Flat** (1) ST changes are strongly indicative of ischemia; Upsloping (0) can be normal or borderline. |
| **`ca`** | Numerical / Ordinal | 0 – 3 vessels | Number of major coronary vessels colored by fluoroscopy | Counts coronary arteries (LAD, LCx, RCA) with >50% luminal diameter narrowing. Values 0-3; 4 denotes missing/unresolved. |
| **`thal`** | Categorical | 1 = Normal<br>2 = Fixed defect<br>3 = Reversible defect | Thallium-201 myocardial perfusion scintigraphy | Evaluates blood flow: **Normal** (adequate perfusion), **Fixed defect** (prior myocardial infarction/scar), **Reversible defect** (inducible ischemia). |
| **`target`** | Target (Binary) | 0 = No Disease ($\le 50\%$ diameter stenosis)<br>1 = Disease Present ($> 50\%$ diameter stenosis) | Diagnosis of heart disease | Primary prediction label. |
