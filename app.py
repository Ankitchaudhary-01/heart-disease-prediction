"""Interactive Clinical Decision Support Web Application for Heart Disease Prediction."""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from src.config import (
    BEST_MODEL_PATH,
    BASELINE_MODEL_PATH,
    METRICS_PATH,
    SAMPLE_BATCH_PATH,
    RAW_DATA_PATH,
    FEATURE_LABELS,
    VALUE_MAPS,
    ALL_FEATURES,
    REPORTS_DIR,
)
from src.predict import HeartDiseasePredictor
from src.data_loader import load_raw_data, clean_and_validate_data
from src.database import (
    save_patient_record,
    save_batch_records,
    get_all_records,
    get_database_stats,
    delete_record,
)

# Page configuration
st.set_page_config(
    page_title="CardioPredict AI - Heart Disease Risk Engine",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clinical dashboard styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-banner {
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .risk-low {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
    }
    .risk-high {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
    }
    .flag-badge {
        display: inline-block;
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px 4px 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor(model_choice: str = "Best Model (Random Forest)"):
    """Load and cache prediction engine."""
    model_path = BEST_MODEL_PATH if "Best" in model_choice else BASELINE_MODEL_PATH
    return HeartDiseasePredictor(model_path=model_path)


@st.cache_data
def get_metrics_summary():
    """Load saved model performance metrics."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def get_dataset():
    """Load cleaned dataset for exploration."""
    raw = load_raw_data()
    return clean_and_validate_data(raw)


def main():
    st.markdown('<div class="main-header">❤️ CardioPredict AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Evidence-based Clinical Decision Support System for Coronary Heart Disease Risk Assessment</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("⚙️ System Controls")
    model_choice = st.sidebar.selectbox(
        "Active Prediction Engine",
        ["Best Model (Random Forest)", "Baseline Model (Logistic Regression)"],
        index=0,
        help="Choose between the cross-validation optimized ensemble model or the interpretable linear baseline.",
    )

    predictor = get_predictor(model_choice)
    metrics_summary = get_metrics_summary()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Patient Presets")
    preset = st.sidebar.radio(
        "Load Sample Case:",
        ["None (Custom Input)", "Case A: High Risk Profile", "Case B: Low Risk Profile", "Case C: Borderline Profile"],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Clinical Context**: Evaluates 13 diagnostic parameters using machine learning validated against the UCI Cleveland Cardiology benchmark cohort."
    )

    # Main Tabs
    tab_single, tab_batch, tab_database, tab_models, tab_eda = st.tabs([
        "🩺 Patient Assessment",
        "📁 Batch Cohort Analysis",
        "🗄️ Patient Database",
        "📊 Model Benchmarks",
        "📖 Clinical Explorer & EDA",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: SINGLE PATIENT ASSESSMENT
    # --------------------------------------------------------------------------
    with tab_single:
        st.subheader("Patient Clinical Data Entry")
        st.write("Enter the patient's diagnostic and treadmill exercise stress test results:")

        # Handle Presets
        default_vals = {
            "age": 55, "sex": 1, "cp": 0, "trestbps": 130, "chol": 240,
            "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
        }
        if preset == "Case A: High Risk Profile":
            default_vals = {
                "age": 67, "sex": 1, "cp": 0, "trestbps": 160, "chol": 286,
                "fbs": 1, "restecg": 2, "thalach": 115, "exang": 1,
                "oldpeak": 3.2, "slope": 1, "ca": 2, "thal": 3
            }
        elif preset == "Case B: Low Risk Profile":
            default_vals = {
                "age": 35, "sex": 0, "cp": 2, "trestbps": 115, "chol": 180,
                "fbs": 0, "restecg": 0, "thalach": 175, "exang": 0,
                "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2
            }
        elif preset == "Case C: Borderline Profile":
            default_vals = {
                "age": 58, "sex": 1, "cp": 1, "trestbps": 138, "chol": 235,
                "fbs": 0, "restecg": 1, "thalach": 145, "exang": 0,
                "oldpeak": 1.4, "slope": 1, "ca": 1, "thal": 2
            }

        with st.form("patient_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### 👤 Demographics & Vitals")
                age = st.slider("Age (years)", min_value=20, max_value=85, value=default_vals["age"])
                sex = st.selectbox(
                    "Biological Sex",
                    options=[1, 0],
                    format_func=lambda x: "Male" if x == 1 else "Female",
                    index=0 if default_vals["sex"] == 1 else 1,
                )
                trestbps = st.number_input(
                    "Resting Blood Pressure (mm Hg)",
                    min_value=80,
                    max_value=220,
                    value=default_vals["trestbps"],
                    help="Normal is < 120 mm Hg. Stage 1 hypertension starts at 130 mm Hg.",
                )
                chol = st.number_input(
                    "Serum Cholesterol (mg/dl)",
                    min_value=100,
                    max_value=600,
                    value=default_vals["chol"],
                    help="Desirable is < 200 mg/dl; > 240 mg/dl is high risk.",
                )

            with col2:
                st.markdown("##### 🩺 Symptoms & Resting Tests")
                cp = st.selectbox(
                    "Chest Pain Type",
                    options=[0, 1, 2, 3],
                    format_func=lambda x: VALUE_MAPS["cp"][x],
                    index=default_vals["cp"],
                    help="Typical angina represents substernal discomfort triggered by exertion.",
                )
                fbs = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dl",
                    options=[0, 1],
                    format_func=lambda x: VALUE_MAPS["fbs"][x],
                    index=default_vals["fbs"],
                )
                restecg = st.selectbox(
                    "Resting ECG",
                    options=[0, 1, 2],
                    format_func=lambda x: VALUE_MAPS["restecg"][x],
                    index=default_vals["restecg"],
                )
                ca = st.selectbox(
                    "Major Vessels Colored by Fluoroscopy (0-3)",
                    options=[0, 1, 2, 3],
                    index=default_vals["ca"],
                    help="Number of coronary arteries with >50% stenosis detected.",
                )

            with col3:
                st.markdown("##### 🏃 Exercise Stress Test Results")
                thalach = st.slider(
                    "Max Heart Rate Achieved (bpm)",
                    min_value=60,
                    max_value=220,
                    value=default_vals["thalach"],
                )
                exang = st.selectbox(
                    "Exercise Induced Angina",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    index=default_vals["exang"],
                )
                oldpeak = st.slider(
                    "ST Depression (Oldpeak in mm)",
                    min_value=0.0,
                    max_value=6.5,
                    value=float(default_vals["oldpeak"]),
                    step=0.1,
                    help="ST depression induced by exercise relative to rest. Values > 1.0 mm signal ischemia.",
                )
                slope = st.selectbox(
                    "Slope of Peak Exercise ST",
                    options=[0, 1, 2],
                    format_func=lambda x: VALUE_MAPS["slope"][x],
                    index=default_vals["slope"],
                )
                thal = st.selectbox(
                    "Thallium Scintigraphy",
                    options=[1, 2, 3],
                    format_func=lambda x: VALUE_MAPS["thal"][x],
                    index=[1, 2, 3].index(default_vals["thal"]),
                    help="Assesses myocardial blood perfusion under exertion.",
                )

            submit_btn = st.form_submit_button("⚡ Compute Heart Disease Risk", use_container_width=True)

        patient_input = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }

        # Prediction evaluation
        result = predictor.predict_single(patient_input)

        # Risk Banner Styling
        tier_class = "risk-low" if result["risk_tier"] == "Low Risk" else (
            "risk-moderate" if result["risk_tier"] == "Moderate Risk" else "risk-high"
        )

        st.markdown(
            f"""
            <div class="risk-banner {tier_class}">
                <div style="font-size: 1.6rem; font-weight: 700;">
                    { '🚨' if result['prediction'] == 1 else '✅' } {result['diagnosis']} — {result['risk_tier'].upper()}
                </div>
                <div style="font-size: 1.1rem; margin-top: 6px;">
                    Estimated CAD Probability: <b>{result['probability_percentage']}</b> (Model Confidence: {result['confidence'] * 100:.1f}%)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Detailed Analysis Columns
        r_col1, r_col2 = st.columns([1, 1])

        with r_col1:
            st.markdown("#### 🔍 Clinical Risk Flags Detected")
            flags = result["clinical_flags"]
            if flags:
                for flag in flags:
                    st.markdown(
                        f"""
                        <div style="background-color: #FEF2F2; border-left: 4px solid #EF4444; padding: 8px 12px; margin-bottom: 8px; border-radius: 4px;">
                            <span style="font-weight: 700; color: #991B1B;">{flag['factor']}</span>: 
                            <span style="color: #4B5563;">{flag['detail']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ No adverse cardiovascular flags detected in the submitted parameters.")

            st.markdown("#### 💡 Clinical Recommendation")
            st.info(result["recommendation"])

        with r_col2:
            st.markdown("#### 📈 Key Biometric Comparison")
            # Comparison chart of patient parameters vs healthy baselines
            metrics_chart_data = pd.DataFrame({
                "Parameter": ["Systolic BP (mm Hg)", "Total Chol (mg/dl)", "Max HR (bpm)", "ST Depr x10 (mm)"],
                "Patient Value": [trestbps, chol, thalach, oldpeak * 10],
                "Standard Normal": [120, 200, 160, 0],
            })
            melted = metrics_chart_data.melt("Parameter", var_name="Group", value_name="Value")
            chart = alt.Chart(melted).mark_bar().encode(
                x=alt.X("Parameter:N", axis=alt.Axis(labelAngle=-15)),
                y=alt.Y("Value:Q"),
                color=alt.Color("Group:N", scale=alt.Scale(range=["#2563EB", "#94A3B8"])),
                xOffset="Group:N",
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)

        # Database persistence
        st.markdown("---")
        st.markdown("#### 💾 Save Patient Evaluation to Database")
        db_col1, db_col2, db_col3 = st.columns([1, 1, 1])
        with db_col1:
            save_pid = st.text_input("Patient Medical ID", value=f"PT-{age}{sex}-{trestbps}", key="save_pid")
        with db_col2:
            save_pname = st.text_input("Patient Name / Initials", value="John Doe" if sex == 1 else "Jane Doe", key="save_pname")
        with db_col3:
            save_notes = st.text_input("Clinical Notes (Optional)", value="Standard outpatient cardiology triage", key="save_notes")

        if st.button("💾 Commit Patient Record to Database", use_container_width=True, key="btn_save_single_db"):
            rec_id = save_patient_record(patient_input, result, patient_id=save_pid, patient_name=save_pname, notes=save_notes)
            st.success(f"✅ Record #{rec_id} for Patient {save_pid} successfully persisted to SQLite database!")

    # --------------------------------------------------------------------------
    # TAB 2: BATCH COHORT ANALYSIS
    # --------------------------------------------------------------------------
    with tab_batch:
        st.subheader("Batch Patient Cohort Risk Screening")
        st.write("Upload a CSV file of multiple patients to execute automated population-level risk screening:")

        sample_btn = st.button("📥 Load Sample Testing Cohort (20 Patients)")

        uploaded_file = st.file_uploader("Upload Cohort CSV file", type=["csv"])

        df_to_process = None
        if sample_btn:
            if SAMPLE_BATCH_PATH.exists():
                df_to_process = pd.read_csv(SAMPLE_BATCH_PATH)
                st.success("Loaded pre-packaged test cohort with 20 patient profiles.")
        elif uploaded_file is not None:
            df_to_process = pd.read_csv(uploaded_file)

        if df_to_process is not None:
            st.markdown("##### Cohort Input Preview:")
            st.dataframe(df_to_process.head(5), use_container_width=True)

            with st.spinner("Processing batch cohort inference..."):
                batch_results = predictor.predict_batch(df_to_process)

            # High-level stats
            total_pts = len(batch_results)
            high_risk_pts = (batch_results["risk_tier"] == "High Risk").sum()
            mod_risk_pts = (batch_results["risk_tier"] == "Moderate Risk").sum()
            low_risk_pts = (batch_results["risk_tier"] == "Low Risk").sum()

            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Total Cohort Size", f"{total_pts} Patients")
            b_col2.metric("High Risk", f"{high_risk_pts} ({high_risk_pts/total_pts*100:.1f}%)", delta_color="inverse")
            b_col3.metric("Moderate Risk", f"{mod_risk_pts} ({mod_risk_pts/total_pts*100:.1f}%)")
            b_col4.metric("Low Risk", f"{low_risk_pts} ({low_risk_pts/total_pts*100:.1f}%)")

            # Risk Tier Distribution Bar Chart
            tier_df = pd.DataFrame(batch_results["risk_tier"].value_counts()).reset_index()
            tier_df.columns = ["Risk Tier", "Patient Count"]
            tier_chart = alt.Chart(tier_df).mark_bar(cornerRadius=6).encode(
                x=alt.X("Risk Tier:N", sort=["Low Risk", "Moderate Risk", "High Risk"]),
                y=alt.Y("Patient Count:Q"),
                color=alt.Color("Risk Tier:N", scale=alt.Scale(
                    domain=["Low Risk", "Moderate Risk", "High Risk"],
                    range=["#10B981", "#F59E0B", "#EF4444"]
                )),
            ).properties(height=280)
            st.altair_chart(tier_chart, use_container_width=True)

            st.markdown("##### Enriched Cohort Predictions Table:")
            st.dataframe(batch_results, use_container_width=True)

            col_dl, col_db = st.columns([1, 1])
            with col_dl:
                st.download_button(
                    label="⬇️ Download Enriched Predictions (CSV)",
                    data=csv_data,
                    file_name="cardiopredict_batch_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_db:
                if st.button("💾 Ingest Cohort into Database", use_container_width=True, key="btn_save_batch_db"):
                    ingest_cnt = save_batch_records(batch_results)
                    st.success(f"✅ Ingested {ingest_cnt} patient records into database!")

    # --------------------------------------------------------------------------
    # TAB 3: PATIENT DATABASE & AUDIT LOGS
    # --------------------------------------------------------------------------
    with tab_database:
        st.subheader("🗄️ Patient Clinical Database & Audit Records")
        st.write("Persistent SQLite database storing evaluated patient records, diagnostic probabilities, and clinical flags:")

        stats = get_database_stats()
        d_s1, d_s2, d_s3, d_s4 = st.columns(4)
        d_s1.metric("Total Stored Records", f"{stats['total_records']} Patients")
        d_s2.metric("High Risk Cohort", f"{stats['high_risk_count']}", delta_color="inverse")
        d_s3.metric("Moderate Risk Cohort", f"{stats['moderate_risk_count']}")
        d_s4.metric("Avg Cohort Probability", f"{stats['avg_probability'] * 100:.1f}%")

        db_records = get_all_records(limit=250)

        if len(db_records) > 0:
            # Filter bar
            flt_col1, flt_col2 = st.columns([1, 2])
            with flt_col1:
                tier_filter = st.selectbox("Filter Risk Tier", ["All", "High Risk", "Moderate Risk", "Low Risk"], key="db_tier_flt")
            with flt_col2:
                search_q = st.text_input("Search Patient ID or Name", placeholder="Type patient ID or name...", key="db_search_q")

            view_df = db_records.copy()
            if tier_filter != "All":
                view_df = view_df[view_df["risk_tier"] == tier_filter]
            if search_q:
                q = search_q.lower()
                view_df = view_df[
                    view_df["patient_id"].astype(str).str.lower().str.contains(q) |
                    view_df["patient_name"].astype(str).str.lower().str.contains(q)
                ]

            st.markdown(f"##### Showing {len(view_df)} of {len(db_records)} records:")
            # Display formatted table
            display_cols = [
                "id", "patient_id", "patient_name", "age", "sex", "diagnosis",
                "probability", "risk_tier", "clinician_notes", "created_at"
            ]
            valid_disp_cols = [c for c in display_cols if c in view_df.columns]
            st.dataframe(view_df[valid_disp_cols], use_container_width=True)

            act_col1, act_col2 = st.columns([1, 1])
            with act_col1:
                db_csv = view_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Export Filtered Database Records (CSV)",
                    data=db_csv,
                    file_name="cardiopredict_patient_database.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with act_col2:
                del_id = st.number_input("Delete Record by Database ID", min_value=1, step=1, value=1, key="del_db_id")
                if st.button("🗑️ Delete Record", use_container_width=True, key="btn_del_db_rec"):
                    if delete_record(int(del_id)):
                        st.warning(f"Record #{del_id} deleted from database. Please switch tabs to refresh.")
                    else:
                        st.error(f"Record #{del_id} not found in database.")
        else:
            st.info("ℹ️ No patient records currently stored in the database. Evaluate a patient in Tab 1 or Tab 2 and commit to database.")

    # --------------------------------------------------------------------------
    # TAB 4: MODEL BENCHMARKS & EVALUATION
    # --------------------------------------------------------------------------
    with tab_models:
        st.subheader("Statistical Validation & Model Benchmarking")
        st.write("Rigorous model comparison conforming to scientific machine learning standards:")

        if metrics_summary:
            best_name = metrics_summary.get("best_model_name", "Random Forest")
            st.success(f"🏆 Winning Architecture: **{best_name}** (Evaluated across 5-Fold Stratified Cross-Validation)")

            # Comparison Table
            models_dict = metrics_summary["models"]
            rows = []
            for name, data in models_dict.items():
                cv = data["cv_metrics"]
                test = data["test_metrics"]
                rows.append({
                    "Model": name,
                    "CV ROC-AUC (Mean ± Std)": f"{cv.get('roc_auc', {}).get('mean', 0):.4f} ± {cv.get('roc_auc', {}).get('std', 0):.4f}",
                    "Test Accuracy": f"{test['accuracy']*100:.1f}%",
                    "Test Precision": f"{test['precision']*100:.1f}%",
                    "Test Recall (Sensitivity)": f"{test['recall']*100:.1f}%",
                    "Test Specificity": f"{test['specificity']*100:.1f}%",
                    "Test F1-Score": f"{test['f1_score']:.4f}",
                    "Test ROC-AUC": f"{test['roc_auc']:.4f}",
                    "Latency (ms/sample)": f"{test['per_sample_latency_ms']:.2f} ms",
                })
            comparison_df = pd.DataFrame(rows)
            st.dataframe(comparison_df, use_container_width=True)

            st.markdown("---")
            v_col1, v_col2 = st.columns(2)

            with v_col1:
                st.markdown("##### ROC Curve (Held-out Test Cohort)")
                roc_file = REPORTS_DIR / "roc_curve.png"
                if roc_file.exists():
                    st.image(str(roc_file), caption="Receiver Operating Characteristic Curve", use_container_width=True)

            with v_col2:
                st.markdown("##### Confusion Matrix (Test Cohort)")
                cm_file = REPORTS_DIR / "confusion_matrix.png"
                if cm_file.exists():
                    st.image(str(cm_file), caption="Held-out Test Set Confusion Matrix", use_container_width=True)

            fi_file = REPORTS_DIR / "feature_importance.png"
            if fi_file.exists():
                st.markdown("##### Top Predictive Clinical Features (Feature Importance)")
                st.image(str(fi_file), caption="Random Forest Gini Feature Importances", use_container_width=True)

            # Demographic Slices
            slice_file = REPORTS_DIR / "demographic_slices.json"
            if slice_file.exists():
                with open(slice_file) as f:
                    slices = json.load(f)
                st.markdown("##### Subpopulation & Fairness Slice Analysis")
                s_cols = st.columns(len(slices))
                for idx, (grp, s_data) in enumerate(slices.items()):
                    s_cols[idx].metric(grp, f"{s_data['accuracy']*100:.1f}% Acc", f"{s_data['count']} cases")

    # --------------------------------------------------------------------------
    # TAB 4: CLINICAL EXPLORER & EDA
    # --------------------------------------------------------------------------
    with tab_eda:
        st.subheader("Exploratory Data Analysis & Clinical Guide")

        raw_df = get_dataset()

        e_col1, e_col2, e_col3 = st.columns(3)
        e_col1.metric("Cohort Patients", len(raw_df))
        e_col2.metric("Coronary Disease Cases", (raw_df["target"] == 1).sum())
        e_col3.metric("Healthy / Control Cases", (raw_df["target"] == 0).sum())

        st.markdown("##### Dataset Inspection")
        st.dataframe(raw_df.head(10), use_container_width=True)

        st.markdown("##### Feature Correlations with Heart Disease Target")
        corrs = raw_df.corr()["target"].drop("target").sort_values(ascending=False).reset_index()
        corrs.columns = ["Feature", "Correlation with Heart Disease"]

        corr_chart = alt.Chart(corrs).mark_bar().encode(
            x=alt.X("Correlation with Heart Disease:Q"),
            y=alt.Y("Feature:N", sort="-x"),
            color=alt.condition(
                alt.datum["Correlation with Heart Disease"] > 0,
                alt.value("#EF4444"),  # Positive correlation with disease
                alt.value("#10B981"),  # Negative correlation (protective)
            ),
        ).properties(height=350)
        st.altair_chart(corr_chart, use_container_width=True)

        st.markdown(
            """
            > [!NOTE]
            > **Clinical Insight**:
            > - **Positive Risk Drivers**: ST depression (`oldpeak`), number of stenosed vessels (`ca`), exercise-induced angina (`exang`), and typical angina (`cp`) demonstrate strongest association with coronary obstruction.
            > - **Protective Indicators**: Higher maximum heart rate during stress testing (`thalach`) reflects superior cardiac functional reserve.
            """
        )

        st.markdown("---")
        st.caption("⚕️ **Medical Disclaimer**: CardioPredict AI is intended solely for clinical research and educational decision support. It should not be used as the sole basis for clinical diagnosis or prescription.")


if __name__ == "__main__":
    main()
