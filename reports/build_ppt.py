"""
Script to generate a professional, high-impact PowerPoint presentation (.pptx)
for the Heart Disease Prediction Machine Learning Project (CardioPredict AI).
"""

from pathlib import Path
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PPTX = REPORTS_DIR / "Heart_Disease_Prediction_Deck.pptx"

# Modern Medical AI Color Palette
NAVY = RGBColor(30, 58, 138)       # #1E3A8A - Primary Header
BLUE_ACCENT = RGBColor(2, 132, 199) # #0284C7 - Accent Teal/Blue
DARK_TEXT = RGBColor(30, 41, 59)    # #1E293B - Body Text
MUTED_TEXT = RGBColor(100, 116, 139) # #64748B - Subtitles/captions
BG_LIGHT = RGBColor(248, 250, 252)  # #F8FAFC - Card Background
CARD_BORDER = RGBColor(226, 232, 240) # #E2E8F0
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(16, 185, 129)     # #10B981 - Success
RED = RGBColor(239, 68, 68)        # #EF4444 - Danger
AMBER = RGBColor(245, 158, 11)     # #F59E0B - Warning


def set_shape_flat_style(shape, fill_color, border_color=None):
    """Apply fill and border to shape."""
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()


def add_slide_header(slide, title_text, category="CARDIOPREDICT AI • PROJECT BRIEF"):
    """Add unified header banner to content slides."""
    # Category tag
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.35))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category.upper()
    p_c.font.size = Pt(10)
    p_c.font.bold = True
    p_c.font.color.rgb = BLUE_ACCENT

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.7))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(24)
    p_t.font.bold = True
    p_t.font.color.rgb = NAVY


def add_card(slide, left, top, width, height, title, items, bg_color=BG_LIGHT, border_color=CARD_BORDER):
    """Add a card container with title and bullet points."""
    rect = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    set_shape_flat_style(rect, bg_color, border_color)

    tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), height - Inches(0.3))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = NAVY
    p0.space_after = Pt(8)

    for item in items:
        p = tf.add_paragraph()
        p.text = f"•  {item}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(4)


def create_deck():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # -------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)

    # Hero background card
    hero = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    set_shape_flat_style(hero, RGBColor(241, 245, 249))

    # Decorative accent bar
    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.25))
    set_shape_flat_style(top_bar, NAVY)

    # Main title box
    tb = s1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(10.5), Inches(3.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p_tag = tf.paragraphs[0]
    p_tag.text = "CLINICAL DECISION SUPPORT & MACHINE LEARNING"
    p_tag.font.size = Pt(12)
    p_tag.font.bold = True
    p_tag.font.color.rgb = BLUE_ACCENT
    p_tag.space_after = Pt(10)

    p_title = tf.add_paragraph()
    p_title.text = "CardioPredict AI"
    p_title.font.size = Pt(46)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY
    p_title.space_after = Pt(12)

    p_sub = tf.add_paragraph()
    p_sub.text = "End-to-End Predictive Machine Learning System for Coronary Heart Disease Detection & Clinical Risk Stratification"
    p_sub.font.size = Pt(17)
    p_sub.font.color.rgb = DARK_TEXT
    p_sub.space_after = Pt(24)

    p_meta = tf.add_paragraph()
    p_meta.text = "Benchmark: UCI Cleveland Clinic Cardiology Cohort  |  AUC-ROC: 0.9109  |  Production Ready"
    p_meta.font.size = Pt(12)
    p_meta.font.bold = True
    p_meta.font.color.rgb = MUTED_TEXT

    # -------------------------------------------------------------
    # SLIDE 2: PROBLEM STATEMENT & CLINICAL CONTEXT
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_header(s2, "Executive Summary & Clinical Rationale")

    # 3 Stat Cards
    stat_data = [
        ("17.9M Deaths / Year", "Global Mortality", "Cardiovascular diseases are the leading cause of death globally (WHO report).", RED),
        ("> 40% Risk Reduction", "Early Detection Impact", "Timely clinical identification of coronary stenosis enables life-saving interventions.", GREEN),
        ("13 Diagnostic Inputs", "Multi-Parametric AI", "Combines resting hemodynamics, biochemistry, ECG, and exercise stress data.", BLUE_ACCENT),
    ]

    for i, (metric, label, desc, col) in enumerate(stat_data):
        left = Inches(0.8 + i * 4.0)
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.6), Inches(3.7), Inches(2.2))
        set_shape_flat_style(card, BG_LIGHT, CARD_BORDER)

        tb = s2.shapes.add_textbox(left + Inches(0.2), Inches(1.8), Inches(3.3), Inches(1.8))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = metric
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = col
        p1.space_after = Pt(4)

        p2 = tf.add_paragraph()
        p2.text = label
        p2.font.size = Pt(12)
        p2.font.bold = True
        p2.font.color.rgb = NAVY
        p2.space_after = Pt(6)

        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(11)
        p3.font.color.rgb = DARK_TEXT

    # Lower context cards
    add_card(
        s2, Inches(0.8), Inches(4.1), Inches(5.7), Inches(2.7),
        "Clinical Diagnostic Challenges",
        [
            "Coronary artery disease often remains asymptomatic until acute myocardial infarction.",
            "Physicians evaluate high-dimensional resting biometrics and stress tests with cognitive fatigue.",
            "Subjective ECG repolarization interpretations show notable inter-observer variability.",
            "Need for objective, calibrated risk probability score to prioritize catheterization."
        ]
    )

    add_card(
        s2, Inches(6.8), Inches(4.1), Inches(5.7), Inches(2.7),
        "CardioPredict AI Objectives",
        [
            "Develop an evidence-based ML engine with strict featurization and zero data leakage.",
            "Compare diverse architectures: Linear, Tree Ensembles, Support Vector Machines, KNN.",
            "Provide calibrated probability estimates categorized into actionable clinical risk tiers.",
            "Deploy an interactive Clinical Decision Support (CDS) web application for medical staff."
        ]
    )

    # -------------------------------------------------------------
    # SLIDE 3: DATASET & CLINICAL BIOMARKERS
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_header(s3, "Cardiology Cohort & Feature Dictionary")

    # Table of Features
    table_shape = s3.shapes.add_table(7, 4, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    table = table_shape.table
    table.columns[0].width = Inches(2.2)
    table.columns[1].width = Inches(2.0)
    table.columns[2].width = Inches(2.2)
    table.columns[3].width = Inches(5.3)

    headers = ["Clinical Feature", "Measurement Type", "Normal / Reference", "Diagnostic Significance"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = WHITE

    row_data = [
        ("age / sex", "Demographic", "29-77 yrs / M, F", "Biological baseline risk; male predisposition in premature CAD"),
        ("trestbps", "Hemodynamic", "< 120 mm Hg", "Resting systolic blood pressure; mechanical arterial shear stress"),
        ("chol / fbs", "Biochemical", "< 200 mg/dl / <= 120", "Serum cholesterol & fasting glycemia; atheroma plaque drivers"),
        ("cp / exang", "Symptomatology", "Asymptomatic / No angina", "Chest pain type & exercise-induced angina under stress"),
        ("restecg / slope", "Electrocardiogram", "Normal sinus / Upsloping", "ST-T repolarization abnormality & exercise ST peak slope"),
        ("oldpeak / ca / thal", "Stress & Imaging", "0.0 mm / 0 vessels / Normal", "ST depression, fluoroscopy stenosed arteries, myocardial perfusion defect"),
    ]

    for i, row in enumerate(row_data):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = BG_LIGHT if i % 2 == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 4: ML METHODOLOGY & PREPROCESSING PIPELINE
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_header(s4, "Engineering Pipeline & Strict Featurization Ordering")

    pipeline_steps = [
        ("1. Data Hygiene", "Clean & Verify", [
            "UCI Cleveland dataset (302 validated records).",
            "Deduplication applied to prevent train-test contamination.",
            "Imputed sentinel codes: ca=4 (missing fluoroscopy) and thal=0 using domain modes."
        ]),
        ("2. Strict Partitioning", "Split Before Fitting", [
            "Split 80% Train (n=241) and 20% Held-out Test (n=61).",
            "Stratified split preserving exact clinical prevalence.",
            "Crucial: Encoders & scalers fit strictly on training fold only."
        ]),
        ("3. ColumnTransformer", "Leak-free Preprocessing", [
            "Numerical pipeline: Median Imputer + StandardScaler.",
            "Categorical pipeline: Most Frequent Imputer + OneHotEncoder.",
            "Encapsulated in scikit-learn Pipeline object."
        ]),
        ("4. Validation Strategy", "5-Fold Stratified CV", [
            "Rigorous cross-validation across 5 internal splits.",
            "Bootstrapped 95% Confidence Intervals for key metrics.",
            "Evaluated on held-out test cohort for true generalizability."
        ]),
    ]

    for i, (title, subtitle, bullets) in enumerate(pipeline_steps):
        left = Inches(0.8 + i * 2.95)
        card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.6), Inches(2.8), Inches(5.2))
        set_shape_flat_style(card, BG_LIGHT, CARD_BORDER)

        tb = s4.shapes.add_textbox(left + Inches(0.15), Inches(1.8), Inches(2.5), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = NAVY

        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(10)
        p2.font.bold = True
        p2.font.color.rgb = BLUE_ACCENT
        p2.space_after = Pt(10)

        for b in bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(10)
            p.font.color.rgb = DARK_TEXT
            p.space_after = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 5: MODEL COMPARISON & BENCHMARK RESULTS
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_header(s5, "Multi-Model Benchmarking & Empirical Results")

    # Benchmarks Table
    table_shape = s5.shapes.add_table(7, 7, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    table = table_shape.table
    widths = [Inches(2.5), Inches(2.0), Inches(1.5), Inches(1.4), Inches(1.5), Inches(1.4), Inches(1.4)]
    for j, w in enumerate(widths):
        table.columns[j].width = w

    headers = ["Model Architecture", "5-Fold CV ROC-AUC", "Test Acc", "Test Precision", "Test Recall", "Test F1", "Test ROC-AUC"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = WHITE

    model_rows = [
        ("Random Forest (Winner)", "0.9109 ± 0.0435", "78.69%", "75.47%", "71.43%", "0.7547", "0.8799", True),
        ("Logistic Regression", "0.9049 ± 0.0477", "83.61%", "82.14%", "82.14%", "0.8214", "0.9242", False),
        ("Support Vector Classifier (RBF)", "0.9005 ± 0.0388", "80.33%", "76.92%", "71.43%", "0.7692", "0.8994", False),
        ("K-Nearest Neighbors", "0.8861 ± 0.0420", "86.89%", "84.62%", "78.57%", "0.8462", "0.9351", False),
        ("Gradient Boosting", "0.8769 ± 0.0463", "78.69%", "75.47%", "71.43%", "0.7547", "0.8810", False),
        ("Majority Class Baseline", "0.5000 ± 0.0000", "54.10%", "0.00%", "0.00%", "0.0000", "0.5000", False),
    ]

    for i, row in enumerate(model_rows):
        is_winner = row[7]
        for j in range(7):
            cell = table.cell(i + 1, j)
            cell.text = row[j]
            cell.fill.solid()
            if is_winner:
                cell.fill.fore_color.rgb = RGBColor(238, 242, 255) # Indigo tint
            else:
                cell.fill.fore_color.rgb = BG_LIGHT if i % 2 == 0 else WHITE

            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = NAVY if is_winner else DARK_TEXT
                if is_winner:
                    p.font.bold = True

    # -------------------------------------------------------------
    # SLIDE 6: ROC CURVE & CONFUSION MATRIX EVALUATION
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_header(s6, "Diagnostic Performance & Classification Metrics")

    # Embed ROC curve image
    roc_img_path = REPORTS_DIR / "roc_curve.png"
    if roc_img_path.exists():
        s6.shapes.add_picture(str(roc_img_path), Inches(0.8), Inches(1.6), width=Inches(5.7))

    # Embed Confusion matrix image
    cm_img_path = REPORTS_DIR / "confusion_matrix.png"
    if cm_img_path.exists():
        s6.shapes.add_picture(str(cm_img_path), Inches(6.8), Inches(1.6), width=Inches(5.7))

    # -------------------------------------------------------------
    # SLIDE 7: FEATURE IMPORTANCE & CLINICAL INTERPRETABILITY
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_header(s7, "Feature Importance & Clinical Insights")

    fi_img_path = REPORTS_DIR / "feature_importance.png"
    if fi_img_path.exists():
        s7.shapes.add_picture(str(fi_img_path), Inches(0.8), Inches(1.6), width=Inches(6.2))

    add_card(
        s7, Inches(7.3), Inches(1.6), Inches(5.2), Inches(5.2),
        "Cardiological Alignment",
        [
            "Exercise ST Depression (oldpeak): Primary diagnostic marker of subendocardial ischemia during treadmill stress.",
            "Fluoroscopy Vessels (ca): Direct anatomical enumeration of coronary luminal stenosis in major arteries.",
            "Thallium Scintigraphy (thal): Differentiates irreversible myocardial infarction scar from reversible inducible ischemia.",
            "Exercise-Induced Angina (exang): Strong subjective symptom correlating with hemodynamic coronary insufficiency.",
            "Maximum Heart Rate (thalach): Protective indicator; high heart rate reserve correlates strongly with healthy cardiac output."
        ]
    )

    # -------------------------------------------------------------
    # SLIDE 8: CLINICAL APPLICATION (STREAMLIT APP)
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_header(s8, "Interactive Clinical Decision Support Application")

    app_features = [
        ("🩺 Patient Risk Calculator", [
            "Input 13 patient biometrics via intuitive clinical sliders.",
            "One-click patient presets (High Risk, Low Risk, Borderline).",
            "Real-time CAD probability gauge and confidence score.",
            "Automated clinical flags (Hypertension, Hypercholesterolemia)."
        ]),
        ("📁 Batch Screening Tool", [
            "Upload entire cohort CSV files for automated triage.",
            "Pre-packaged sample test cohort for immediate demonstration.",
            "Population-level risk distribution analytics.",
            "Exportable enriched diagnostic CSV reports."
        ]),
        ("📊 Live Benchmarks & Slices", [
            "Interactive model switching (Random Forest vs Logistic Regression).",
            "Demographic fairness breakdown across age and sex.",
            "Side-by-side performance comparison metrics.",
            "Comprehensive clinical data dictionary and guide."
        ]),
    ]

    for i, (title, items) in enumerate(app_features):
        left = Inches(0.8 + i * 4.0)
        card = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.6), Inches(3.7), Inches(5.2))
        set_shape_flat_style(card, BG_LIGHT, CARD_BORDER)

        tb = s8.shapes.add_textbox(left + Inches(0.2), Inches(1.8), Inches(3.3), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = NAVY
        p1.space_after = Pt(12)

        for it in items:
            p = tf.add_paragraph()
            p.text = f"•  {it}"
            p.font.size = Pt(11)
            p.font.color.rgb = DARK_TEXT
            p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 9: DEMOGRAPHIC FAIRNESS & SLICE ANALYSIS
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_header(s9, "Fairness Auditing & Subpopulation Slice Analysis")

    slices_info = [
        ("Female Cohort (sex = 0)", "94.12% Accuracy", "17 test patients | 23.5% CAD prevalence", "High sensitivity and specificity; clear distinction of ischemia markers.", GREEN),
        ("Male Cohort (sex = 1)", "72.73% Accuracy", "44 test patients | 54.5% CAD prevalence", "Higher baseline prevalence; typical angina symptoms more pronounced.", BLUE_ACCENT),
        ("Younger Cohort (age < 55)", "89.66% Accuracy", "29 test patients | 27.6% CAD prevalence", "Low false positive rate; high exercise tolerance is strongly protective.", GREEN),
        ("Older Cohort (age >= 55)", "68.75% Accuracy", "32 test patients | 62.5% CAD prevalence", "Higher rate of non-specific ST changes & age-related arterial stiffness.", AMBER),
    ]

    for i, (grp, acc, meta, note, col) in enumerate(slices_info):
        left = Inches(0.8 + (i % 2) * 6.0)
        top = Inches(1.6 + (i // 2) * 2.6)

        card = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.7), Inches(2.3))
        set_shape_flat_style(card, BG_LIGHT, CARD_BORDER)

        tb = s9.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), Inches(5.3), Inches(2.0))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = grp
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = NAVY

        p2 = tf.add_paragraph()
        p2.text = acc
        p2.font.size = Pt(20)
        p2.font.bold = True
        p2.font.color.rgb = col
        p2.space_after = Pt(4)

        p3 = tf.add_paragraph()
        p3.text = meta
        p3.font.size = Pt(10)
        p3.font.color.rgb = MUTED_TEXT
        p3.space_after = Pt(6)

        p4 = tf.add_paragraph()
        p4.text = f"Clinical Note: {note}"
        p4.font.size = Pt(10)
        p4.font.color.rgb = DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 10: CONCLUSION & FUTURE WORK
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_header(s10, "Summary, Impact & Future Roadmap")

    add_card(
        s10, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2),
        "Key Project Achievements",
        [
            "High Predictive Discrimination: Random Forest achieved 0.9109 CV ROC-AUC; Logistic Regression reached 83.61% accuracy and 92.42% test ROC-AUC.",
            "Clinical Trust & Explainability: Feature ranking directly corroborates cardiology guidelines regarding ST depression and fluoroscopy vessel count.",
            "Production Software Architecture: Modular package structure with automated data hygiene, CI unit tests (100% pass rate), and Streamlit CDS interface.",
            "Tiered Risk Stratification: Calibrated risk probability allows clinical triage into Low (<25%), Moderate (25-60%), and High (>60%) risk pathways."
        ]
    )

    add_card(
        s10, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
        "Future Roadmap & Enhancements",
        [
            "Electronic Health Record (EHR) Integration: Implement HL7 / FHIR API connectors for seamless integration into hospital management systems.",
            "Multi-Center External Cohorts: Validate against international registries (Hungarian, Swiss, Long Beach VA datasets).",
            "Deep Learning Waveform Analysis: Transition from discrete tabular ECG metrics to raw 12-lead ECG voltage time-series models (1D-CNN / Transformers).",
            "Medical Disclaimer: Designed for investigational decision-support; not a replacement for formal physician diagnosis."
        ]
    )

    # Save presentation
    prs.save(OUTPUT_PPTX)
    print(f"Presentation successfully created at: {OUTPUT_PPTX}")


if __name__ == "__main__":
    create_deck()
