import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# 2. BROWN EDITORIAL DESIGN
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&family=Pinyon+Script&family=Cormorant+Garamond:wght@400;500;600;700&display=swap');

/* Global App Styling */
.stApp {
    background: #FDFBF7;
    color: #4A2E2B; /* Dark Brown */
}

/* Force all general text elements to brown */
p, div, span, label, li, h1, h2, h3, h4, h5, h6 {
    color: #4A2E2B !important;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

#luxury-header {
    background: linear-gradient(120deg, #FFFDF9, #F5EBE6);
    border: 1px solid #D7C4B7;
    border-radius: 20px;
    padding: 25px 20px;
    text-align: center;
    margin-bottom: 25px;
}

.top-label {
    font-family: 'Cormorant Garamond', serif;
    color: #8C5A3C !important; /* Golden Warm Brown */
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 2px;
}

.main-title {
    font-family: 'UnifrakturMaguntia', serif;
    color: #3E2723 !important; /* Deep Espresso Brown */
    font-size: 42px;
    line-height: 1.3;
    margin: 12px 0;
}

.subtitle {
    font-family: 'Pinyon Script', cursive;
    color: #6D4C41 !important; /* Soft Warm Brown */
    font-size: 25px;
}

.panel {
    background: #FFFEFC;
    border: 1px solid #E6D8CE;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 15px;
}

.section-title {
    font-family: 'Pinyon Script', cursive;
    color: #5C3A21 !important; /* Rich Brown */
    font-size: 32px;
}

.result-number {
    font-family: 'Cormorant Garamond', serif;
    color: #3E2723 !important;
    font-size: 54px;
    font-weight: 700;
}

.result-category {
    background: #F2E3D5;
    color: #4A2E2B !important;
    border: 1px solid #CDB4A0;
    padding: 9px 14px;
    border-radius: 10px;
    font-weight: 600;
    display: inline-block;
}

/* Primary Button Styling */
div.stButton > button {
    background: #5C3A21;
    color: #FFFDF9 !important;
    border: 1px solid #8C5A3C;
    border-radius: 12px;
    min-height: 48px;
    font-family: 'Cormorant Garamond', serif;
    font-size: 19px;
    font-weight: 600;
}

div.stButton > button:hover {
    background: #4A2E2B;
    color: #FFFDF9 !important;
    border: 1px solid #8C5A3C;
}

/* Metrics and Streamlit Cards */
[data-testid="stMetric"] {
    background: #FFFEFC;
    border: 1px solid #E6D8CE;
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #4A2E2B !important;
}

label, p, li {
    font-family: 'Cormorant Garamond', serif;
    font-size: 17px;
    color: #4A2E2B !important;
}

.footer {
    text-align: center;
    color: #7D6356 !important; /* Muted Warm Brown */
    font-family: 'Cormorant Garamond', serif;
    font-size: 14px;
    padding: 20px 5px;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. CREATE PRACTICE DATA
# --------------------------------------------------

@st.cache_resource
def train_model():
    np.random.seed(42)

    n = 300

    attendance = np.random.randint(40, 101, n)
    study_hours = np.random.randint(0, 9, n)
    assignments = np.random.randint(30, 101, n)
    previous_marks = np.random.randint(25, 96, n)
    noise = np.random.normal(0, 5, n)

    final_marks = (
        0.20 * attendance
        + 1.5 * study_hours
        + 0.20 * assignments
        + 0.45 * previous_marks
        + noise
    )

    final_marks = np.clip(final_marks, 0, 100)

    df = pd.DataFrame({
        "Attendance": attendance,
        "Study_Hours": study_hours,
        "Assignment_Completion": assignments,
        "Previous_Marks": previous_marks,
        "Final_Marks": final_marks.round(1)
    })

    X = df[
        [
            "Attendance",
            "Study_Hours",
            "Assignment_Completion",
            "Previous_Marks"
        ]
    ]

    y = df["Final_Marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)

    return model, mae, r2


model, mae, r2 = train_model()

# --------------------------------------------------
# 4. APP HEADER
# --------------------------------------------------

st.markdown("""
<div id="luxury-header">
    <div class="top-label">
        Academic Intelligence · Student Edition
    </div>

    <div class="main-title">
        🎓 Student Performance Predictor
    </div>

    <div class="subtitle">
        Understand your progress. Discover opportunities.
        Take the next step with confidence.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 5. INPUT PANEL
# --------------------------------------------------

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(
        '<div class="section-title">Student Information</div>',
        unsafe_allow_html=True
    )

    st.write("Adjust the values to estimate academic performance.")

    attendance = st.slider(
        "Class Attendance (%)",
        min_value=40,
        max_value=100,
        value=85,
        step=1
    )

    study_hours = st.slider(
        "Daily Study Hours",
        min_value=0,
        max_value=8,
        value=3,
        step=1
    )

    assignments = st.slider(
        "Assignment Completion (%)",
        min_value=30,
        max_value=100,
        value=80,
        step=1
    )

    previous_marks = st.slider(
        "Previous Examination Marks",
        min_value=25,
        max_value=95,
        value=70,
        step=1
    )

    predict_button = st.button(
        "✦ Predict Academic Performance",
        use_container_width=True
    )

# --------------------------------------------------
# 6. PREDICTION FUNCTION
# --------------------------------------------------

def predict_student(
    attendance,
    study_hours,
    assignments,
    previous_marks
):
    student = pd.DataFrame([{
        "Attendance": attendance,
        "Study_Hours": study_hours,
        "Assignment_Completion": assignments,
        "Previous_Marks": previous_marks
    }])

    prediction = float(model.predict(student)[0])

    prediction = max(0, min(100, prediction))

    if prediction >= 80:
        category = "High"
    elif prediction >= 60:
        category = "Moderate"
    else:
        category = "Needs Improvement"

    suggestions = []

    if attendance < 75:
        suggestions.append(
            "Attend classes regularly to strengthen your understanding."
        )

    if study_hours < 2:
        suggestions.append(
            "Set aside more time for daily revision."
        )

    if assignments < 75:
        suggestions.append(
            "Try to complete assignments on time."
        )

    if previous_marks < 50:
        suggestions.append(
            "Revise the basics and practise questions regularly."
        )

    if not suggestions:
        suggestions.append(
            "Maintain your study routine and revise regularly."
        )

    return prediction, category, suggestions


# --------------------------------------------------
# 7. RESULTS PANEL
# --------------------------------------------------

with right:
    st.markdown(
        '<div class="section-title">Your Prediction</div>',
        unsafe_allow_html=True
    )

    if predict_button:
        prediction, category, suggestions = predict_student(
            attendance,
            study_hours,
            assignments,
            previous_marks
        )

        st.markdown(
            f'<div class="result-number">{prediction:.1f}'
            f'<span style="font-size:24px; color:#4A2E2B !important;"> / 100</span></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-category">'
            f'{category} Performance</div>',
            unsafe_allow_html=True
        )

        st.write("")

        st.write("**Estimated performance indicator**")
        st.progress(int(round(prediction)))

        st.write("**Personalised study suggestions**")

        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

        st.caption(
            "This is an estimate, not a guaranteed examination result."
        )

    else:
        st.info(
            "Adjust the student information and tap "
            "'Predict Academic Performance' to see your results."
        )

# --------------------------------------------------
# 8. MODEL INFORMATION
# --------------------------------------------------

with st.expander("About the AI model"):
    st.write(
        "This app uses Linear Regression, a supervised machine-learning "
        "algorithm, to estimate final marks from four input features."
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Mean Absolute Error",
        f"{mae:.2f} marks"
    )

    col2.metric(
        "R² Score",
        f"{r2:.3f}"
    )

    st.caption(
        "These evaluation results use a held-out portion of synthetic "
        "practice data. They do not establish real-world predictive accuracy."
    )

# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    STUDENT ACADEMIC PERFORMANCE PREDICTOR<br>
    A Machine Learning Learning Project<br><br>
    Designed to encourage learning, reflection and improvement.
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational prototype only. The model was trained on artificially "
    "generated data, not real student records. Its predictions and "
    "performance categories should not be used for academic decisions."
)
