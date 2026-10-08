from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "churn_model.joblib"
META_PATH = BASE / "models" / "metadata.json"

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="📊",
    layout="wide"
)

if not MODEL_PATH.exists():
    st.error("Model not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

st.title("📊 ChurnGuard")
st.subheader("Customer Churn Prediction & Retention Dashboard")
st.write(
    "Enter customer information to estimate churn risk and receive "
    "a simple retention recommendation."
)

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines", ["No phone service", "No", "Yes"]
)

internet = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security", ["No internet service", "No", "Yes"]
)
online_backup = st.sidebar.selectbox(
    "Online Backup", ["No internet service", "No", "Yes"]
)
device_protection = st.sidebar.selectbox(
    "Device Protection", ["No internet service", "No", "Yes"]
)
tech_support = st.sidebar.selectbox(
    "Tech Support", ["No internet service", "No", "Yes"]
)
streaming_tv = st.sidebar.selectbox(
    "Streaming TV", ["No internet service", "No", "Yes"]
)
streaming_movies = st.sidebar.selectbox(
    "Streaming Movies", ["No internet service", "No", "Yes"]
)

contract = st.sidebar.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

tenure = st.sidebar.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = monthly_charges * tenure

st.sidebar.number_input(
    "Total Charges",
    value=total_charges,
    disabled=True
)



input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

if st.button("🔍 Predict Churn Risk", type="primary"):
    probability = float(model.predict_proba(input_df)[0, 1])

    if probability >= 0.70:
        risk = "HIGH"
        recommendation = (
            "Prioritize this customer for retention. Consider a personalized "
            "offer, support call, plan review, or service-quality follow-up."
        )
    elif probability >= 0.40:
        risk = "MEDIUM"
        recommendation = (
            "Monitor this customer and consider proactive engagement, "
            "especially around contract and service preferences."
        )
    else:
        risk = "LOW"
        recommendation = (
            "No immediate retention action is required. Continue normal "
            "customer engagement."
        )

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{probability:.1%}")

    with col2:
        st.metric("Risk Level", risk)

    with col3:
        st.metric("Tenure", f"{tenure} months")

    st.progress(probability)

    if risk == "HIGH":
        st.error(f"🚨 HIGH CHURN RISK\n\n{recommendation}")
    elif risk == "MEDIUM":
        st.warning(f"⚠️ MEDIUM CHURN RISK\n\n{recommendation}")
    else:
        st.success(f"✅ LOW CHURN RISK\n\n{recommendation}")

    st.info(
        "Important: this model estimates risk from historical sample data. "
        "It does not prove that a customer will churn or that a particular "
        "retention action will prevent churn."
    )

st.divider()

st.subheader("Model Information")
c1, c2 = st.columns(2)
with c1:
    st.write(f"**Selected model:** {metadata['best_model']}")
with c2:
    st.write(f"**Test ROC-AUC:** {metadata['roc_auc']:.3f}")

st.caption(
    "ChurnGuard is a portfolio/educational ML application built with "
    "Python, Pandas, Scikit-learn and Streamlit."
)
