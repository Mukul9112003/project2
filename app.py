import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telco Customer Churn Prediction", layout="centered")
st.title("📊 Telco Customer Churn Prediction App")

@st.cache_resource
def load_model():
    return joblib.load("xgb_telco_pipeline.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("❌ Failed to load model")
    st.exception(e)
    st.stop()

expected_cols = [
    "gender", "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
    "Family", "TenureGroup"
]

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
tenure = st.sidebar.number_input("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)
family = st.sidebar.selectbox(
    "Family",
    ["0_Yes_Yes", "0_Yes_No", "0_No_Yes", "0_No_No",
     "1_Yes_Yes", "1_Yes_No", "1_No_Yes", "1_No_No"]
)

def tenure_bucket(tenure):
    if tenure <= 12:
        return "0-1 Year"
    elif tenure <= 24:
        return "1-2 Years"
    elif tenure <= 48:
        return "2-4 Years"
    else:
        return "4+ Years"

tenure_group = tenure_bucket(tenure)

customer_data = pd.DataFrame([[ 
    gender, tenure, phone_service, multiple_lines,
    internet_service, online_security, online_backup,
    device_protection, tech_support, streaming_tv,
    streaming_movies, contract, paperless_billing,
    payment_method, monthly_charges, total_charges,
    family, tenure_group
]], columns=expected_cols)

if st.sidebar.button("Predict Churn"):
    prob = model.predict_proba(customer_data)[0][1]
    label = "🔴 Likely to Churn" if prob >= 0.5 else "🟢 Likely to Stay"

    st.subheader("Prediction Result")
    st.write(label)
    st.metric("Churn Probability", f"{prob:.2f}")
