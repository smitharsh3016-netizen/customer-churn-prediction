import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

model = joblib.load("customer_churn_model.pkl")

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict whether they may churn.")

gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.number_input("Tenure in months", min_value=0, max_value=100, value=12)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)
internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)
online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)
online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)
device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)
tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)
streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)
streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)
contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=85.50
)
total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1026.00
)

if st.button("Predict Churn"):
    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(customer_data)[0]
    probability = model.predict_proba(customer_data)[0][1]

    if prediction == 1:
        st.error("Prediction: This customer is likely to churn.")
    else:
        st.success("Prediction: This customer is likely to stay.")

    st.write(f"Churn probability: **{probability:.2%}**")
    st.subheader("Customer Input Summary")
    st.dataframe(customer_data, hide_index=True, use_container_width=True)
