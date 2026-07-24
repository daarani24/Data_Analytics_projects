import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")

st.write(
    """
    This application predicts whether a telecom customer is likely to churn
    using Machine Learning models.
    """
)

lr_model = joblib.load("../models/logistic_regression_model.pkl")
rf_model = joblib.load("../models/random_forest_model.pkl")
xgb_model = joblib.load("../models/xgboost_model.pkl")
model_columns = joblib.load("../models/model_columns.pkl")
scaler = joblib.load("../models/scaler.pkl")

st.header("Select Machine Learning Model")

model_name = st.selectbox(
    "Choose Model",
    (
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    )
)

st.header("Customer Information")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0,1])
    partner = st.selectbox("Partner", ["No","Yes"])
    dependents = st.selectbox("Dependents", ["No","Yes"])
    phone = st.selectbox("Phone Service", ["No","Yes"])
    multiple = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )
    internet = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

with col2:
    security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )
    backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )
    protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )
    support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )
    tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )
    movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    st.subheader("Contract & Billing")

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0
)

if st.button("Predict"):

    # Create dataframe with all columns initialized to 0
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)

    # Numerical Features
    input_data["SeniorCitizen"] = senior
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly
    input_data["TotalCharges"] = total

    # Gender
    if gender == "Male":
        input_data["gender_Male"] = 1

    # Partner
    if partner == "Yes":
        input_data["Partner_Yes"] = 1

    # Dependents
    if dependents == "Yes":
        input_data["Dependents_Yes"] = 1

    # Phone Service
    if phone == "Yes":
        input_data["PhoneService_Yes"] = 1

    # Multiple Lines
    if multiple == "Yes":
        input_data["MultipleLines_Yes"] = 1
    elif multiple == "No phone service":
        input_data["MultipleLines_No phone service"] = 1

    # Internet Service
    if internet == "Fiber optic":
        input_data["InternetService_Fiber optic"] = 1
    elif internet == "No":
        input_data["InternetService_No"] = 1

    # Online Security
    if security == "Yes":
        input_data["OnlineSecurity_Yes"] = 1
    elif security == "No internet service":
        input_data["OnlineSecurity_No internet service"] = 1

    # Online Backup
    if backup == "Yes":
        input_data["OnlineBackup_Yes"] = 1
    elif backup == "No internet service":
        input_data["OnlineBackup_No internet service"] = 1

    # Device Protection
    if protection == "Yes":
        input_data["DeviceProtection_Yes"] = 1
    elif protection == "No internet service":
        input_data["DeviceProtection_No internet service"] = 1

    # Tech Support
    if support == "Yes":
        input_data["TechSupport_Yes"] = 1
    elif support == "No internet service":
        input_data["TechSupport_No internet service"] = 1

    # Streaming TV
    if tv == "Yes":
        input_data["StreamingTV_Yes"] = 1
    elif tv == "No internet service":
        input_data["StreamingTV_No internet service"] = 1

    # Streaming Movies
    if movies == "Yes":
        input_data["StreamingMovies_Yes"] = 1
    elif movies == "No internet service":
        input_data["StreamingMovies_No internet service"] = 1

    # Contract
    if contract == "One year":
        input_data["Contract_One year"] = 1
    elif contract == "Two year":
        input_data["Contract_Two year"] = 1

    # Paperless Billing
    if paperless == "Yes":
        input_data["PaperlessBilling_Yes"] = 1

    # Payment Method
    if payment == "Credit card (automatic)":
        input_data["PaymentMethod_Credit card (automatic)"] = 1
    elif payment == "Electronic check":
        input_data["PaymentMethod_Electronic check"] = 1
    elif payment == "Mailed check":
        input_data["PaymentMethod_Mailed check"] = 1

    # Select Model
    if model_name == "Logistic Regression":
        prediction = lr_model.predict(scaler.transform(input_data))[0]

    elif model_name == "Random Forest":
        prediction = rf_model.predict(input_data)[0]

    else:
        prediction = xgb_model.predict(input_data)[0]

        st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn.")
    else:
        st.success("✅ Customer is NOT likely to Churn.")