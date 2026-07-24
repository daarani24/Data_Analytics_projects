import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Risk Modeling",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Risk Modeling Dashboard")

st.write("""
Predict whether a loan application is likely to be approved
using a trained Machine Learning model.
""")

model = joblib.load("../models/credit_risk_model.pkl")
model_columns = joblib.load("../models/model_columns.pkl")

col1, col2 = st.columns(2)
with col1:

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes","No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0","1","2","3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate","Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["No","Yes"]
    )

with col2:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0
    )

    co_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0
    )

    loan_term = st.number_input(
        "Loan Amount Term",
        min_value=0.0
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0,0.0]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban","Semiurban","Rural"]
    )

if st.button("Predict Loan Status"):

    input_data = pd.DataFrame(0, index=[0], columns=model_columns)

    # Numerical Features
    input_data["ApplicantIncome"] = applicant_income
    input_data["CoapplicantIncome"] = co_income
    input_data["LoanAmount"] = loan_amount
    input_data["Loan_Amount_Term"] = loan_term
    input_data["Credit_History"] = credit_history

    # Gender
    if gender == "Male":
        input_data["Gender_Male"] = 1

    # Married
    if married == "Yes":
        input_data["Married_Yes"] = 1

    # Dependents
    if dependents == "1":
        input_data["Dependents_1"] = 1
    elif dependents == "2":
        input_data["Dependents_2"] = 1
    elif dependents == "3+":
        input_data["Dependents_3+"] = 1

    # Education
    if education == "Not Graduate":
        input_data["Education_Not Graduate"] = 1

    # Self Employed
    if self_employed == "Yes":
        input_data["Self_Employed_Yes"] = 1

    # Property Area
    if property_area == "Semiurban":
        input_data["Property_Area_Semiurban"] = 1
    elif property_area == "Urban":
        input_data["Property_Area_Urban"] = 1

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan is likely to be Approved")
    else:
        st.error("❌ Loan is likely to be Rejected")

    st.write(f"**Approval Probability:** {probability[1]*100:.2f}%")
    st.write(f"**Rejection Probability:** {probability[0]*100:.2f}%")