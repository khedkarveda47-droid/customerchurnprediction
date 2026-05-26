import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("📞 Customer Churn Prediction")

st.write("Predict whether customer will leave company")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])

senior = st.selectbox("Senior Citizen", [0, 1])

partner = st.selectbox("Has Partner", ["Yes", "No"])

dependents = st.selectbox("Has Dependents", ["Yes", "No"])

tenure = st.slider("Tenure (Months)", 1, 72, 12)

phone = st.selectbox("Phone Service", ["Yes", "No"])

paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

monthlycharges = st.number_input(
    "Monthly Charges",
    min_value=100.0,
    max_value=10000.0,
    value=500.0
)

totalcharges = st.number_input(
    "Total Charges",
    min_value=100.0,
    max_value=500000.0,
    value=5000.0
)

# Convert inputs to numbers
gender = 1 if gender == "Male" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0
phone = 1 if phone == "Yes" else 0
paperless = 1 if paperless == "Yes" else 0

# Predict button
if st.button("Predict Churn"):

    # Create EXACT 19 features
    sample = pd.DataFrame([[
        gender,
        senior,
        partner,
        dependents,
        tenure,
        phone,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        paperless,
        monthlycharges,
        totalcharges,
        0,
        0
    ]])

    # Prediction
    prediction = model.predict(sample)

    # Result
    if prediction[0] == 1:

        st.error("⚠ Customer Will Leave")

    else:

        st.success("✅ Customer Will Stay")

        