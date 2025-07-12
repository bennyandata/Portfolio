import streamlit as st
import joblib
import pandas as pd

# Load model

import os
model_path = os.path.join(os.path.dirname(__file__), "fraud_detection_lr_pipeline.pkl")
model = joblib.load(model_path)

# model = joblib.load("fraud_detection_lr_pipeline.pkl")

st.title("Fraud Detection App")

st.write("Enter transaction details below:")

amount = st.number_input("Amount", value=100.0)
oldbalanceOrg = st.number_input("Old Balance Origin", value=1000.0)
newbalanceOrig = st.number_input("New Balance Origin", value=900.0)
oldbalanceDest = st.number_input("Old Balance Dest", value=0.0)
newbalanceDest = st.number_input("New Balance Dest", value=0.0)
step = st.number_input("Step", value=1)

# Collect input into dataframe
input_df = pd.DataFrame({
    'step': [step],
    'amount': [amount],
    'oldbalanceOrg': [oldbalanceOrg],
    'newbalanceOrig': [newbalanceOrig],
    'oldbalanceDest': [oldbalanceDest],
    'newbalanceDest': [newbalanceDest]
})

# Predict
if st.button("Predict Fraud"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.error("⚠️ Fraud Detected!")
    else:
        st.success("✅ Transaction seems normal.")
