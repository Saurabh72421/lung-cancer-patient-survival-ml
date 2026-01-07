import streamlit as st
import numpy as np
import joblib

st.title("🫁 Lung Cancer Survival Prediction")

model = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')

age = st.number_input("Age", 18, 100)
smoking = st.selectbox("Smoking Status", ["No", "Yes"])
smoking = 1 if smoking == "Yes" else 0

if st.button("Predict"):
    data = scaler.transform([[age, smoking]])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if pred == 1:
        st.success(f"Likely to Survive (Probability: {prob:.2f})")
    else:
        st.error(f"High Risk (Probability: {prob:.2f})")