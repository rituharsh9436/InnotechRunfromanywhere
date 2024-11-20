import pandas as pd
import numpy as np
import streamlit as st
import pickle as pk
import scikit_learn

model = pk.load(open("Heart_disease_model.pkl", "rb"))
data = pd.read_csv("heart_disease.csv")

st.title("💓 Heart Disease Prediction App")
st.markdown(
    "This tool predicts the likelihood of heart disease based on medical information. "
    "Please fill in the details below to get a prediction."
)

st.subheader("Personal Information")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=["Male", "Female"], help="Select the patient's gender")
    gen = 1 if gender == 'Male' else 0
    age = st.number_input("Age", min_value=1, max_value=120, step=1, help="Enter the patient's age")

with col2:
    BMI = st.number_input("BMI (Body Mass Index)", min_value=0.0, help="Enter the patient's BMI (e.g., 25.3)")
    heartRate = st.number_input("Heart Rate", min_value=0.0, help="Enter the patient's resting heart rate")

st.subheader("Lifestyle Factors")
col3, col4 = st.columns(2)

with col3:
    currentSmoker = st.radio("Current Smoker?", options=[0, 1], help="0 for No, 1 for Yes")
    cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0.0, help="Number of cigarettes smoked per day")

with col4:
    BPMeds = st.radio("Blood Pressure Medication?", options=[0, 1], help="0 for No, 1 for Yes")
    diabetes = st.radio("Diabetes?", options=[0, 1], help="0 for No, 1 for Yes")

st.subheader("Medical History")
prevalentStroke = st.radio("History of Stroke?", options=[0, 1], help="0 for No, 1 for Yes")
prevalentHyp = st.radio("History of Hypertension?", options=[0, 1], help="0 for No, 1 for Yes")
totChol = st.number_input("Total Cholesterol", min_value=0.0, help="Total cholesterol level (mg/dL)")
sysBP = st.number_input("Systolic Blood Pressure", min_value=0.0, help="Systolic blood pressure (mmHg)")
diaBP = st.number_input("Diastolic Blood Pressure", min_value=0.0, help="Diastolic blood pressure (mmHg)")
glucose = st.number_input("Glucose Level", min_value=0.0, help="Blood glucose level (mg/dL)")

if st.button("Predict"):
    try:
        input_data = np.array([[gen, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, 
                                prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])
        output = model.predict(input_data)
        if output[0] == 0:
            st.success("✅ The patient is healthy. No signs of heart disease.")
        else:
            st.error("⚠️ The patient is at risk of heart disease. Consult a doctor.")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
