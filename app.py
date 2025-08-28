# app.py : Streamlit web app for employee salary classification

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_salary_model.pkl")

le_edu = joblib.load("le_edu.pkl")
le_gender = joblib.load("le_gender.pkl")
le_job = joblib.load("le_job.pkl")
# Load others if needed (le_workclass, le_gender, etc.)

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")

# Sidebar inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education_level = st.sidebar.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])



job_title = st.sidebar.selectbox(
    "Job Title",
    [
        "Account Manager",
        "Accountant",
        "Administrative Assistant",
        "Business Analyst",
        "Consultant",
        "Customer Service Representative",
        "Data Analyst",
        "Data Scientist",
        "Designer",
        "Developer",
        "Engineer",
        "Finance Manager",
        "HR Manager",
        "IT Support",
        "Lawyer",
        "Marketing Manager",
        "Operations Manager",
        "Product Manager",
        "Project Manager",
        "Sales Associate",
        "Sales Manager",
        "Software Engineer",
        "Teacher",
        "Technician",
        "Writer"
    ]
)
years_of_experience = st.sidebar.number_input("Years of Experience", min_value=0, max_value=50, value=5)

# Create input DataFrame
input_df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education_level],
    "Job Title": [job_title],
    "Years of Experience": [years_of_experience]
})

st.write("### Input Data")
st.write(input_df)

# Prediction
if st.button("Classify"):
  try:
        input_df["Gender"] = le_gender.transform(input_df["Gender"])
        input_df["Education Level"] = le_edu.transform(input_df["Education Level"])

        
        input_df["Job Title"] = le_job.transform(input_df["Job Title"])

        prediction = model.predict(input_df)
        st.success(f"🎯 Prediction: {prediction[0]}")
  except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

