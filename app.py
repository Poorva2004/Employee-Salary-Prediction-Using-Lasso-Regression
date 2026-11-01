
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessor
model = joblib.load("final_lasso_salary_model.pkl")
preprocessor = joblib.load("salary_preprocessor.pkl")

# Get categories learned during model training
categories = preprocessor.named_transformers_["cat"].categories_

# Page configuration
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered"
)

# Application heading
st.title("Employee Salary Prediction")
st.write("Predict employee salary using Lasso Regression.")

st.divider()

# Employee input form
st.subheader("Enter Employee Details")

with st.form("salary_prediction_form"):

    job_title = st.selectbox(
        "Job Title",
        categories[0].tolist()
    )

    experience_years = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=60,
        value=2
    )

    education_level = st.selectbox(
        "Education Level",
        categories[1].tolist()
    )

    skills_count = st.number_input(
        "Number of Skills",
        min_value=0,
        value=5
    )

    industry = st.selectbox(
        "Industry",
        categories[2].tolist()
    )

    company_size = st.selectbox(
        "Company Size",
        categories[3].tolist()
    )

    location = st.selectbox(
        "Location",
        categories[4].tolist()
    )

    remote_work = st.selectbox(
        "Remote Work",
        categories[5].tolist()
    )

    certifications = st.number_input(
        "Number of Certifications",
        min_value=0,
        value=2
    )

    submitted = st.form_submit_button("Predict Salary")

# Prediction
if submitted:

    employee_data = pd.DataFrame([{
        "job_title": job_title,
        "experience_years": experience_years,
        "education_level": education_level,
        "skills_count": skills_count,
        "industry": industry,
        "company_size": company_size,
        "location": location,
        "remote_work": remote_work,
        "certifications": certifications
    }])

    # Transform employee data
    processed_data = preprocessor.transform(employee_data)

    # Predict salary
    prediction = model.predict(processed_data)[0]

    # Display result
    st.success("Salary Prediction Completed!")

    st.metric(
        label="Predicted Salary",
        value=f"{prediction:,.2f} salary units"
    )

    st.info(
        "This is an estimated salary based on the dataset "
        "and is not a guaranteed salary offer."
    )
