import numpy as np
import pickle
import streamlit as st

# Set up the Streamlit page title and configuration
st.set_page_config(
    page_title="Salary Prediction App", page_icon="💼", layout="centered"
)
# Load the pre-trained machine learning model from pickle file
model = pickle.load(open("model.pkl", "rb"))

# App header
st.title("Salary Prediction Model")
st.write(
    "This is a machine learning web app to predict the expected salary using"
    " the linear regression model."
)
# --- 1. Multiple User Inputs Section ---

# Input 1: Years of Experience
years_experience = st.number_input(
    "Years of Experience", min_value=0.0, max_value=40.0, step=0.5, value=1.0
)

# Input 2: Job Level (Selectbox / Dropdown)
job_level = st.selectbox(
    "Select Job Level", ["Junior", "Mid-Level", "Senior", "Lead"]
)

# Input 3: Number of Certifications (Number Input)
certifications_count = st.number_input(
    "Enter Number of Certifications", min_value=0, max_value=15, step=1, value=0
)
# --- 2. Action Button and Prediction Logic ---

# Create a clear button for triggering the model prediction
if st.button("Predict Salary"):
    # Map categorical job level to numerical values safely inside the button click
    level_mapping = {"Junior": 1, "Mid-Level": 2, "Senior": 3, "Lead": 4}
    encoded_job_level = level_mapping[job_level]

    # Combine all user inputs into a 2D array matching the training feature columns
    user_features = np.array(
        [[years_experience, encoded_job_level, certifications_count]]
    )

    try:
        # Predict the salary using the loaded machine learning model
        predicted_salary = model.predict(user_features)
        final_salary = round(predicted_salary[0], 2)

    # Display the success result message to the user
        st.success(f"The estimated annual salary is: ${final_salary:,.2f}")

    except Exception as e:
    # Handle any potential errors during prediction
        st.error(
        "An error occurred during prediction. Please make sure your model is"
        f" loaded properly. Details: {e}"
        )