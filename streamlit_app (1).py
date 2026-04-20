
import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("Data Scientist Salary Prediction")
st.write("Enter the features below to predict the Data Scientist's salary.")

# Load the trained model
try:
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the label encoders
try:
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders_dict = pickle.load(f)
    st.success("Label encoders loaded successfully!")
except Exception as e:
    st.error(f"Error loading label encoders: {e}")
    st.stop()

# Input features from the user
rating = st.slider('Rating', min_value=1.0, max_value=5.0, value=3.5, step=0.1)

# Get unique values for dropdowns from encoders (if available) or provide default inputs
company_name_options = list(label_encoders_dict['Company Name'].classes_) if 'Company Name' in label_encoders_dict else []
company_name_input = st.selectbox('Company Name', options=company_name_options) if company_name_options else st.text_input('Company Name (Enter value if no options)')

job_title_options = list(label_encoders_dict['Job Title'].classes_) if 'Job Title' in label_encoders_dict else []
job_title_input = st.selectbox('Job Title', options=job_title_options) if job_title_options else st.text_input('Job Title (Enter value if no options)')

location_options = list(label_encoders_dict['Location'].classes_) if 'Location' in label_encoders_dict else []
location_input = st.selectbox('Location', options=location_options) if location_options else st.text_input('Location (Enter value if no options)')

employment_status_options = list(label_encoders_dict['Employment Status'].classes_) if 'Employment Status' in label_encoders_dict else []
employment_status_input = st.selectbox('Employment Status', options=employment_status_options) if employment_status_options else st.text_input('Employment Status (Enter value if no options)')

job_roles_options = list(label_encoders_dict['Job Roles'].classes_) if 'Job Roles' in label_encoders_dict else []
job_roles_input = st.selectbox('Job Roles', options=job_roles_options) if job_roles_options else st.text_input('Job Roles (Enter value if no options)')

salaries_reported = st.number_input('Salaries Reported', min_value=1, value=10)

# Make prediction
if st.button('Predict Salary'):
    try:
        # Encode categorical inputs
        encoded_company_name = label_encoders_dict['Company Name'].transform([company_name_input])[0] if 'Company Name' in label_encoders_dict else int(company_name_input)
        encoded_job_title = label_encoders_dict['Job Title'].transform([job_title_input])[0] if 'Job Title' in label_encoders_dict else int(job_title_input)
        encoded_location = label_encoders_dict['Location'].transform([location_input])[0] if 'Location' in label_encoders_dict else int(location_input)
        encoded_employment_status = label_encoders_dict['Employment Status'].transform([employment_status_input])[0] if 'Employment Status' in label_encoders_dict else int(employment_status_input)
        encoded_job_roles = label_encoders_dict['Job Roles'].transform([job_roles_input])[0] if 'Job Roles' in label_encoders_dict else int(job_roles_input)

        # Create a DataFrame for prediction
        input_data = pd.DataFrame([[
            rating,
            encoded_company_name,
            encoded_job_title,
            salaries_reported,
            encoded_location,
            encoded_employment_status,
            encoded_job_roles
        ]],
        columns=['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'])

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Salary: ${prediction:,.2f}")
    except ValueError as ve:
        st.error(f"Input Error: {ve}. Please ensure all inputs are valid and encoded values are provided if dropdowns are not available.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
