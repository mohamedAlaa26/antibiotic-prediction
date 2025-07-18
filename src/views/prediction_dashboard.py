import streamlit as st
import pandas as pd
import requests
import sys
import os

sys.path.append(os.path.dirname(__file__))

from request_preprocessing import change_types


st.set_page_config(page_title="Antibiotic Predictor (Input Only)", layout="centered")

st.title("🧪 Enter Clinical Features")

# Grouped input fields
st.header("🩺 Vital Signs")
median_heartrate = st.number_input("Median Heart Rate", value=80.0)
median_resprate = st.number_input("Median Respiratory Rate", value=18.0)
median_temp = st.number_input("Median Temperature (°C)", value=37.0)
median_sysbp = st.number_input("Median Systolic BP", value=120.0)
median_diasbp = st.number_input("Median Diastolic BP", value=80.0)

st.header("🧬 Lab Results")
median_wbc = st.number_input("WBC Count", value=7.0)
median_hgb = st.number_input("Hemoglobin", value=13.5)
median_plt = st.number_input("Platelet Count", value=250.0)
median_na = st.number_input("Sodium", value=140.0)
median_hco3 = st.number_input("Bicarbonate (HCO3)", value=22.0)
median_bun = st.number_input("BUN", value=15.0)
median_cr = st.number_input("Creatinine", value=1.0)

st.header("📊 Encoded Inputs")
culture_description_encoded = st.selectbox("Culture Description (encoded)", ['URINE', 'BLOOD', 'RESPIRATORY'])
age = st.number_input("AGE", value=50)
gender = st.selectbox("Gender (encoded)", ['male', 'female'])

# collect inputs into a dictionary
inputs = {
    "median_heartrate": median_heartrate,
    "median_resprate": median_resprate,
    "median_temp": median_temp,
    "median_sysbp": median_sysbp,
    "median_diasbp": median_diasbp,
    "median_wbc": median_wbc,
    "median_hgb": median_hgb,
    "median_plt": median_plt,
    "median_na": median_na,
    "median_hco3": median_hco3,
    "median_bun": median_bun,
    "median_cr": median_cr,
    "culture_description_encoded": culture_description_encoded,
    "age": age,
    "gender":gender
}



# Change types of inputs
change_types(inputs)


# Submit button (currently no model)
if st.button("Submit"):
    """
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=inputs)
        
        if response.status_code == 200:
            result = response.json()
            st.success("Prediction received!")
            st.json(result)  # Display the JSON response in Streamlit
        else:
            st.error(f"Request failed with status code {response.status_code}")
            st.text(response.text)  # Show error details
    except requests.exceptions.RequestException as e:
        st.error("Error connecting to the prediction API.")
        st.text(str(e))

    """
    print(inputs['gender'],inputs['age'])
    st.success("Inputs received! (Model prediction is disabled for now)")

