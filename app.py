import pickle
import bz2
import streamlit as st
import numpy as np
from app_logger import log
import warnings
warnings.filterwarnings("ignore")

# Load Classification and Regression models
with bz2.BZ2File('Classification.pkl', 'rb') as file:
    model_C = pickle.load(file)
with bz2.BZ2File('Regression.pkl', 'rb') as file:
    model_R = pickle.load(file)

# Define the Streamlit app
def main():
    st.title("Forest Fire Prediction App")

    # Sidebar for user inputs
    st.sidebar.title("Enter Parameters")
    temperature = st.sidebar.number_input("Temperature (°C)", value=25.0, step=0.1)
    wind_speed = st.sidebar.number_input("Wind Speed (km/h)", value=10, step=1)
    ffmc = st.sidebar.number_input("FFMC", value=85.0, step=0.1)
    dmc = st.sidebar.number_input("DMC", value=150.0, step=0.1)
    isi = st.sidebar.number_input("ISI", value=10.0, step=0.1)

    # Predictions
    if st.sidebar.button("Predict (Classification)"):
        try:
            features = np.array([temperature, wind_speed, ffmc, dmc, isi]).reshape(1, -1)
            prediction = model_C.predict(features)[0]
            if prediction == 0:
                st.sidebar.write("Forest is Safe!")
            else:
                st.sidebar.write("Forest is in Danger!")
            log.info('Prediction done for Classification model')
        except Exception as e:
            st.sidebar.error('Input error, check input', e)
    
    if st.sidebar.button("Predict (Regression)"):
        try:
            features = np.array([temperature, wind_speed, ffmc, dmc, isi]).reshape(1, -1)
            prediction = model_R.predict(features)[0]
            if prediction > 15:
                st.sidebar.warning(f"Fuel Moisture Code index is {prediction:.4f} ---- Warning!!! High hazard rating")
            else:
                st.sidebar.success(f"Fuel Moisture Code index is {prediction:.4f} ---- Safe.. Low hazard rating")
            log.info('Prediction done for Regression model')
        except Exception as e:
            st.sidebar.error('Input error, check input', e)

if __name__ == "__main__":
    main()
