import requests
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import base64
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from streamlit_lottie import st_lottie     # using streamlit lottie for animated objects insertion in web application


def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_animation_welcome = "https://assets5.lottiefiles.com/private_files/lf30_1TcivY.json"
lottie_anime_json_1 = load_lottie_url(lottie_animation_welcome)
st_lottie(lottie_anime_json_1, key = "welcome")

def main():
    
    st.title("Red Wine Quality App")
    
    # Create 12 input fields for numbers
    fixed_Acidity = st.number_input("Enter Fixed Acidity value", min_value=0.0, value=0.0)
    volatile_Acidity = st.number_input("Enter Volatile Acidity value", min_value=0.0, value=0.0)
    citric_Acid = st.number_input("Enter Citric Acid value", min_value=0.0, value=0.0)
    residual_Sugar = st.number_input("Enter Residual Sugar value", min_value=0.0, value=0.0)
    chlorides = st.number_input("Enter Chlorides value", min_value=0.0, value=0.0)
    free_SO2 = st.number_input("Enter Free Sulfur Dioxide value", min_value=0.0, value=0.0)
    total_SO2 = st.number_input("Enter Total Sulfur Dioxide value", min_value=0.0, value=0.0)
    density_min_value = 0.0
    density_max_value = 1.0
    density = st.number_input("Enter Density value", min_value=density_min_value, max_value=density_max_value, value=0.0)
    pH = st.number_input("Enter pH value", min_value=0.0, value=0.0)
    sulphates = st.number_input("Enter Sulphates value", min_value=0.0, value=0.0)
    alcohol = st.number_input("Enter Alcohol value", min_value=0.0, value=0.0)
    quality_min_value = 3
    quality_max_value = 8
    quality = st.number_input("Enter Quality value (integer)", min_value=quality_min_value, max_value=quality_max_value, value=3, step=1, format="%d")

    # Active SO2
    active_SO2 = total_SO2-free_SO2

    # Creating sweetness feature conditions

    conditions = [
        (residual_Sugar <= 4),
        (residual_Sugar > 4) & (residual_Sugar <= 12),
        (residual_Sugar > 12) & (residual_Sugar <= 45),
        (residual_Sugar > 45)
    ]

    choices = [1, 2, 3, 4] # 1-dry, 2-semi dry, 3-medium sweet, 4-sweet
    sweetness = np.select(conditions, choices, default=np.nan)
    sweetness = sweetness.astype(int)

    # Creating pH Acidity conditions

    conditions = [
        (pH <= 3.5),
        (pH >= 3.5) & ((pH < 3.6)),
        (pH >= 3.6)
    ]

    choices = [1, 2, 3] # 1 - high, 2 - medium, 3 - low
    pH_Acidity = np.select(conditions, choices, default=np.nan)
    pH_Acidity = pH_Acidity.astype(int)


    input_data = np.array([fixed_Acidity, volatile_Acidity, citric_Acid,
                           chlorides, free_SO2, total_SO2, density, pH, sulphates, alcohol, active_SO2, sweetness, pH_Acidity]).reshape(1, -1)

    model = joblib.load('model.joblib')
    
    # Make prediction
    prediction = model.predict(input_data)
    # st.write(prediction)

    # Display prediction
    if (prediction[0] == 1):
        st.write('Predicted Quality: Bad Quality Wine')
    else:
        st.write('Predicted Quality: Good Quality Wine')

if __name__ == "__main__":
    main()
