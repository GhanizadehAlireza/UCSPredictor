# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 00:32:59 2025

@author: ALI REZA
"""


# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
# Set the page title and icon. This is the first thing Streamlit runs.
st.set_page_config(
    page_title="UCSPredictor (test)",
    page_icon="🛣️",
    layout="wide"
)

# --- Model Loading ---
# Use st.cache_resource to load the model only once, which improves performance.
# This is like pre-loading a heavy piece of testing equipment.
@st.cache_resource
def load_model():
    """Loads the pre-trained pavement performance model."""
    model = joblib.load('LightGBoost-UCS-Magnetite-IOT.joblib')
    return model

model = load_model()

# --- Application Header ---
st.title("UCS Prediction test Program")
st.markdown("""
This is a demo.
""")

# --- Sidebar for User Inputs ---
# We use a sidebar to keep the main interface clean.
st.sidebar.header("Inputs")

# Create sliders for the 4 numerical inputs.
# These widgets are intuitive for selecting numerical values.
# The names and ranges are analogous to your model's inputs.


def user_input_features():
    """Creates sidebar widgets and returns user inputs as a DataFrame."""
    f1 = st.sidebar.slider(
        'PC (%)', min_value=5.0, max_value=9.0, value=7.0, step=0.1
    )
    f2 = st.sidebar.slider(
        'MC (%)', min_value=7.5, max_value=12.0, value=10.0, step=0.1
    )
    f3 = st.sidebar.slider(
        'CT (day)', min_value=7.0, max_value=91.0, value=28.0, step=1.0
    )
    # Using number_input for traffic because it's a large number
    f4 = st.sidebar.number_input(
        'CE (kN-m/m²)', min_value=600.0, max_value=2700.0, value=2700.0, step=100.0
    ) # Convert millions to actual number for the model

    # Store the inputs in a dictionary
    data = {
        'PC (%)': f1,
        'MC (%)': f2,
        'CT (day)': f3,
        'CE (kN-m/m²)': f4
    }
    
    # Convert the dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Display User Inputs ---
# Show the user the values they have selected.
st.subheader('Specified Input Parameters')
st.dataframe(input_df, use_container_width=True)

# --- Prediction and Display ---
# Create a button to trigger the prediction.
if st.sidebar.button('Predict UCS'):
    # The model expects a NumPy array, so we convert the DataFrame's values.
    # The shape should be (1, 4) for a single prediction with 4 features.
    prediction_input = input_df.values.reshape(1, -1)
    
    # Make the prediction
    prediction = model.predict(prediction_input)
    
    # Display the result
    st.subheader('Prediction Result')
    
    # The st.metric widget is excellent for displaying a single, key number.
    st.metric(
        label="Predicted UCS",
        value=f"{prediction[0]:.2f} MPa",
        delta=f"{prediction[0]-2.1:.2f} UCS vs. target", # Optional: Compare to a baseline (e.g., 15 years)
        delta_color="normal" # "inverse" or "off"
    )
    
    st.info("Note: The 'delta' compares the prediction to a target UCS.", icon="ℹ️")

else:
    st.info('Click the "Predict" button in the sidebar to see the result.')


