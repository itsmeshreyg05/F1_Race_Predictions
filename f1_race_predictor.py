# Add this cell to your notebook

import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model_path = 'race_outcome_model.pkl'
scaler_path = 'feature_scaler.pkl'

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure the files are in the correct directory.")

# Streamlit app
def run_streamlit_app():
    st.title("F1 Race Outcome Predictor")
    st.write("Enter the race details below to predict the winner.")

    # Input fields for race details
    grid = st.number_input("Grid Position (1-20)", min_value=1, max_value=20, value=1)
    quali_race_delta = st.number_input("Qualifying to Race Delta", value=0.0)
    track_difficulty = st.number_input("Track Difficulty (1-10)", min_value=1, max_value=10, value=5)
    weather_impact = st.number_input("Weather Impact (0-10)", min_value=0.0, max_value=10.0, value=2.0)
    driver_consistency = st.number_input("Driver Consistency (0-1)", min_value=0.0, max_value=1.0, value=0.85)
    constructor_form = st.number_input("Constructor Form (0-1)", min_value=0.0, max_value=1.0, value=0.9)
    avg_stint_length = st.number_input("Average Stint Length (laps)", min_value=0.0, value=15.0)

    # Predict button
    if st.button("Predict"):
        # Prepare input data
        input_data = pd.DataFrame([{
            'grid': grid,
            'Quali_Race_Delta': quali_race_delta,
            'Track_Difficulty': track_difficulty,
            'Weather_Impact': weather_impact,
            'Driver_Consistency': driver_consistency,
            'Constructor_Form': constructor_form,
            'Avg_Stint_Length': avg_stint_length
        }])

        # Scale the input data
        try:
            scaled_data = scaler.transform(input_data)
        except Exception as e:
            st.error(f"Error scaling input data: {e}")
            return

        # Make prediction
        try:
            prediction = model.predict(scaled_data)
            prediction_prob = model.predict_proba(scaled_data)[:, 1]
            st.success(f"Predicted Winner: {'Yes' if prediction[0] == 1 else 'No'}")
            st.info(f"Winning Probability: {prediction_prob[0]:.2%}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    run_streamlit_app()




# import os

# model_path = 'f1_winner_predictor_rf.pkl'
# scaler_path = 'f1_winner_predictor_xgb.pkl'

# print(f"Model file exists: {os.path.exists(model_path)}")
# print(f"Scaler file exists: {os.path.exists(scaler_path)}")

# # model_path = 'path/to/race_outcome_model.pkl'
# # scaler_path = 'path/to/feature_scaler.pkl'