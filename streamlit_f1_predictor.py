import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

# Set page configuration as the FIRST Streamlit command
st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")

# Custom CSS for enhanced visual appeal
st.markdown("""
    <style>
    /* Main background with a subtle gradient */
    .main {
        background: linear-gradient(to bottom, #1a1a1a, #2d2d2d);
        color: #ffffff;
    }
    /* Title styling */
    h1 {
        color: #ff1801; /* F1 red */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    /* Markdown text */
    .stMarkdown {
        color: #e0e0e0; /* Light gray for readability */
    }
    /* Form styling */
    .stForm {
        background-color: #2a2a2a; /* Darker form background */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(255, 24, 1, 0.3); /* Red-tinted shadow */
        border: 1px solid #555555; /* Silver border */
    }
    /* Button styling */
    .stButton>button {
        background-color: #ff1801; /* F1 red */
        color: #ffffff;
        border-radius: 5px;
        border: 2px solid #ffffff; /* White border for contrast */
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #d81401; /* Darker red on hover */
        transform: scale(1.05); /* Slight zoom effect */
        box-shadow: 0 0 10px rgba(255, 24, 1, 0.5);
    }
    /* Input fields */
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: #3c3c3c; /* Dark gray */
        color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #aaaaaa; /* Silver border */
    }
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #e0e0e0; /* Light gray labels */
    }
    /* Success and info messages */
    .stAlert {
        background-color: #3c3c3c;
        color: #ffffff;
        border-radius: 5px;
        border: 1px solid #ff1801;
    }
    /* Footer */
    .footer {
        background: linear-gradient(to right, #ffffff 50%, #000000 50%);
        color: #ff1801;
        padding: 10px;
        text-align: center;
        border-radius: 5px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏎️ Formula 1 Race Outcome Predictor")
st.markdown("""
Predict a driver's finishing position with this sleek, F1-inspired tool!  
Enter race and driver details below to get a prediction.
""")

# Function to load data
@st.cache_data
def load_data():
    data_dir = "Formula 1 World Championship (1950 - 2024)"
    required_files = {
        'races': 'races.csv',
        'results': 'results.csv',
        'drivers': 'drivers.csv',
        'pit_stops': 'pit_stops.csv',
        'circuits': 'circuits.csv',
        'constructors': 'constructors.csv'
    }
    
    dataframes = {}
    for key, file in required_files.items():
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            dataframes[key] = pd.read_csv(file_path)
        else:
            st.error(f"Missing file: {file_path}. Ensure all data files are in '{data_dir}'.")
            return None
    return dataframes

# Function to prepare data
def prepare_data(dataframes):
    if dataframes is None:
        return None
    
    # Unpack dataframes
    races = dataframes['races']
    results = dataframes['results']
    drivers = dataframes['drivers']
    pit_stops = dataframes['pit_stops']
    circuits = dataframes['circuits']
    constructors = dataframes['constructors']
    
    # Merge datasets
    data = (results.merge(races, on='raceId')
            .merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
            .merge(circuits[['circuitId', 'name']], on='circuitId', suffixes=('', '_circuit'))
            .merge(constructors[['constructorId', 'name']], on='constructorId', suffixes=('', '_constructor')))
    
    # Create driver and team names
    data['Driver'] = data['forename'] + ' ' + data['surname']
    data['Team'] = data['name_constructor']
    
    # Filter for recent years (2018-2023)
    data = data[data['year'] >= 2018]
    
    # Aggregate pit stops
    pit_agg = pit_stops.groupby(['raceId', 'driverId']).agg(
        pit_stop_count=('stop', 'count'),
        total_pit_time=('milliseconds', 'sum')
    ).reset_index()
    data = data.merge(pit_agg, on=['raceId', 'driverId'], how='left')
    data['pit_stop_count'] = data['pit_stop_count'].fillna(0)
    data['total_pit_time'] = data['total_pit_time'].fillna(0)
    
    # Feature engineering
    data['rolling_3race_avg'] = data.groupby('driverId')['positionOrder'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    data['track_avg_position'] = data.groupby(['driverId', 'circuitId'])['positionOrder'].transform('mean')
    data['season_progress'] = data.groupby('year')['round'].transform(lambda x: x / x.max())
    data['team_rolling_avg'] = data.groupby(['constructorId', 'year'])['positionOrder'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Simulate weather data
    data['weather_temp'] = np.random.uniform(15, 30, size=len(data))
    data['weather_rainfall'] = np.random.choice([0, 0.1, 0.2, 0.5], size=len(data), p=[0.7, 0.2, 0.08, 0.02])
    
    # Target variable
    data['next_position'] = data['positionOrder']
    
    # Convert object columns to string to avoid Arrow serialization issues
    for col in ['Driver', 'name', 'Team']:
        data[col] = data[col].astype('string')
    
    # Select features
    features = ['Driver', 'name', 'Team', 'grid', 'rolling_3race_avg', 'track_avg_position', 
                'season_progress', 'team_rolling_avg', 'weather_temp', 'weather_rainfall', 'next_position']
    return data[features]

# Load and prepare data
dataframes = load_data()
if dataframes is not None:
    data = prepare_data(dataframes)
    
    if data is not None:
        # Display data types as dictionary to avoid Arrow error
        st.write("Data Types Before Model Training:")
        st.write(dict(data.dtypes))
        
        # Train model
        X = data.drop('next_position', axis=1)
        y = data['next_position']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define features
        numeric_features = ['grid', 'rolling_3race_avg', 'track_avg_position', 'season_progress', 
                            'team_rolling_avg', 'weather_temp', 'weather_rainfall']
        categorical_features = ['Driver', 'name', 'Team']
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])
        
        # Fit preprocessor
        preprocessor.fit(X_train)
        
        # Transform training and validation data
        X_train_transformed = preprocessor.transform(X_train)
        X_val_transformed = preprocessor.transform(X_val)
        
        # Train XGBoost regressor
        regressor = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            early_stopping_rounds=50
        )
        
        try:
            regressor.fit(X_train_transformed, y_train, eval_set=[(X_val_transformed, y_val)], verbose=False)
            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            st.stop()
        
        # Streamlit form for predictions
        st.header("🏁 Predict Race Outcome")
        with st.form("prediction_form"):
            st.subheader("Race and Driver Details")
            col1, col2 = st.columns(2)
            
            with col1:
                driver = st.selectbox("Driver", sorted(data['Driver'].unique()), help="Pick your driver!")
                race_name = st.selectbox("Race", sorted(data['name'].unique()), help="Choose the Grand Prix.")
                team = st.selectbox("Team", sorted(data['Team'].unique()), help="Select the team.")
                grid = st.number_input("Grid Position", min_value=1, max_value=20, value=1, help="Starting position.")
            
            with col2:
                rolling_3race_avg = st.number_input("3-Race Avg Position", min_value=1.0, max_value=20.0, value=5.0, help="Driver's avg position in last 3 races.")
                track_avg_position = st.number_input("Track Avg Position", min_value=1.0, max_value=20.0, value=5.0, help="Driver's avg at this track.")
                season_progress = st.slider("Season Progress", min_value=0.0, max_value=1.0, value=0.5, help="0 = season start, 1 = end.")
                team_rolling_avg = st.number_input("Team Avg Position", min_value=1.0, max_value=20.0, value=5.0, help="Team’s recent performance.")
                weather_temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=40.0, value=25.0, help="Race day temp.")
                weather_rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=10.0, value=0.0, help="Rain expected?")
            
            submit = st.form_submit_button("Predict Now")
            
            if submit:
                # Create input DataFrame
                input_data = pd.DataFrame([{
                    'Driver': driver,
                    'name': race_name,
                    'Team': team,
                    'grid': grid,
                    'rolling_3race_avg': rolling_3race_avg,
                    'track_avg_position': track_avg_position,
                    'season_progress': season_progress,
                    'team_rolling_avg': team_rolling_avg,
                    'weather_temp': weather_temp,
                    'weather_rainfall': weather_rainfall
                }])
                
                # Convert object columns to string
                for col in ['Driver', 'name', 'Team']:
                    input_data[col] = input_data[col].astype('string')
                
                try:
                    # Transform input data
                    input_transformed = preprocessor.transform(input_data)
                    
                    # Make prediction
                    prediction = regressor.predict(input_transformed)[0]
                    st.success(f"Predicted Finishing Position: **{round(prediction, 2)}**")
                    st.info("Note: Predictions are based on historical data and simulated weather. Actual results may vary.")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Data preparation failed. Please check the data files.")
else:
    st.warning("Please ensure all required data files are in the 'Formula 1 World Championship (1950 - 2024)' directory.")

# Footer with custom class
st.markdown('<div class="footer">Built with ❤️ using Streamlit | Data Source: Formula 1 World Championship (1950 - 2024)</div>', unsafe_allow_html=True)