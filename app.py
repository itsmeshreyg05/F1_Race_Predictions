import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration as the FIRST Streamlit command
st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")

# Custom CSS (kept from your code)
st.markdown("""
    <style>
    /* ... (your full CSS remains here) ... */
    .main { background: linear-gradient(to bottom, #1a1a1a, #2d2d2d); color: #ffffff; }
    h1, h2, h3 { color: #ff1801; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); }
    .stMarkdown { color: #e0e0e0; }
    .stForm { background-color: #2a2a2a; border-radius: 10px; padding: 20px; box-shadow: 0 4px 12px rgba(255, 24, 1, 0.3); border: 1px solid #555555; }
    .stButton>button { background-color: #ff1801; color: #ffffff; border-radius: 5px; border: 2px solid #ffffff; font-weight: bold; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #d81401; transform: scale(1.05); box-shadow: 0 0 10px rgba(255, 24, 1, 0.5); }
    .stSelectbox, .stNumberInput, .stSlider { background-color: #3c3c3c; color: #ffffff; border-radius: 5px; padding: 10px; border: 1px solid #aaaaaa; }
    .stSelectbox label, .stNumberInput label, .stSlider label { color: #e0e0e0; }
    .stAlert { background-color: #3c3c3c; color: #ffffff; border-radius: 5px; border: 1px solid #ff1801; }
    .stMetric { background-color: #2a2a2a; border-radius: 8px; padding: 15px; border: 1px solid #555; box-shadow: 0 2px 6px rgba(0,0,0,0.2); }
    .stMetricLabel { color: #e0e0e0; font-size: 0.9em; }
    .stMetricValue { color: #ff1801; font-size: 1.5em; font-weight: bold; }
    .footer { background: linear-gradient(to right, #ffffff 50%, #000000 50%); color: #ff1801; padding: 10px; text-align: center; border-radius: 5px; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- Cached Functions for Efficiency ---

@st.cache_data # Cache the raw data loading
def load_data():
    """Loads all necessary raw CSV files."""
    data_dir = "Formula 1 World Championship (1950 - 2024)"
    required_files = {
        'races': 'races.csv', 'results': 'results.csv', 'drivers': 'drivers.csv',
        'pit_stops': 'pit_stops.csv', 'circuits': 'circuits.csv', 'constructors': 'constructors.csv'
    }
    dataframes = {}
    all_files_found = True
    for key, file in required_files.items():
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            try:
                dataframes[key] = pd.read_csv(file_path)
            except Exception as e:
                st.error(f"Error reading {file_path}: {e}")
                all_files_found = False
        else:
            st.error(f"Missing file: {file_path}. Ensure all data files are in '{data_dir}'.")
            all_files_found = False
    return dataframes if all_files_found else None

@st.cache_data # Cache data preparation
def prepare_data(_dataframes): # Use _dataframes to avoid conflict with global 'dataframes'
    """Merges, cleans, and engineers features on the loaded data."""
    if _dataframes is None:
        return None
    try:
        # Unpack dataframes
        races = _dataframes['races']
        results = _dataframes['results']
        drivers = _dataframes['drivers']
        pit_stops = _dataframes['pit_stops']
        circuits = _dataframes['circuits']
        constructors = _dataframes['constructors']

        # --- Merging ---
        data = (results.merge(races[['raceId', 'year', 'round', 'circuitId', 'name']], on='raceId')
                .merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
                .merge(circuits[['circuitId', 'name']], on='circuitId', suffixes=('', '_circuit'))
                .merge(constructors[['constructorId', 'name']], on='constructorId', suffixes=('', '_constructor')))

        # --- Basic Renaming & Cleaning ---
        data.rename(columns={'name': 'raceName', 'name_constructor': 'Team'}, inplace=True)
        data['Driver'] = data['forename'] + ' ' + data['surname']

        # Convert relevant columns to numeric, coercing errors
        cols_to_numeric = ['positionOrder', 'grid', 'points', 'laps', 'milliseconds', 'total_pit_time', 'pit_stop_count']
        for col in cols_to_numeric:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Filter years BEFORE creating rolling features if desired
        data = data[data['year'] >= 2018].copy()
        if data.empty:
            st.warning("No data found for years >= 2018.")
            return None

        # Aggregate and merge pit stops
        pit_agg = pit_stops.groupby(['raceId', 'driverId']).agg(
            pit_stop_count=('stop', 'count'),
            total_pit_time=('milliseconds', 'sum')
        ).reset_index()
        data = data.merge(pit_agg, on=['raceId', 'driverId'], how='left')
        # Fill NaNs introduced by merging/coercion
        data['pit_stop_count'] = data['pit_stop_count'].fillna(0).astype(int)
        data['total_pit_time'] = data['total_pit_time'].fillna(0)
        data['grid'] = data['grid'].fillna(data['grid'].median()) # Fill grid NaNs
        data['positionOrder'] = data['positionOrder'].fillna(data['positionOrder'].median()) # Fill position NaNs
        data['positionOrder'] = data['positionOrder'].astype(int) # Ensure target is int

        # --- Feature Engineering ---
        data.sort_values(by=['driverId', 'year', 'round'], inplace=True)
        data['rolling_3race_avg'] = data.groupby('driverId')['positionOrder'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        data['track_avg_position'] = data.groupby(['driverId', 'circuitId'])['positionOrder'].transform(
             lambda x: x.shift(1).mean()
        )
        data['season_progress'] = data.groupby('year')['round'].transform(lambda x: x / x.max())
        data['team_rolling_avg'] = data.groupby(['constructorId', 'year'])['positionOrder'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )

        # --- Target Variable ---
        data['target_position'] = data['positionOrder']

        # --- Final Cleaning & Selection ---
        # Define features *including weather* which will be input by user for prediction
        features = ['Driver', 'raceName', 'Team', 'grid', 'rolling_3race_avg',
                    'track_avg_position', 'season_progress', 'team_rolling_avg',
                    'weather_temp', 'weather_rainfall'] # Weather added here
        target = 'target_position'
        final_cols = features + [target]

        # Create placeholder weather columns if they don't exist (will be filled later)
        if 'weather_temp' not in data.columns: data['weather_temp'] = np.nan
        if 'weather_rainfall' not in data.columns: data['weather_rainfall'] = np.nan

        data_final = data.copy() # Use all columns for now, select later in training/prediction

        # Handle remaining NaNs in engineered features
        eng_features_to_fill = ['rolling_3race_avg', 'track_avg_position', 'team_rolling_avg']
        for col in eng_features_to_fill:
             if data_final[col].isnull().any():
                  data_final[col] = data_final[col].fillna(data_final[col].median())

        # Convert relevant columns to string type for consistency
        for col in ['Driver', 'raceName', 'Team']:
            if col in data_final.columns:
                data_final[col] = data_final[col].astype('string')

        # Drop rows where target is still NaN (shouldn't happen after earlier fill but safety check)
        data_final = data_final.dropna(subset=[target])

        return data_final

    except KeyError as e:
         st.error(f"Data preparation failed: Missing expected column {e}. Check CSV files and merge logic.")
         return None
    except Exception as e:
        st.error(f"Error during data preparation: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

@st.cache_resource # Cache the trained model and preprocessor
def train_model(_data): # Use _data to avoid scope conflict
    """Trains the XGBoost Regressor and returns the fitted model, preprocessor, and validation data."""
    if _data is None or _data.empty:
        st.error("Cannot train model: Prepared data is empty or not loaded.")
        return None, None, None, None, None

    try:
        # Define features for the model (including weather, excluding target)
        features = ['Driver', 'raceName', 'Team', 'grid', 'rolling_3race_avg',
                    'track_avg_position', 'season_progress', 'team_rolling_avg',
                    'weather_temp', 'weather_rainfall']
        target = 'target_position'

        # Ensure all selected features exist in the prepared data
        missing_features = [f for f in features if f not in _data.columns]
        if missing_features:
            st.error(f"Cannot train model: Missing required features in prepared data: {missing_features}")
            return None, None, None, None, None

        X = _data[features]
        y = _data[target]

        # Ensure target is numeric
        y = pd.to_numeric(y, errors='coerce')
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        if X.empty or y.empty:
             st.error("No valid training data after target cleaning.")
             return None, None, None, None, None
        y = y.astype(int)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define numeric and categorical features *based on the final feature list*
        numeric_features = [f for f in features if f in X_train.select_dtypes(include=np.number).columns]
        categorical_features = [f for f in features if f in X_train.select_dtypes(include=['string', 'object', 'category']).columns]

        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='drop' # Drop any columns not specified
            )

        # Fit preprocessor on TRAINING data only
        print("Fitting preprocessor...")
        preprocessor.fit(X_train)
        print("Preprocessor fitted.")

        # Transform training and validation data
        print("Transforming data...")
        X_train_transformed = preprocessor.transform(X_train)
        X_val_transformed = preprocessor.transform(X_val)
        print("Data transformed.")

        # Define and Train XGBoost Regressor
        regressor = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1
        )

        print("Training XGBoost model...")
        regressor.fit(X_train_transformed, y_train,
                      eval_set=[(X_val_transformed, y_val)],
                      verbose=False)
        print("Model training complete.")

        return preprocessor, regressor, X_val, y_val, X_val_transformed # Return validation sets too

    except Exception as e:
        st.error(f"Error during model training: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, None

# --- Main App Logic ---

# Title and description
st.title("🏎️ Formula 1 Race Outcome Predictor")


# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose a section:",
    ["Predict Outcome", "Data Overview & EDA", "Model Performance"]
)
st.sidebar.markdown("---")
st.sidebar.info("Data Source: F1 Kaggle Dataset (2018-2023)")


# Load data (cached)
dataframes_raw = load_data()

if dataframes_raw:
    # Prepare data (cached)
    prepared_data = prepare_data(dataframes_raw)

    if prepared_data is not None and not prepared_data.empty:
        # Train model and get preprocessor, validation data (cached resource)
        preprocessor, regressor, X_val, y_val, X_val_transformed = train_model(prepared_data)

        # ============ Predict Outcome Section ============
        if app_mode == "Predict Outcome":
            st.header("🏁 Predict Race Outcome")
            st.markdown("Enter race details to predict a driver's finishing position.")

            if preprocessor and regressor:
                with st.form("prediction_form"):
                    st.subheader("Race and Driver Details")
                    col1, col2 = st.columns(2)

                    # Use available unique values from the prepared data
                    driver_options = sorted(prepared_data['Driver'].unique())
                    race_options = sorted(prepared_data['raceName'].unique())
                    team_options = sorted(prepared_data['Team'].unique())

                    with col1:
                        driver = st.selectbox("Driver", driver_options, index=0, help="Select the driver.")
                        race_name = st.selectbox("Race", race_options, index=0, help="Select the Grand Prix.")
                        team = st.selectbox("Team", team_options, index=0, help="Select the team.")
                        grid = st.number_input("Grid Position", min_value=1, max_value=25, value=10, step=1, help="Driver's starting grid position.")

                    with col2:
                        # Use historical averages as defaults
                        default_rolling_avg = prepared_data[prepared_data['Driver'] == driver]['rolling_3race_avg'].mean() if driver in prepared_data['Driver'].values else prepared_data['rolling_3race_avg'].median()
                        default_track_avg = prepared_data[(prepared_data['Driver'] == driver) & (prepared_data['raceName'] == race_name)]['track_avg_position'].mean() if driver in prepared_data['Driver'].values and race_name in prepared_data['raceName'].values else prepared_data['track_avg_position'].median()
                        default_team_avg = prepared_data[prepared_data['Team'] == team]['team_rolling_avg'].mean() if team in prepared_data['Team'].values else prepared_data['team_rolling_avg'].median()

                        rolling_3race_avg = st.number_input("Est. 3-Race Avg Position", min_value=1.0, max_value=25.0, value=float(f"{default_rolling_avg:.1f}"), step=0.1, help="Driver's estimated recent avg position.")
                        track_avg_position = st.number_input("Est. Track Avg Position", min_value=1.0, max_value=25.0, value=float(f"{default_track_avg:.1f}"), step=0.1, help="Driver's typical performance at this track.")
                        season_progress = st.slider("Season Progress", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="0=start, 1=end.")
                        team_rolling_avg = st.number_input("Est. Team Avg Position", min_value=1.0, max_value=25.0, value=float(f"{default_team_avg:.1f}"), step=0.1, help="Team’s estimated recent performance.")
                        weather_temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=45.0, value=25.0, step=0.5, help="Expected race day temperature.")
                        weather_rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, help="Expected rainfall (0 for dry).")

                    submit = st.form_submit_button("Predict Finishing Position")

                    if submit:
                        # Create input DataFrame
                        input_data = pd.DataFrame([{
                            'Driver': driver,
                            'raceName': race_name, # Use 'raceName' consistently
                            'Team': team,
                            'grid': grid,
                            'rolling_3race_avg': rolling_3race_avg,
                            'track_avg_position': track_avg_position,
                            'season_progress': season_progress,
                            'team_rolling_avg': team_rolling_avg,
                            'weather_temp': weather_temp,
                            'weather_rainfall': weather_rainfall
                        }])

                        # Ensure data types match training data
                        for col in ['Driver', 'raceName', 'Team']:
                            input_data[col] = input_data[col].astype('string')
                        for col in ['grid', 'rolling_3race_avg', 'track_avg_position', 'season_progress',
                                    'team_rolling_avg', 'weather_temp', 'weather_rainfall']:
                             input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
                        input_data = input_data.fillna(prepared_data[input_data.columns].median(numeric_only=True))


                        st.markdown("---")
                        st.subheader("🔮 Prediction")
                        try:
                            # Transform input data using the FITTED preprocessor
                            input_transformed = preprocessor.transform(input_data)

                            # Make prediction
                            prediction = regressor.predict(input_transformed)[0]

                            # Display prediction
                            st.metric(label=f"Predicted Finish for {driver} ({team}) at {race_name}",
                                      value=f"P {round(prediction)}") # Round prediction
                            st.success(f"Prediction successful!")
                            st.info("Note: Based on historical data (2018-2023) and provided inputs.")

                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                            st.error("Check inputs and model training.")

            else:
                st.warning("Model or preprocessor could not be trained/loaded. Prediction unavailable.")

        # ============ Data Overview & EDA Section ============
        elif app_mode == "Data Overview & EDA":
            st.header("📊 Data Overview & Exploratory Data Analysis")
            st.markdown("Explore the prepared dataset used for training the model (2018-2023).")

            tab1, tab2, tab3 = st.tabs(["Sample Data", "Summary Statistics", "Correlations"])

            with tab1:
                st.subheader("Prepared Data Sample")
                st.dataframe(prepared_data.sample(min(100, len(prepared_data))))

            with tab2:
                st.subheader("Descriptive Statistics (Numeric Features)")
                numeric_cols_eda = prepared_data.select_dtypes(include=np.number).columns
                st.dataframe(prepared_data[numeric_cols_eda].describe())

            with tab3:
                st.subheader("Feature Correlation Heatmap (Numeric Features)")
                numeric_cols_corr = prepared_data.select_dtypes(include=np.number).columns
                if len(numeric_cols_corr) > 1:
                    corr = prepared_data[numeric_cols_corr].corr()
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
                    ax.set_title("Correlation Matrix of Numeric Features", pad=20)
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    st.pyplot(fig)
                else:
                    st.warning("Not enough numeric features to calculate correlations.")

        # ============ Model Performance Section ============
        elif app_mode == "Model Performance":
            st.header("⚙️ Model Performance Evaluation")
            st.markdown("Performance metrics of the trained XGBoost Regressor on the validation set.")

            if regressor and preprocessor and X_val is not None and y_val is not None and X_val_transformed is not None:
                try:
                    y_pred_val = regressor.predict(X_val_transformed)

                    mae = mean_absolute_error(y_val, y_pred_val)
                    mse = mean_squared_error(y_val, y_pred_val)
                    r2 = r2_score(y_val, y_pred_val)

                    st.subheader("Regression Metrics (Validation Set)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
                    col2.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
                    col3.metric("R-squared (R²)", f"{r2:.3f}")

                    st.markdown("---")
                    st.subheader("Feature Importances")

                    # Get feature names AFTER one-hot encoding
                    try:
                        feature_names_out = preprocessor.get_feature_names_out()
                    except AttributeError: # Handle older scikit-learn versions
                        # Manually construct names (less robust)
                        num_names = preprocessor.transformers_[0][2]
                        cat_encoder = preprocessor.transformers_[1][1]
                        cat_names = cat_encoder.get_feature_names_out(categorical_features)
                        feature_names_out = list(num_names) + list(cat_names)


                    importances = regressor.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': feature_names_out,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)

                    st.dataframe(importance_df.head(15)) # Display top 15 features

                    # Plot top features
                    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis', ax=ax_imp)
                    ax_imp.set_title('Top 15 Feature Importances')
                    st.pyplot(fig_imp)

                    st.markdown("---")
                    st.subheader("Residual Plot (Predicted vs. Actual)")
                    fig_res, ax_res = plt.subplots(figsize=(8, 6))
                    ax_res.scatter(y_val, y_pred_val, alpha=0.5)
                    ax_res.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2) # y=x line
                    ax_res.set_xlabel("Actual Position")
                    ax_res.set_ylabel("Predicted Position")
                    ax_res.set_title("Predicted vs. Actual Finishing Position")
                    ax_res.grid(True)
                    st.pyplot(fig_res)

                except Exception as e:
                    st.error(f"Error during model evaluation: {e}")
            else:
                st.warning("Model evaluation cannot be performed. Training might have failed or validation data is missing.")

    else:
        st.warning("Data preparation failed. Please check the data files and preparation logic.")
else:
    st.warning("Initial data loading failed. Please ensure all required CSV files are present in the specified directory.")


# Footer
st.markdown('<div class="footer">F1 Predictor | Streamlit Demo</div>', unsafe_allow_html=True)