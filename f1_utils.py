# f1_utils.py
import pandas as pd
import numpy as np
import joblib
import streamlit as st # Import streamlit for caching decorators
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder # Ensure necessary imports
from sklearn.metrics import precision_recall_curve, auc
import warnings
import os # Import os to check file existence

warnings.filterwarnings('ignore')

# --- Constants ---
DATA_DIR = 'Formula 1 World Championship (1950 - 2024)'
WEATHER_DIR = 'F1-Weather'
PROCESSED_DATA_PATH = 'f1_data_enhanced.csv' # Using the most enhanced file from notebook
DRIVERS_PATH = f'{DATA_DIR}/drivers.csv'
MODEL_CLF_BEST_PATH = 'race_outcome_model.pkl' # Path for the best classifier saved
MODEL_CLF_RF_PATH = 'f1_winner_predictor_rf.pkl'
MODEL_CLF_XGB_PATH = 'f1_winner_predictor_xgb.pkl'
MODEL_REG_PATH = 'position_regressor_model.pkl' # Path for the position regressor
SCALER_PATH = 'feature_scaler.pkl' # Scaler for classifier
REGRESSOR_FEATURES_PATH = 'regressor_features.joblib'
REGRESSOR_SCALER_PATH = 'regressor_scaler.pkl' # Specific scaler for regressor
MODEL_FEATURES_PATH = 'model_features.joblib' # Features for classifier

# --- Data Loading ---
# @st.cache_data # Remove caching for now to ensure fresh loads during debugging
def load_processed_data(file_path=PROCESSED_DATA_PATH):
    """Loads the preprocessed and enhanced F1 data."""
    if not os.path.exists(file_path):
         st.error(f"Error: Data file '{file_path}' not found.")
         # Try loading f1_merged_data as fallback
         fallback_path = 'f1_merged_data.csv'
         if os.path.exists(fallback_path):
              st.warning(f"Trying fallback data file: '{fallback_path}'")
              file_path = fallback_path
         else:
              fallback_path_2 = 'f1_processed_final.csv'
              if os.path.exists(fallback_path_2):
                   st.warning(f"Trying fallback data file: '{fallback_path_2}'")
                   file_path = fallback_path_2
              else:
                   st.error("No suitable data file found.")
                   return None

    try:
        data = pd.read_csv(file_path, low_memory=False)
        print(f"Data loaded successfully from {file_path}")

        # --- Essential Cleaning/Type Conversion Post-Load ---
        if 'win' not in data.columns:
            if 'positionOrder' in data.columns:
                data['positionOrder'] = pd.to_numeric(data['positionOrder'], errors='coerce')
                data = data.dropna(subset=['positionOrder'])
                data['positionOrder'] = data['positionOrder'].astype(int)
                data['win'] = (data['positionOrder'] == 1).astype(int)
                print("Created 'win' column.")
            else:
                st.error("Cannot determine race winners: 'positionOrder' column missing.")
                return None

        numeric_cols_to_convert = [
            'grid', 'positionOrder', 'points', 'laps', 'milliseconds', 'rank',
            'fastestLap', 'fastestLapSpeed', 'pit_stop_count', 'total_pit_time',
            'Quali_Race_Delta', 'Avg_Stint_Length', 'Track_Difficulty',
            'Weather_Impact', 'Driver_Consistency', 'Constructor_Form',
            'temperature', 'rainfall', 'is_wet'
        ]
        for col in numeric_cols_to_convert:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        if 'date' in data.columns:
             try: data['date'] = pd.to_datetime(data['date']); print("Converted 'date' column to datetime.")
             except Exception as e: st.warning(f"Could not convert 'date': {e}. Dropping."); data = data.drop(columns=['date'])

        numeric_cols_in_data = data.select_dtypes(include=np.number).columns
        for col in numeric_cols_in_data:
             if data[col].isnull().any(): data[col] = data[col].fillna(data[col].median())

        data = data.dropna(subset=['win'])
        return data

    except Exception as e:
        st.error(f"Error loading or processing data from {file_path}: {e}")
        return None

# @st.cache_data # Remove caching for now
def load_drivers_data(file_path=DRIVERS_PATH):
     """Loads drivers data and creates a 'name' column."""
     if not os.path.exists(file_path):
          st.error(f"Error: Drivers file '{file_path}' not found.")
          return None
     try:
          drivers = pd.read_csv(file_path)
          drivers['name'] = drivers['forename'] + ' ' + drivers['surname']
          print(f"Drivers data loaded successfully from {file_path}")
          return drivers
     except Exception as e:
        st.error(f"Error loading drivers data: {e}")
        return None

# --- Model & Scaler Loading ---
# @st.cache_resource # Remove caching for now
def load_model(model_path=MODEL_CLF_BEST_PATH, fallback1=MODEL_CLF_RF_PATH, fallback2=MODEL_CLF_XGB_PATH):
    """Loads the best trained classification model, with fallbacks."""
    paths_to_try = [model_path, fallback1, fallback2]
    loaded_model = None
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                print(f"Classifier model loaded successfully from {path}")
                loaded_model = model
                # Attempt to save feature names if needed
                if hasattr(model, 'feature_names_in_') and not os.path.exists(MODEL_FEATURES_PATH):
                     try:
                          joblib.dump(list(model.feature_names_in_), MODEL_FEATURES_PATH)
                          print(f"Saved classifier features to {MODEL_FEATURES_PATH}")
                     except Exception as e: st.warning(f"Could not save classifier features: {e}")
                break
            except Exception as e: st.error(f"Error loading model from {path}: {e}")
        else: print(f"Model file not found at {path}. Trying next...")
    if loaded_model is None: st.error("Error: No valid classification model file found.")
    return loaded_model

# @st.cache_resource # Remove caching for now
def load_regressor_model(model_path=MODEL_REG_PATH):
    """Loads the trained position regressor model."""
    if not os.path.exists(model_path):
        st.warning(f"Regressor model file '{model_path}' not found. Position prediction unavailable.")
        return None
    try:
        model = joblib.load(model_path)
        print(f"Regressor model loaded successfully from {model_path}")
        # Save features if needed
        if hasattr(model, 'feature_names_in_') and not os.path.exists(REGRESSOR_FEATURES_PATH):
             try:
                 joblib.dump(list(model.feature_names_in_), REGRESSOR_FEATURES_PATH)
                 print(f"Saved regressor features to {REGRESSOR_FEATURES_PATH}")
             except Exception as e: st.warning(f"Could not save regressor features: {e}")
        return model
    except Exception as e: st.error(f"Error loading regressor model: {e}"); return None

# @st.cache_resource # Remove caching for now
def load_scaler(scaler_path=SCALER_PATH):
    """Loads the feature scaler for the classifier."""
    if not os.path.exists(scaler_path):
        st.error(f"Error: Classifier scaler file '{scaler_path}' not found.")
        return None
    try:
        scaler = joblib.load(scaler_path)
        print(f"Classifier scaler loaded successfully from {scaler_path}")
        return scaler
    except Exception as e: st.error(f"Error loading classifier scaler: {e}"); return None

# @st.cache_resource # Remove caching for now
def load_regressor_scaler(scaler_path=REGRESSOR_SCALER_PATH):
    """Loads the feature scaler specifically for the regressor."""
    if not os.path.exists(scaler_path):
        st.error(f"Error: Regressor scaler file '{scaler_path}' not found.")
        return None
    try:
        scaler = joblib.load(scaler_path)
        print(f"Regressor scaler loaded successfully from {scaler_path}")
        return scaler
    except Exception as e: st.error(f"Error loading regressor scaler: {e}"); return None

# @st.cache_data # Remove caching for now
def load_model_features(features_path=MODEL_FEATURES_PATH):
     """Loads the list of feature names the classifier was trained on."""
     if not os.path.exists(features_path):
         st.error(f"Error: Classifier features file '{features_path}' not found.")
         return None
     try:
          features = joblib.load(features_path)
          print(f"Classifier features loaded successfully from {features_path}")
          return list(features)
     except Exception as e: st.error(f"Error loading classifier features: {e}"); return None

# @st.cache_data # Remove caching for now
def load_regressor_features(features_path=REGRESSOR_FEATURES_PATH):
     """Loads the list of feature names the regressor model was trained on."""
     if not os.path.exists(features_path):
         st.error(f"Error: Regressor features file '{features_path}' not found.")
         return None
     try:
          features = joblib.load(features_path)
          print(f"Regressor features loaded successfully from {features_path}")
          return list(features)
     except Exception as e: st.error(f"Error loading regressor features: {e}"); return None


# --- Prediction Input Preparation ---
def prepare_prediction_input(input_dict, historical_data, target_features, scaler):
    """Prepares a single row DataFrame for prediction for a specific model."""
    if scaler is None or target_features is None:
        st.error("Scaler or target features list not provided for prediction preparation.")
        return None
    if not hasattr(scaler, 'n_features_in_'): # Check for attribute indicating it's fitted
        st.error("Scaler object provided is not fitted. Fit it before prediction.")
        return None

    pred_df = pd.DataFrame([input_dict])

    # 1. Ensure all target_features columns exist, add if missing
    for feature in target_features:
        if feature not in pred_df.columns:
            if feature in historical_data.columns:
                if pd.api.types.is_numeric_dtype(historical_data[feature]): default_val = historical_data[feature].median()
                else: default_val = historical_data[feature].mode()[0] if not historical_data[feature].mode().empty else ''
                pred_df[feature] = default_val
            else: pred_df[feature] = 0; st.warning(f"Feature '{feature}' missing, added default 0.")

    # 2. Select only target_features in the correct order
    try: pred_df = pred_df[target_features]
    except KeyError as e: st.error(f"KeyError selecting target features. Missing: {e}."); return None

    # 3. Handle Categorical Features (DROPPING)
    categorical_cols_target = [col for col in target_features if col in pred_df.select_dtypes(include=['object', 'category']).columns]
    if categorical_cols_target:
        pred_df = pred_df.drop(columns=categorical_cols_target)
        target_features_numeric = [f for f in target_features if f not in categorical_cols_target]
    else: target_features_numeric = target_features

    # 4. Scale Numeric Features - CRITICAL: Use scaler.get_feature_names_out() if available, or ensure consistency
    try:
        # Get feature names the scaler *was fitted on*
        try:
            # Scikit-learn >= 1.0
            scaler_features_fitted = list(scaler.get_feature_names_out())
        except AttributeError:
             # Scikit-learn < 1.0 - Requires manual saving/loading of fitted columns
             # For now, assume the scaler was fitted on *all* numeric cols in target_features_numeric
             # This is less safe but a common pattern if names weren't saved
             scaler_features_fitted = [col for col in target_features_numeric if pd.api.types.is_numeric_dtype(pred_df[col])]
             if not scaler_features_fitted:
                  st.error("Cannot determine scaler features and feature_names_in_ not available.")
                  return None
             st.warning("Scaler feature names not available directly, inferring from input numeric columns.")


        # Identify numeric columns that are BOTH in the current DataFrame AND were seen by the scaler during fit
        cols_to_scale = [col for col in target_features_numeric if col in scaler_features_fitted and col in pred_df.columns]

        if not cols_to_scale: st.warning("No matching numeric columns found to scale.")
        else:
             numeric_data_pred = pred_df[cols_to_scale].copy()
             if numeric_data_pred.isnull().any().any(): numeric_data_pred = numeric_data_pred.fillna(numeric_data_pred.median())

             # Check if DataFrame is empty before scaling
             if numeric_data_pred.empty:
                  st.warning("Numeric data for scaling is empty.")
             else:
                  scaled_values = scaler.transform(numeric_data_pred)
                  pred_df[cols_to_scale] = scaled_values

    except Exception as e: st.error(f"Error during feature scaling: {e}"); return None

    # Ensure the final DataFrame has the columns expected by the *model* (after dropping categoricals)
    final_model_features = [f for f in target_features if f in pred_df.columns]
    return pred_df[final_model_features]


def predict_driver_outcome(model_clf, model_reg, scaler_clf, scaler_reg,
                           historical_data, drivers_df,
                           future_race_inputs, clf_features, reg_features):
    """Predicts win probability and position for all active drivers."""
    # Simplified checks - assumes variables are not None if called
    if model_clf is None or scaler_clf is None or clf_features is None:
         st.error("Classifier components missing."); return pd.DataFrame()
    if model_reg is None or scaler_reg is None or reg_features is None:
         st.warning("Regressor components missing. Position prediction disabled.")
         model_reg = None # Ensure it's None if components missing


    predictions = []
    latest_year = historical_data['year'].max()
    active_drivers = historical_data[historical_data['year'] == latest_year]['driverId'].unique()
    if len(active_drivers) == 0: active_drivers = historical_data['driverId'].unique()

    for driver_id in active_drivers:
        driver_hist = historical_data[historical_data['driverId'] == driver_id].tail(5)
        input_dict_base = future_race_inputs.copy(); input_dict_base['driverId'] = driver_id

        # --- Estimate Driver/Team Specific Features ---
        features_to_estimate = {
            'Quali_Race_Delta': driver_hist['Quali_Race_Delta'].median() if 'Quali_Race_Delta' in driver_hist and not driver_hist['Quali_Race_Delta'].isnull().all() else historical_data['Quali_Race_Delta'].median(),
            'Driver_Consistency': driver_hist['Driver_Consistency'].median() if 'Driver_Consistency' in driver_hist and not driver_hist['Driver_Consistency'].isnull().all() else historical_data['Driver_Consistency'].median(),
            'Constructor_Form': driver_hist['Constructor_Form'].median() if 'Constructor_Form' in driver_hist and not driver_hist['Constructor_Form'].isnull().all() else historical_data['Constructor_Form'].median(),
            'Avg_Stint_Length': driver_hist['Avg_Stint_Length'].median() if 'Avg_Stint_Length' in driver_hist and not driver_hist['Avg_Stint_Length'].isnull().all() else historical_data['Avg_Stint_Length'].median(),
            'pit_stop_count': driver_hist['pit_stop_count'].median() if 'pit_stop_count' in driver_hist and not driver_hist['pit_stop_count'].isnull().all() else historical_data['pit_stop_count'].median(),
            'total_pit_time': driver_hist['total_pit_time'].median() if 'total_pit_time' in driver_hist and not driver_hist['total_pit_time'].isnull().all() else historical_data['total_pit_time'].median(),
            'points': driver_hist['points'].median() if 'points' in driver_hist and not driver_hist['points'].isnull().all() else historical_data['points'].median(),
            'rank': driver_hist['rank'].median() if 'rank' in driver_hist and not driver_hist['rank'].isnull().all() else historical_data['rank'].median(),
            'laps': driver_hist['laps'].median() if 'laps' in driver_hist and not driver_hist['laps'].isnull().all() else historical_data['laps'].median(),
            'milliseconds': driver_hist['milliseconds'].median() if 'milliseconds' in driver_hist and not driver_hist['milliseconds'].isnull().all() else historical_data['milliseconds'].median(),
            'constructorId': historical_data.loc[historical_data['driverId'] == driver_id, 'constructorId'].iloc[-1] if not driver_hist.empty and 'constructorId' in driver_hist else 0,
        }

        # --- Prepare for Classifier ---
        input_dict_clf = input_dict_base.copy()
        for feat, value in features_to_estimate.items():
            if feat in clf_features: input_dict_clf[feat] = value
        # Ensure required base features are present
        for feat in future_race_inputs:
             if feat in clf_features and feat not in input_dict_clf: input_dict_clf[feat] = future_race_inputs[feat]
        input_df_clf = prepare_prediction_input(input_dict_clf, historical_data, clf_features, scaler_clf)

        win_prob = np.nan
        if input_df_clf is not None:
            try: win_prob = model_clf.predict_proba(input_df_clf)[:, 1][0]
            except Exception as e: st.warning(f"Clf prediction error driver {driver_id}: {e}")

        # --- Prepare for Regressor ---
        predicted_pos = np.nan
        if model_reg is not None and reg_features is not None and scaler_reg is not None:
             input_dict_reg = input_dict_base.copy()
             for feat, value in features_to_estimate.items():
                 if feat in reg_features: input_dict_reg[feat] = value
             # Ensure required base features are present
             for feat in future_race_inputs:
                 if feat in reg_features and feat not in input_dict_reg: input_dict_reg[feat] = future_race_inputs[feat]
             input_df_reg = prepare_prediction_input(input_dict_reg, historical_data, reg_features, scaler_reg)

             if input_df_reg is not None:
                 try: predicted_pos = model_reg.predict(input_df_reg)[0]
                 except Exception as e: st.warning(f"Reg prediction error driver {driver_id}: {e}")

        predictions.append({'driverId': driver_id, 'win_probability': win_prob, 'predicted_position': round(predicted_pos) if pd.notna(predicted_pos) else np.nan})

    # --- Final Processing ---
    if not predictions: st.warning("No predictions generated."); return pd.DataFrame()
    results_df = pd.DataFrame(predictions).dropna(subset=['win_probability']) # Drop rows where win prob failed
    if results_df.empty: st.warning("No valid win predictions."); return pd.DataFrame()
    results_df = pd.merge(results_df, drivers_df[['driverId', 'name']], on='driverId', how='left')
    results_df['name'] = results_df['name'].fillna('ID: ' + results_df['driverId'].astype(str))
    results_df = results_df.sort_values(['win_probability', 'predicted_position'], ascending=[False, True]).reset_index(drop=True)
    return results_df


# --- Plotting Functions (Keep as before) ---
def plot_correlation(data, valid_cols):
    if len(valid_cols) < 2: st.warning("Not enough valid columns for correlation plot."); return None
    numeric_data = data[valid_cols].select_dtypes(include=np.number); numeric_data = numeric_data.fillna(numeric_data.median())
    if numeric_data.shape[1] < 2: st.warning("Not enough numeric columns after filtering/filling."); return None
    fig, ax = plt.subplots(figsize=(12, 10)); sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title("Feature Correlation Matrix", pad=20, fontsize=16); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout(); return fig

def plot_qualifying_vs_race(data):
    req_cols = ['grid', 'positionOrder', 'win'];
    if not all(col in data.columns for col in req_cols): st.warning(f"Missing columns for Quali vs Race plot: {req_cols}"); return None
    plot_data = data.copy();
    for col in ['grid', 'positionOrder']: plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
    plot_data = plot_data.dropna(subset=['grid', 'positionOrder', 'win']); plot_data['win'] = plot_data['win'].astype(int)
    if plot_data.empty: st.warning("No valid data for Quali vs Race plot."); return None
    if len(plot_data) > 5000: plot_data = plot_data.sample(5000, random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6)); sns.scatterplot(x='grid', y='positionOrder', hue='win', alpha=0.6, palette={0:'cornflowerblue', 1:'salmon'}, data=plot_data, ax=ax)
    max_pos_g = plot_data['grid'].max(); max_pos_p = plot_data['positionOrder'].max()
    if pd.notna(max_pos_g) and pd.notna(max_pos_p): ax.plot([1, max(max_pos_g, max_pos_p)], [1, max(max_pos_g, max_pos_p)], 'k--', alpha=0.7)
    ax.set_xlabel("Qualifying Position (Grid)"); ax.set_ylabel("Race Finish Position"); ax.set_title("Qualifying vs Race Performance (Win=Salmon)")
    try: handles, labels = ax.get_legend_handles_labels(); ax.legend(handles=handles, labels=['No Win', 'Win'], title='Win')
    except Exception: ax.legend(title='Win') # Fallback
    plt.tight_layout(); return fig

def plot_pr_curve(y_true, y_proba, label, ax):
    if len(np.unique(y_true)) < 2: st.warning(f"Cannot plot PR curve for {label}: Only one class present."); return
    precision, recall, _ = precision_recall_curve(y_true, y_proba); pr_auc = auc(recall, precision)
    ax.plot(recall, precision, marker='.', markersize=3, label=f'{label} (AUC = {pr_auc:.2f})')