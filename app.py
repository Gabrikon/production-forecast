from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import joblib
import traceback

# Initialize Flask app
app = Flask(__name__)

# --- File Paths ---
DATA_PATH = 'filled_oil.csv'
MODELS_PATH = 'final_models.joblib'
FEATURE_SCALER_PATH = 'feature_scaler.joblib'
TARGET_SCALERS_PATH = 'target_scalers.joblib'

# --- Load Data, Models, and Scalers ---
try:
    df = pd.read_csv(DATA_PATH, parse_dates=['DATE'])
    df = df.set_index('DATE').sort_index()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).clip(lower=0)

    # Load the trained models and scalers
    final_models = joblib.load(MODELS_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scalers = joblib.load(TARGET_SCALERS_PATH)

    print("Data, models, and scalers loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading required files: {e}")
    print("Please ensure 'filled_oil.csv', 'final_models.joblib', 'feature_scaler.joblib', and 'target_scalers.joblib' are in the same directory as app.py")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    exit()

# --- Feature Engineering Functions ---
def create_features(df):
    """Creates time-based and rolling window features."""
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    col_mapping = {
        'NET (bbls/d)': 'net',
        'GROSS (bbls/d)': 'gross',
        'WATER (bbls/d)': 'water',
        'BS.W%': 'bsw'
    }
    # Create rolling windows for key columns
    for orig_col, col_std in col_mapping.items():
        # Use .copy() to avoid SettingWithCopyWarning
        #col_std = col.split(" ")[0].lower()
        df[f'{col_std}_rolling_avg7'] = df[orig_col].rolling(7).mean().copy()
        df[f'{col_std}_rolling_avg30'] = df[orig_col].rolling(30).mean().copy()
        df[f'{col_std}_rolling_avg90'] = df[orig_col].rolling(90).mean().copy()
        df[f'{col_std}_rolling_std7'] = df[orig_col].rolling(7).std().copy()
        df[f'{col_std}_rolling_std30'] = df[orig_col].rolling(30).std().copy()
        df[f'{col_std}_rolling_std90'] = df[orig_col].rolling(90).std().copy()

    # Create lag features
    for col in ['GROSS (bbls/d)', 'BS.W%', 'NET (bbls/d)', 'WATER (bbls/d)']:
        for lag in [1, 3, 7]:
            # Use .copy() to avoid SettingWithCopyWarning
            df[f'{col}_lag{lag}'] = df[col].shift(lag).copy()

    return df

# Apply feature engineering to get the expected feature names
df_features_engineered = create_features(df.copy())
df_features_engineered = df_features_engineered.dropna()  # Drop NaNs

# Define features and targets based on the processed historical data
targets = ['NET (bbls/d)', 'WATER (bbls/d)']
features = [col for col in df_features_engineered.columns if col not in targets]
print(f"Features used for training: {features}")

def create_future_features_with_bootstrap(last_date, periods, historical_df, features, n_samples=100):
    """
    Creates future features using bootstrapping from recent historical data.
    """
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    future_df = pd.DataFrame(index=future_dates)

    # Basic time features
    future_df['day_of_week'] = future_df.index.dayofweek
    future_df['day_of_month'] = future_df.index.day
    future_df['month'] = future_df.index.month
    future_df['year'] = future_df.index.year

    # Mapping between original columns and their standardized names
    col_mapping = {
        'NET (bbls/d)': 'net',
        'GROSS (bbls/d)': 'gross',
        'WATER (bbls/d)': 'water',
        'BS.W%': 'bsw'
    }

    # Create bootstrap samples for each target column
    for orig_col, std_col in col_mapping.items():
        # Get recent data (last 90 days for better bootstrap)
        recent_data = historical_df[orig_col].values[-90:]

        # Generate bootstrap samples
        bootstrap_samples = np.random.choice(recent_data, size=(periods, n_samples), replace=True)

        # Calculate statistics from bootstrap samples
        future_df[f'{std_col}_mean'] = bootstrap_samples.mean(axis=1)
        future_df[f'{std_col}_std'] = bootstrap_samples.std(axis=1)
        future_df[f'{std_col}_median'] = np.median(bootstrap_samples, axis=1)

        # Store both versions - original and standardized
        future_df[orig_col] = future_df[f'{std_col}_median']  # Original name for lag features
        future_df[std_col] = future_df[f'{std_col}_median']   # Standardized name for rolling features

    # Create rolling features using standardized names
    window_sizes = [7, 30, 90]
    for std_col in col_mapping.values():
        for window in window_sizes:
            # Rolling averages
            future_df[f'{std_col}_rolling_avg{window}'] = future_df[std_col].rolling(window).mean()
            # Rolling std
            future_df[f'{std_col}_rolling_std{window}'] = future_df[std_col].rolling(window).std()

        # Forward fill the initial NaN values using ffill()
        for window in window_sizes:
            future_df[f'{std_col}_rolling_avg{window}'] = future_df[f'{std_col}_rolling_avg{window}'].ffill()
            future_df[f'{std_col}_rolling_std{window}'] = future_df[f'{std_col}_rolling_std{window}'].ffill()

    # Create lag features using original column names
    for orig_col in col_mapping.keys():
        for lag in [1, 3, 7]:
            future_df[f'{orig_col}_lag{lag}'] = future_df[orig_col].shift(lag)
            # Fill initial NaN values with the first available value using ffill()
            future_df[f'{orig_col}_lag{lag}'] = future_df[f'{orig_col}_lag{lag}'].ffill()

    # Ensure all required feature columns are present
    for feature in features:
        if feature not in future_df.columns:
            print(f"Adding missing feature: {feature}")
            future_df[feature] = np.nan  # Add missing columns as NaN

    # Forward fill any remaining NaN values
    future_df = future_df.ffill().bfill()  # Use bfill() as well in case ffill() doesn't cover everything

    return future_df

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main page with input for forecast days."""
    return render_template('updated_index_html.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    """Generates and returns the forecast data for both targets as JSON for Chart.js."""
    try:
        print("Received forecast request.")
        
        # Get forecast_periods from form data
        forecast_periods = request.form.get('forecast_periods')
        print(f"Raw forecast_periods value: {forecast_periods}")
        
        if not forecast_periods:
            print("Error: forecast_periods parameter is missing.")
            return jsonify({"error": "Missing forecast periods parameter."}), 400
            
        # Convert to integer
        days = int(forecast_periods)
        print(f"Requested forecast days: {days}")

        if days <= 0:
            print("Error: Number of forecast days must be positive.")
            return jsonify({"error": "Number of forecast days must be positive."}), 400

        # Get the last date from the historical data
        last_date = df.index[-1]
        print(f"Last historical date: {last_date}")

        print("Creating future features...")
        # Generate future features using bootstrapping
        future_df = create_future_features_with_bootstrap(last_date, days, df, features)
        print("Future features created.")
        
        # Debug output - check if all required features are present
        missing_cols = set(features) - set(future_df.columns)
        if missing_cols:
            print(f"WARNING: Missing columns after feature creation: {missing_cols}")
            # Add missing columns with zero values
            for col in missing_cols:
                future_df[col] = 0

        print("Scaling future features...")
        # Make sure all columns are in the same order as features list
        future_df_ordered = future_df[features]
        # Scale future features
        future_df_scaled = feature_scaler.transform(future_df_ordered)
        future_df_scaled = pd.DataFrame(future_df_scaled, columns=features, index=future_df.index)
        print("Future features scaled.")

        print("Making predictions and inverse transforming...")
        # Make predictions and inverse transform for both targets
        forecasts = {}
        for target in targets:
            print(f"Predicting for target: {target}")
            preds_scaled = final_models[target].predict(future_df_scaled)
            preds_original = target_scalers[target].inverse_transform(
                preds_scaled.reshape(-1, 1)).flatten()
            forecasts[target] = pd.Series(preds_original, index=future_df.index)
            print(f"Prediction for {target} complete.")

        print("Preparing data for Chart.js...")
        # Prepare data for Chart.js - include historical data for context
        combined_net = pd.concat([df['NET (bbls/d)'], forecasts['NET (bbls/d)']])
        combined_water = pd.concat([df['WATER (bbls/d)'], forecasts['WATER (bbls/d)']])
        
        # Sort by date and remove duplicates (keep the first occurrence)
        combined_net = combined_net.sort_index().groupby(level=0).first()
        combined_water = combined_water.sort_index().groupby(level=0).first()
        print(combined_net.index)
        # Format for Chart.js
        chart_data = {
            'labels': combined_net.index.strftime('%Y-%m-%d').tolist(),
            'net_production': combined_net.values.tolist(),
            'water_production': combined_water.values.tolist()
        }
        print("Data prepared for Chart.js.")

        # Return the data as JSON
        return jsonify(chart_data)

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return jsonify({"error": "Invalid number of forecast days. Please enter a whole number."}), 400
    except Exception as e:
        print(f"An error occurred during forecasting: {e}")
        traceback.print_exc()  # Print detailed traceback
        return jsonify({"error": f"An error occurred during forecasting: {str(e)}"}), 500

if __name__ == '__main__':
    try:
        app.run(
            host='0.0.0.0',  # Explicitly set host
            port=5001,       # Use alternative port
            debug=True,      # Debug mode
            use_reloader=False,  # Disable reloader
            use_debugger=True
        )
    except Exception as e:
        print(f"Error starting Flask app: {e}")
