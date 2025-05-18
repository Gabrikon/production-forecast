# Oil and Water Production Forecasting Web Application

This project is a simple Flask web application that provides historical visualization and future forecasts for oil (NET bbls/d) and water (WATER bbls/d) production based on a pre-trained XGBoost model.

Users can input the number of future days they want to forecast, and the application will display interactive charts showing both historical data and the generated forecast.

## Features

* Loads historical production data from a CSV file.
* Loads pre-trained XGBoost models and scalers.
* Generates future production forecasts using bootstrapping and the loaded models.
* Displays historical data and future forecasts using interactive Chart.js plots.
* Allows users to specify the number of days for the forecast.

## Prerequisites

Before running this application, ensure you have the following installed:

* Python 3.6 or higher
* `pip` (Python package installer)

You will also need the following files in the root directory of the project:

* `filled_oil.csv`: The historical production data.
* `final_models.joblib`: The saved XGBoost models for 'NET (bbls/d)' and 'WATER (bbls/d)'.
* `feature_scaler.joblib`: The saved scaler for the features.
* `target_scalers.joblib`: The saved scaler for the targets ('NET (bbls/d)' and 'WATER (bbls/d)').

## Project Structure

.
├── app.py              # Flask application code
├── filled_oil.csv      # Historical data file
├── final_models.joblib # Saved models file
├── feature_scaler.joblib # Saved feature scaler file
├── target_scalers.joblib # Saved target scalers file
├── requirements.txt    # Python dependencies
├── Procfile            # (Optional, for deployment) Process file for hosting platforms
└── templates/
    └── index.html      # HTML template for the web interface

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a virtual environment:**
    It's recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install dependencies:**
    Install the required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Place required files:**
    Ensure `filled_oil.csv`, `final_models.joblib`, `feature_scaler.joblib`, and `target_scalers.joblib` are in the root directory of the project (the same directory as `app.py`).

## Running the Application

1.  **Activate your virtual environment** (if you haven't already).

2.  **Run the Flask application:**
    ```bash
    python app.py
    ```
    The application will start a development server, typically at `http://127.0.0.1:5000/`.

3.  **Open in browser:**
    Open your web browser and navigate to the address shown in the terminal (usually `http://127.0.0.1:5000/`).

## Using the Web Interface

Once the application is running and you access it in your browser:

1.  You will see an input field labeled "Number of Periods to Forecast:".
2.  Enter the desired number of future days for the forecast.
3.  Click the "Get Forecast" button.
4.  The application will generate the forecast and display two charts below: one for NET Oil Production and one for WATER Production, showing both historical data and the generated forecast.

