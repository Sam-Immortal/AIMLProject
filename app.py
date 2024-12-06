from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

# Load and preprocess data globally
DATA_FILE = "templates/Wheat_Bengaluru.csv"

def preprocess_data(file_path):
    """
    Preprocesses the dataset for SARIMA modeling.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.Series: A time series of monthly average modal prices.
    """
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Ensure required columns are present
    required_columns = ['Price Date', 'Modal Price (Rs./Quintal)']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"The dataset must contain the column '{col}'.")

    # Convert the 'Price Date' column to datetime format
    data['Price Date'] = pd.to_datetime(data['Price Date'], errors='coerce')
    data = data.dropna(subset=['Price Date'])  # Drop rows with invalid dates

    # Convert the 'Modal Price (Rs./Quintal)' column to numeric format
    data['Modal Price (Rs./Quintal)'] = pd.to_numeric(data['Modal Price (Rs./Quintal)'], errors='coerce')
    data = data.dropna(subset=['Modal Price (Rs./Quintal)'])  # Drop rows with invalid prices

    # Set 'Price Date' as the index
    data.set_index('Price Date', inplace=True)

    # Resample to monthly frequency, calculating the mean for each month
    monthly_data = data['Modal Price (Rs./Quintal)'].resample('M').mean()

    # Fill any remaining missing values with linear interpolation
    monthly_data = monthly_data.interpolate(method='linear')

    return monthly_data

# Train SARIMA model and make predictions
def predict_prices(data, periods=12):
    """
    Trains a SARIMA model and forecasts future prices.

    Parameters:
        data (pd.Series): Time series data for training the model.
        periods (int): Number of future periods to forecast.

    Returns:
        pd.Series: Forecasted values.
    """
    try:
        model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=periods)
        return forecast
    except Exception as e:
        print(f"Error in SARIMA model training: {e}")
        return pd.Series(dtype=float)

@app.route('/')
def start():
    """
    Default route to render the start.html page.
    """
    return render_template('start.html')

# Route for predictions
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Preprocess data
        processed_data = preprocess_data(DATA_FILE)

        # Generate predictions
        forecast = predict_prices(processed_data)
        forecast_dates = pd.date_range(start=processed_data.index[-1], periods=13, freq="M")[1:]
        predictions = {
            "dates": [date.strftime("%B %Y") for date in forecast_dates],
            "prices": [round(price, 2) for price in forecast]
        }
        return jsonify(predictions)
    except Exception as e:
        print(f"Error in prediction route: {e}")
        return jsonify({"error": str(e)})

@app.route('/login')
def login():
    """
    Route to render the login.html page.
    """
    return render_template('login.html')

@app.route('/home')
def home():
    """
    Route to render the main index.html page.
    """
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)