# LSTM Stock Prediction

This project uses Long Short-Term Memory (LSTM) networks to predict stock prices. It features an ETL (Extract, Transform, Load) pipeline, model building, prediction, and evaluation functionalities, all wrapped in a Streamlit web application.

## Features

- *Data Extraction*: Fetch historical stock data using yfinance.
- *Data Transformation*: Split and reshape data for LSTM model.
- *Model Training*: Train LSTM model using TensorFlow and Keras.
- *Prediction*: Generate stock price predictions.
- *Evaluation*: Evaluate the model using MAPE and variance ratio.
- *Streamlit Interface*: User-friendly interface for selecting stocks, training models, and viewing predictions.

## Installation

1. Clone the repository:
    sh
    git clone https://github.com/DecX-x/LSTM_Stock-Prediction.git
    cd LSTM_Stock-Prediction
    

2. Install the required dependencies:
    sh
    pip install -r requirements.txt
    

## Usage

1. Run the Streamlit application:
    sh
    streamlit run app.py
    

2. Select a stock ticker from the dropdown menu or enter a custom ticker.

3. Click "Train Model" to start training the LSTM model.

4. View the predictions and evaluation metrics.

## File Overview

- *app.py*: Main application file containing the ETL process, LSTM model, prediction, evaluation, and Streamlit interface.

## Classes and Functions

### ETL Class

Handles the extraction, transformation, and loading of stock data.

- __init__(self, ticker, test_size=0.2, period='max', n_input=5, timestep=5)
- extract_historic_data(self)
- split_data(self)
- window_and_reshape(self, data)
- transform(self, train, test)
- etl(self)
- to_supervised(self, train, n_out=5)

### PredictAndForecast Class

Handles the prediction process using the trained LSTM model.

- __init__(self, model, train, test, n_input=5)
- forecast(self, history)
- get_predictions(self)

### Evaluate Class

Evaluates the performance of the prediction model.

- __init__(self, actual, predictions)
- compare_var(self)
- evaluate_model_with_mape(self)

### Functions

- build_lstm(etl: ETL, epochs=50, batch_size=32, progress_bar=None) -> tf.keras.Model
- plot_results(test, preds, df, title_suffix=None, xlabel='Stock Price')

---

Feel free to customize this README.md file further to better suit your project.
