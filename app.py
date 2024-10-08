import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import time
import pickle

class ETL:
    def __init__(self, ticker, test_size=0.2, period='max', n_input=5, timestep=5) -> None:
        self.ticker = ticker
        self.period = period
        self.test_size = test_size
        self.n_input = n_input
        self.df = self.extract_historic_data()
        self.timestep = timestep
        self.train, self.test = self.etl()
        self.X_train, self.y_train = self.to_supervised(self.train)
        self.X_test, self.y_test = self.to_supervised(self.test)

    def extract_historic_data(self) -> pd.Series:
        t = yf.Ticker(self.ticker)
        history = t.history(period=self.period)
        return history.Close

    def split_data(self) -> tuple:
        data = self.extract_historic_data()
        if len(data) != 0:
            train_idx = round(len(data) * (1-self.test_size))
            train = data[:train_idx]
            test = data[train_idx:]
            train = np.array(train)
            test = np.array(test)
            return train[:, np.newaxis], test[:, np.newaxis]
        else:
            raise Exception('Data set is empty, cannot split.')

    def window_and_reshape(self, data) -> np.array:
        NUM_FEATURES = 1
        samples = int(data.shape[0] / self.timestep)
        result = np.array(np.array_split(data, samples))
        return result.reshape((samples, self.timestep, NUM_FEATURES))

    def transform(self, train, test) -> np.array:
        train_remainder = train.shape[0] % self.timestep
        test_remainder = test.shape[0] % self.timestep
        if train_remainder != 0 and test_remainder != 0:
            train = train[train_remainder:]
            test = test[test_remainder:]
        elif train_remainder != 0:
            train = train[train_remainder:]
        elif test_remainder != 0:
            test = test[test_remainder:]
        return self.window_and_reshape(train), self.window_and_reshape(test)

    def etl(self) -> tuple[np.array, np.array]:
        train, test = self.split_data()
        return self.transform(train, test)

    def to_supervised(self, train, n_out=5) -> tuple:
        data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
        X, y = [], []
        in_start = 0
        for _ in range(len(data)):
            in_end = in_start + self.n_input
            out_end = in_end + n_out
            if out_end <= len(data):
                x_input = data[in_start:in_end, 0]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                y.append(data[in_end:out_end, 0])
                in_start += 1
        return np.array(X), np.array(y)

class PredictAndForecast:
    def __init__(self, model, train, test, n_input=5) -> None:
        self.model = model
        self.train = train
        self.test = test
        self.n_input = n_input
        self.predictions = self.get_predictions()

    def forecast(self, history) -> np.array:
        data = np.array(history)
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        input_x = data[-self.n_input:, :]
        input_x = input_x.reshape((1, len(input_x), input_x.shape[1]))
        yhat = self.model.predict(input_x, verbose=0)
        yhat = yhat[0]
        return yhat

    def get_predictions(self) -> np.array:
        history = [x for x in self.train]
        predictions = []
        for i in range(len(self.test)):
            yhat_sequence = self.forecast(history)
            predictions.append(yhat_sequence)
            history.append(self.test[i, :])
        return np.array(predictions)

class Evaluate:
    def __init__(self, actual, predictions) -> None:
        self.actual = actual
        self.predictions = predictions
        self.var_ratio = self.compare_var()
        self.mape = self.evaluate_model_with_mape()

    def compare_var(self):
        return abs(1 - (np.var(self.predictions) / np.var(self.actual)))

    def evaluate_model_with_mape(self):
        return mean_absolute_percentage_error(self.actual.flatten(), self.predictions.flatten())

def build_lstm(etl: ETL, epochs=25, batch_size=32) -> tf.keras.Model:
    n_timesteps, n_features, n_outputs = 5, 1, 5
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    history = model.fit(etl.X_train, etl.y_train, batch_size=batch_size, epochs=epochs, validation_data=(etl.X_test, etl.y_test), verbose=1, callbacks=callbacks)
    return model, history

def plot_results(test, preds, df, title_suffix=None, xlabel='Stock Price'):
    fig, ax = plt.subplots(figsize=(20,6))
    plot_test = test[1:]
    plot_preds = preds[1:]
    x = df[-(plot_test.shape[0]*plot_test.shape[1]):].index
    plot_test = plot_test.reshape((plot_test.shape[0]*plot_test.shape[1], 1))
    plot_preds = plot_preds.reshape((plot_test.shape[0]*plot_test.shape[1], 1))
    ax.plot(x, plot_test, label='actual')
    ax.plot(x, plot_preds, label='preds')
    ax.set_title('Predictions vs. Actual' if title_suffix is None else f'Predictions vs. Actual - {title_suffix}')
    ax.set_xlabel('Date')
    ax.set_ylabel(xlabel)
    ax.legend()
    st.pyplot(fig)

st.title('Stock Price Prediction with LSTM')
st.write('Select a stock ticker and train the model to predict future stock prices.')

tickers = ['META', 'NVDA', 'AAPL']
selected_ticker = st.selectbox('Select Stock Ticker', tickers)

if st.button('Train Model'):
    st.write(f'Training model for {selected_ticker}...')
    data = ETL(selected_ticker)
    model, history = build_lstm(data)
    baseline_preds = PredictAndForecast(model, data.train, data.test)
    baseline_evals = Evaluate(data.test, baseline_preds.predictions)
    st.write(f'MAPE: {baseline_evals.mape}')
    st.write(f'Variance Ratio: {baseline_evals.var_ratio}')
    st.session_state['model'] = model
    st.session_state['data'] = data
    st.session_state['predictions'] = baseline_preds.predictions

if 'model' in st.session_state:
    if st.button('Show Predictions'):
        plot_results(st.session_state['data'].test, st.session_state['predictions'], st.session_state['data'].df, title_suffix=selected_ticker)

    if st.button('Download Model'):
        model_filename = f'{selected_ticker}_lstm_model.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(st.session_state['model'], f)
        st.download_button('Download Trained Model', data=open(model_filename, 'rb'), file_name=model_filename)