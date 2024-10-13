import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Data Collection
ticker_symbol = 'AAPL'  # Apple Inc.
data = yf.download(ticker_symbol, start='2010-01-01', end='2023-10-11')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Moving Averages
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# Exponential Moving Average
data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()

# Relative Strength Index (RSI)
delta = data['Close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
window_length = 14
RS = up.rolling(window_length).mean() / down.rolling(window_length).mean()
RSI = 100 - 100 / (1 + RS)
data['RSI'] = RSI

# Prepare the dataset
close_prices = data['Close'].values.reshape(-1, 1)
scaled_close = scaler.fit_transform(close_prices)

# Split into training and testing data
training_data_len = int(np.ceil(len(scaled_close) * 0.8))

# Create the training data
train_data = scaled_close[0:training_data_len, :]

# Create the test data
test_data = scaled_close[training_data_len - 60:, :]

# Function to create datasets
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

# Create the datasets
X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Reshape the data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10)

# Get the predicted values
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Get the actual values
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)

print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')

# Create a DataFrame for plotting
train = data[:training_data_len]
valid = data[training_data_len:]
valid = valid.copy()
valid['Predictions'] = predictions

# Plot the data
plt.figure(figsize=(14,7))
plt.title('LSTM Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'], label='Training Data')
plt.plot(valid['Close'], label='Actual Price')
plt.plot(valid['Predictions'], label='Predicted Price')
plt.legend()
plt.show()

# Model summary
#model.summary()

# Check the data structure
#print(data.head())
#print(data.info())
#print(data.describe())
#print(data.isnull().sum())


# Plot the closing price history
"""plt.figure(figsize=(14,7))
plt.plot(data['Close'], label='Close Price')
plt.title(f'{ticker_symbol} Closing Price History')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()
"""

