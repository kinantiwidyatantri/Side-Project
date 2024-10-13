<h1>Stock Price Prediction Using LSTM Neural Networks</h1>

<p>This project focuses on predicting the stock prices of Apple Inc. (AAPL) using historical data and machine learning techniques. 
  We collected historical stock data from 2010 to 2023 with the <i>yfinance</i> library. 
  The data was enriched by calculating technical indicators such as Moving Averages (MA20, MA50), 
  Exponential Moving Average (EMA), and Relative Strength Index (RSI) to capture market trends.</p>

<p>The closing prices were scaled using <i>MinMaxScaler</i> to normalize the data for better model performance. 
    An LSTM (Long Short-Term Memory) neural network was constructed using TensorFlow's Keras API. 
    The model was trained to learn from the sequential data patterns and make future price predictions.</p>

<p>After training, the model's performance was evaluated using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). 
  A graph is included to visualize the model's predicted prices against the actual closing prices, demonstrating the model's accuracy and effectiveness.</p>

  <img width="1440" alt="Screenshot 2024-10-11 at 21 16 56" src="https://github.com/user-attachments/assets/91e82c44-225c-4b99-aeef-efb2d3416d1c">
