# pip install yfinance pandas numpy scikit-learn matplotlib tensorflow

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 📌 Step 1: User Input
stock_symbol = input("Enter stock symbol (e.g., AAPL, MSFT): ")
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")

# 📌 Step 2: Fetch Data
data = yf.download(stock_symbol, start=start_date, end=end_date)
if data.empty:
    print("No data found. Check your symbol and date range.")
    exit()

# 📌 Step 3: Use only 'Close' price
data = data[['Close']]

# 📌 Step 4: Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create features and labels
def create_dataset(dataset, time_step=10):
    X, y = [], []
    for i in range(len(dataset)-time_step):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)

# Reshape for FFNN (not 3D like LSTM)
X_train = X
y_train = y

# 📌 Step 5: Build Feed Forward Neural Network
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 📌 Step 6: Train Model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# 📌 Step 7: Predict and Plot
predicted = model.predict(X_train)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_train.reshape(-1, 1))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(real_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
