import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Download Stock Data
# For example, let's use Apple (AAPL) stock data
data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')

# Step 2: Preprocess the Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Step 3: Create Sequences
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_sequences(scaled_data, time_step)

# Reshape X for LSTM input
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 4: Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the Model
early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
model.fit(X, y, epochs=10, batch_size=32, callbacks=[early_stop])

# Step 6: Save the Model
model.save('lstm_stock_model.h5')
