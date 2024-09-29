#arjun tyagi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Set the start date and current end date
start = '2010-01-01'
end = datetime.now().strftime('%Y-%m-%d')  # Get today's date

st.title('Stock Price Forecasting with LSTM Model (AIBF project)')
user_input = st.text_input('Enter stock name (ticker)', 'TATASTEEL.NS')

# Download stock data
df = pd.DataFrame(yf.download(user_input, start=start, end=end))

st.subheader('Data from 2010 to 2024')
st.write(df.describe())

# Display the current closing price
current_close = df['Close'].iloc[-1]
st.subheader(f'Current Closing Price: {current_close:.2f}')

# Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA(moving average)')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA(moving average)')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

# Prepare training and testing data
data_training = pd.DataFrame(df['Close'][0: int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load the model
model = load_model('keras_model.h5')

# Prepare the input data for prediction
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
y_predicted = model.predict(x_test)

# Inverse scale predictions
y_predicted = scaler.inverse_transform(np.column_stack((y_predicted, np.zeros(len(y_predicted)))))[:, 0]
y_test = scaler.inverse_transform(np.column_stack((y_test, np.zeros(len(y_test)))))[:, 0]

# Predicted vs Original
st.subheader('Predicted vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

# Prepare for next 10 days prediction
last_100_days = input_data[-100:].reshape((1, 100, 1))  # Reshape for the model
next_10_days = []

for _ in range(10):
    next_price = model.predict(last_100_days)
    next_10_days.append(next_price[0][0])  # Store predicted price
    next_price_reshaped = next_price.reshape((1, 1, 1))  # Reshape next_price to (1, 1, 1)
    last_100_days = np.append(last_100_days[:, 1:, :], next_price_reshaped, axis=1)  # Append along the time step dimension

# Inverse transform the next 10 days predictions
next_10_days = scaler.inverse_transform(np.column_stack((next_10_days, np.zeros(len(next_10_days)))))[:, 0]

# Create a DataFrame for the next 10 days
next_10_days_df = pd.DataFrame(next_10_days, columns=['Predicted Price'])
next_10_days_df.index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=10, freq='B')  # Business days

# Plot the next 10 days prediction
st.subheader('Next 10 Days Price Prediction on basis of previous 100 days')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(next_10_days_df.index, next_10_days_df['Predicted Price'], 'g', label='Predicted Price for Next 10 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Next 10 Days Price Prediction')
plt.legend()
st.pyplot(fig3)
