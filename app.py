# Import necessary libraries
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
user_input = st.text_input('Enter any stock name (ticker)', 'TATASTEEL.NS')

# Add an "Enter" button after the input field
if st.button('Predict'):
    # Download stock data
    try:
        df = pd.DataFrame(yf.download(user_input, start=start, end=end))
        if df.empty:
            st.error('No data found for the specified ticker. Please check the ticker symbol.')
            st.stop()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    st.subheader('Data from 2010 to today')
    st.write(df.describe())

    # Display the current closing price
    current_close = df['Close'].iloc[-1]
    st.subheader(f'Current Closing Price: {current_close:.2f}')

    # Visualizations
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA (moving average)')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA & 200MA (moving average)')
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
    try:
        model = load_model('keras_model.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

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
    st.subheader('Predicted vs Original using LSTM')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Days')
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig2)

    # Prepare for next 30 days prediction
    last_100_days = input_data[-100:].reshape((1, 100, 1))  # Reshape for the model
    temp_input = list(last_100_days.flatten())  # Convert the input data to a list for manipulation
    lst_output = []  # List to store the predicted outputs

    n_steps = 100
    i = 0

    while i < 30:
        if len(temp_input) > 100:
            x_input = np.array(temp_input[1:])  # Drop the first element and use the rest
            x_input = x_input.reshape((1, n_steps, 1))  # Reshape for the model
            yhat = model.predict(x_input, verbose=0)  # Predict next step
            temp_input.extend(yhat[0].tolist())  # Add prediction to the input
            temp_input = temp_input[1:]  # Remove the first element
            lst_output.extend(yhat.tolist())  # Add prediction to the output list
            i += 1
        else:
            x_input = np.array(temp_input).reshape((1, n_steps, 1))  # Reshape the input
            yhat = model.predict(x_input, verbose=0)  # Predict next step
            temp_input.extend(yhat[0].tolist())  # Add prediction to the input
            lst_output.extend(yhat.tolist())  # Add prediction to the output list
            i += 1

    # Inverse transform the next 30 days predictions
    next_30_days = scaler.inverse_transform(np.column_stack((lst_output, np.zeros(len(lst_output)))))[:, 0]

    # Create a DataFrame for the next 30 days
    next_30_days_df = pd.DataFrame(next_30_days, columns=['Predicted Price'])
    next_30_days_df.index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')  # Business days

    # Plot the next 30 days prediction
  # Plot the next 30 days prediction
    st.subheader('Next 30 Days Price Prediction based on previous 100 days')
    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(next_30_days_df.index, next_30_days_df['Predicted Price'], 'g', label='Predicted Price for Next 30 Days')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Next 30 Days Price Prediction')
    plt.legend()
    st.pyplot(fig3)
