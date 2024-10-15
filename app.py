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
    df = pd.DataFrame(yf.download(user_input, start=start, end=end))

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
    fig = plt.figure(figsize=(1
