# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tweepy
from textblob import TextBlob

# Twitter API keys
consumer_key = 'J8byEqCJVeadFYXaXXpxB0XPA'
consumer_secret = 'BtCnypxBLpOcjmH40o6sdeFkVtkEVN9ETZVj0fjLyR6kBMAduJ'
access_token = '593352028-586dxldnHIrPKM2aSfsq0yJBwe9ulEQNk6LWMlln'
access_token_secret = 'JOnyIQx4oiR96Sp72vMQwZFJRdoOy2dtCXZqS7kbyrV2k'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Streamlit app setup
st.title('Stock Price Forecasting with LSTM Model and Sentiment Analysis')
user_input = st.text_input('Enter any stock name (ticker)', 'TATASTEEL.NS')

# Set start and end dates
start = '2010-01-01'
end = datetime.now().strftime('%Y-%m-%d')  # Current date

if st.button('Predict'):
    try:
        # Fetch stock data
        df = pd.DataFrame(yf.download(user_input, start=start, end=end))
        if df.empty:
            st.error('No data found for the specified ticker. Please check the ticker symbol.')
            st.stop()

        st.subheader('Data from 2010 to today')
        st.write(df.describe())

        # Closing Price Chart
        st.subheader('Closing Price vs Time chart')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        # Moving Average Chart (100MA and 200MA)
        ma100 = df['Close'].rolling(100).mean()
        ma200 = df['Close'].rolling(200).mean()

        st.subheader('Closing Price with 100MA & 200MA')
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='100MA', color='red')
        plt.plot(ma200, label='200MA', color='green')
        plt.plot(df['Close'], label='Closing Price', color='blue')
        plt.legend()
        st.pyplot(fig2)

        # Data Preprocessing for LSTM
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Load pre-trained LSTM model
        model = load_model('keras_model.h5')

        # Prepare test data
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)

        # Inverse transform predictions
        y_predicted = scaler.inverse_transform(np.column_stack((y_predicted, np.zeros(len(y_predicted)))))[:, 0]
        y_test = scaler.inverse_transform(np.column_stack((y_test, np.zeros(len(y_test)))))[:, 0]

        # Plot Predicted vs Original Prices
        st.subheader('Predicted vs Original Prices')
        fig3 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Original Price', color='blue')
        plt.plot(y_predicted, label='Predicted Price', color='red')
        plt.legend()
        st.pyplot(fig3)

        # Sentiment Analysis for Tata Steel
        st.subheader('Sentiment Analysis for Tata Steel')
        query = "Tata Steel -filter:retweets"
        tweets = tweepy.Cursor(api.search_tweets, q=query, lang='en', tweet_mode='extended').items(200)

        sentiments = []
        for tweet in tweets:
            text = tweet.full_text
            analysis = TextBlob(text)
            sentiment_score = analysis.sentiment.polarity
            sentiments.append({
                "Date": tweet.created_at.date(),
                "Sentiment": sentiment_score
            })

        sentiment_df = pd.DataFrame(sentiments)
        sentiment_df = sentiment_df.groupby('Date').mean().reset_index()

        st.write("Sentiment Data:")
        st.write(sentiment_df)

        # Merge sentiment with stock data
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        merged_df = pd.merge(df, sentiment_df, on='Date', how='left')
        merged_df['Sentiment'] = merged_df['Sentiment'].fillna(0)

        # Plot Closing Price vs Sentiment
        st.subheader('Closing Price vs Sentiment')
        fig4 = plt.figure(figsize=(12, 6))
        plt.plot(merged_df['Date'], merged_df['Close'], label='Closing Price', color='blue')
        plt.plot(merged_df['Date'], merged_df['Sentiment'] * 100, label='Sentiment Score (scaled)', color='orange')
        plt.legend()
        st.pyplot(fig4)

    except Exception as e:
        st.error(f"An error occurred: {e}")
