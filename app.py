# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col

# Create Spark session
spark = SparkSession.builder.appName("StockPriceForecasting").getOrCreate()

# Set the start date and current end date
start = '2010-01-01'
end = datetime.now().strftime('%Y-%m-%d')  # Get today's date

st.title('Stock Price Forecasting with LSTM Model (AIBF project)')
user_input = st.text_input('Enter any stock name (ticker)', 'TATASTEEL.NS')

# Add an "Enter" button after the input field
if st.button('Predict'):
    # Download stock data
    df = yf.download(user_input, start=start, end=end)
    
    # Convert to PySpark DataFrame
    df_spark = spark.createDataFrame(df.reset_index())
    
    st.subheader('Data from 2010 to today')
    st.write(df.describe())  # Display descriptive stats in Streamlit

    # Display the current closing price
    current_close = df['Close'].iloc[-1]
    st.subheader(f'Current Closing Price: {current_close:.2f}')

    # Visualizations
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'])
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA (moving average)')
    ma100 = df['Close'].rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, label='100MA')
    plt.plot(df['Close'], label='Close Price')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA & 200MA (moving average)')
    ma200 = df['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, label='100MA')
    plt.plot(ma200, label='200MA')
    plt.plot(df['Close'], label='Close Price')
    plt.legend()
    st.pyplot(fig)

    # Split data for training and testing (70% train, 30% test)
    train_len = int(len(df) * 0.70)
    data_training = df[:train_len]
    data_testing = df[train_len:]

    # Convert training data to PySpark DataFrame
    data_training_spark = spark.createDataFrame(data_training.reset_index())
    
    # Prepare data for scaling
    def vectorize(column):
        return Vectors.dense(column)

    df_spark_with_vectors = df_spark.select(col("Close")).rdd.map(lambda row: vectorize([row[0]])).toDF(["features"])

    # Scaling with MinMaxScaler
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(df_spark_with_vectors)
    scaled_training_data = scaler_model.transform(df_spark_with_vectors)

    # Extract the numpy array for use with the Keras model
    data_training_array = np.array(scaled_training_data.select("scaled_features").rdd.map(lambda row: row[0][0]).collect()).reshape(-1, 1)

    # Load the pre-trained LSTM model
    model = load_model('keras_model.h5')

    # Prepare input data for testing
    past_100_days = data_training[-100:]
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

    # Prepare input data for prediction in the required format
    input_data = scaler_model.transform(spark.createDataFrame(final_df[['Close']].reset_index())).select("scaled_features").rdd.map(lambda row: row[0][0]).collect()

    # Reshape input data for prediction
    x_test = []
    for i in range(100, len(input_data)):
        x_test.append(input_data[i-100: i])

    x_test = np.array(x_test)

    # Make predictions using the LSTM model
    y_predicted = model.predict(x_test)

    # Inverse scale predictions
    y_predicted_inverse = scaler_model.inverse_transform(spark.createDataFrame(y_predicted, ["Close"]))
    y_test = np.array(data_testing['Close'])

    # Visualize Predicted vs Original
    st.subheader('Predicted vs Original using LSTM')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Days')
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig2)

    # Next 10-day prediction logic
    last_100_days = np.array(input_data[-100:]).reshape(1, 100, 1)
    next_10_days = []

    for _ in range(10):
        next_price = model.predict(last_100_days)
        next_10_days.append(next_price[0][0])
        last_100_days = np.append(last_100_days[:, 1:, :], [[next_price]], axis=1)

    next_10_days = scaler_model.inverse_transform(spark.createDataFrame(pd.DataFrame(next_10_days), ["Close"]))

    # Create DataFrame for the next 10 days
    next_10_days_df = pd.DataFrame(next_10_days.collect(), columns=['Predicted Price'])
    next_10_days_df.index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=10, freq='B')

    # Plot the next 10 days prediction
    st.subheader('Next 10 Days Price Prediction based on previous 100 days')
    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(next_10_days_df.index, next_10_days_df['Predicted Price'], 'g', label='Predicted Price for Next 10 Days')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig3)
