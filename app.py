import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model  # type: ignore
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Apply custom CSS for background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://media.istockphoto.com/id/1389093478/photo/stock-market-or-forex-trading-graph-with-map-world-representing-the-global-network-line-wire.jpg?s=612x612&w=0&k=20&c=luy7RF6EZsg7ls-_9ualIiAxZLPW9Kmk3jgL_HH6gtQ=");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    .stHeader {
        color: white !important;
        font-size: 28px !important;
        text-align: center;
        font-weight: bold;
        background: rgba(0, 0, 0, 0.5);
        padding: 10px;
        border-radius: 10px;
    }

    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.8);
        color: black;
        border-radius: 5px;
    }

    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 10px;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
model = load_model('D:\Stock Market Projections\Stock Predictions Model.keras')

st.markdown('<h1 class="stHeader">Stock Market Predictor</h1>', unsafe_allow_html=True)

# User Input for Stock Symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Set date range
start = '2012-01-01'
end = '2022-12-31'

# Fetch stock data
data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

# Split the data into train and test sets
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

# Prepare test data with last 100 days from training
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plot MA50
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label="50-day MA")
plt.plot(data.Close, 'g', label="Stock Price")
plt.legend()
plt.show()
st.pyplot(fig1)

# Plot MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label="50-day MA")
plt.plot(ma_100_days, 'b', label="100-day MA")
plt.plot(data.Close, 'g', label="Stock Price")
plt.legend()
plt.show()
st.pyplot(fig2)

# Plot MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label="100-day MA")
plt.plot(ma_200_days, 'b', label="200-day MA")
plt.plot(data.Close, 'g', label="Stock Price")
plt.legend()
plt.show()
st.pyplot(fig3)

# Prepare data for LSTM model
x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Predict
predict = model.predict(x)

# Scale back to original values
scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

# Plot Predictions
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, 'g', label="Original Price")
plt.plot(predict, 'r', label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)
