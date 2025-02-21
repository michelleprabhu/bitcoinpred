import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.express as px

# Load trained models
xgb_model = joblib.load("xgboost_model.pkl")
lstm_model = joblib.load("lstm_model.pkl")  # Assuming it's a serialized sklearn-like model

st.title("Bitcoin Price Prediction App")

st.sidebar.header("Input Features")
date_input = st.sidebar.date_input("Select Date", datetime.date.today())
market_index = st.sidebar.number_input("Market Index", value=50000.0)
oil_price = st.sidebar.number_input("Oil Price (USD)", value=80.0)
money_supply = st.sidebar.number_input("Money Supply", value=1e12)

features = np.array([[market_index, oil_price, money_supply]])

if st.sidebar.button("Predict with XGBoost"):
    prediction = xgb_model.predict(features)[0]
    st.write(f"### XGBoost Predicted Bitcoin Price: ${prediction:,.2f}")

if st.sidebar.button("Predict with LSTM"):
    prediction = lstm_model.predict(features)[0]  # Modify if LSTM requires different input formatting
    st.write(f"### LSTM Predicted Bitcoin Price: ${prediction:,.2f}")

# Historical Data Visualization (Assuming a CSV file with Bitcoin prices is available)
st.header("Historical Bitcoin Prices")
price_data = pd.read_csv("historical_bitcoin_prices.csv")
st.dataframe(price_data.tail(10))
fig = px.line(price_data, x="Date", y="Price", title="Bitcoin Price Trend")
st.plotly_chart(fig)
