import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

# Function to fetch crypto market data
def fetch_crypto_data(crypto="bitcoin", currency="usd", days=2):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart"
    params = {"vs_currency": currency, "days": days}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error(f"API Error {response.status_code}: {response.text}")
        return pd.DataFrame()
    data = response.json()
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# Detect outliers using IQR
def detect_outliers(df):
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df["price"] < lower) | (df["price"] > upper)]

# Prediction (Linear Regression)
def predict_prices(df, hours=5):
    df["time_index"] = np.arange(len(df))
    X = df["time_index"].values.reshape(-1, 1)
    y = df["price"].values
    model = LinearRegression().fit(X, y)

    future_index = np.arange(len(df), len(df) + hours).reshape(-1, 1)
    predictions = model.predict(future_index)
    future_time = [df["timestamp"].iloc[-1] + timedelta(hours=i+1) for i in range(hours)]
    return pd.DataFrame({"timestamp": future_time, "predicted_price": predictions})

# Streamlit UI
st.title("📊 Crypto Dashboard with Forecasting")

crypto = st.sidebar.selectbox("Choose Cryptocurrency", ["bitcoin", "ethereum", "dogecoin"])
df = fetch_crypto_data(crypto, "usd", days=2)

if not df.empty:
    # Convert to multiple currencies
    conversion_rates = {"USD": 1, "INR": 83, "GBP": 0.78}
    for cur, rate in conversion_rates.items():
        df[f"price_{cur}"] = df["price"] * rate

    # Last 5 hours vs yesterday
    now = df["timestamp"].max()
    last_5h = df[df["timestamp"] >= now - timedelta(hours=5)]
    yesterday = df[(df["timestamp"] >= now - timedelta(days=1)) & (df["timestamp"] < now - timedelta(days=1, hours=-5))]

    # Plot last 5 hours
    st.subheader("📈 Last 5 Hours Price Trend")
    st.line_chart(last_5h.set_index("timestamp")["price"])

    # Outliers
    st.subheader("⚠️ Detected Outliers")
    outliers = detect_outliers(last_5h)
    st.write(outliers)
    if not outliers.empty:
        fig, ax = plt.subplots()
        ax.plot(last_5h["timestamp"], last_5h["price"], label="Price")
        ax.scatter(outliers["timestamp"], outliers["price"], color="red", label="Outliers")
        ax.legend()
        st.pyplot(fig)

    # Yesterday vs Today comparison
    st.subheader("📊 Yesterday vs Last 5 Hours")
    compare_df = pd.DataFrame({
        "Yesterday": yesterday["price"].reset_index(drop=True),
        "Last 5 Hours": last_5h["price"].reset_index(drop=True)
    })
    st.line_chart(compare_df)

    # Prediction for next 5 hours
    st.subheader("🔮 Price Prediction (Next 5 Hours)")
    future_df = predict_prices(df, hours=5)
    st.write(future_df)
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["price"], label="Historical")
    ax.plot(future_df["timestamp"], future_df["predicted_price"], "r--", label="Predicted")
    ax.legend()
    st.pyplot(fig)
