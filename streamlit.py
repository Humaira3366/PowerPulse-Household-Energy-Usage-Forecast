import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pickle

# Streamlit config
st.set_page_config(page_title="PowerPulse - Energy Forecast", layout="wide", page_icon="💡")

# Load model and features
model = joblib.load("model.pkl")
with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv(
        r"C:\Users\Humaira\OneDrive\Desktop\guvi_power forecast\household_power_consumption.txt",
        sep=";",
        parse_dates={"DateTime": ["Date", "Time"]},
        infer_datetime_format=True,
        na_values="?",
        low_memory=False
    )
    df.dropna(inplace=True)
    df["Global_active_power"] = df["Global_active_power"].astype(float)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime")
    
    # Feature engineering
    df["lag_1"] = df["Global_active_power"].shift(1)
    df["lag_2"] = df["Global_active_power"].shift(2)
    df["rolling_mean_3"] = df["Global_active_power"].rolling(window=3).mean()
    df["rolling_mean_7"] = df["Global_active_power"].rolling(window=7).mean()
    df["day"] = df["DateTime"].dt.day
    df["month"] = df["DateTime"].dt.month
    df["weekday"] = df["DateTime"].dt.weekday
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)

    df.dropna(inplace=True)
    return df

df = load_data()

# Title
st.title("⚡ PowerPulse: Household Energy Usage Forecast")
st.markdown("Analyze usage trends and predict household energy consumption.")

# Sidebar filters
st.sidebar.header("📅 Date Filter")
start = st.sidebar.date_input("Start Date", df["DateTime"].min().date())
end = st.sidebar.date_input("End Date", df["DateTime"].max().date())

filtered_df = df[(df["DateTime"].dt.date >= start) & (df["DateTime"].dt.date <= end)]

# Show data (limited)
if st.checkbox("🔍 Show Filtered Data"):
    max_rows = st.slider("Rows to display", 100, 5000, 1000, step=100)
    st.dataframe(filtered_df.head(max_rows))

# Line plot (safe sampling)
st.subheader("📈 Global Active Power Over Time")
plot_df = filtered_df.copy()
if len(plot_df) > 5000:
    plot_df = plot_df.sample(5000).sort_values("DateTime")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(plot_df["DateTime"], plot_df["Global_active_power"], color='teal')
ax.set_title("Global Active Power Over Time")
ax.set_xlabel("DateTime")
ax.set_ylabel("Power (kW)")
st.pyplot(fig)

# Prediction section
st.subheader("🔮 Predict Next Energy Usage (Based on Latest Values)")

latest_row = df.iloc[-1:]
input_data = latest_row[features]

if st.button("🚀 Predict Next Hour"):
    pred = model.predict(input_data)[0]
    st.success(f"🔋 Predicted Global Active Power: **{pred:.3f} kW**")

# Static model metrics
st.subheader("📊 Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("RMSE", "0.21")
c2.metric("MAE", "0.17")
c3.metric("R² Score", "0.89")
