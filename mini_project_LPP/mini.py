import streamlit as st
import cloudpickle  # Using cloudpickle instead of pickle
import numpy as np

# Load the model and dataframe
with open('pipe.pkl', 'rb') as f:
    pipe = cloudpickle.load(f)

with open('df.pkl', 'rb') as f:
    df = cloudpickle.load(f)

st.title("💻 Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop (in kg)')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS Display
ips = st.selectbox('IPS Display', ['No', 'Yes'])

# Screen Size
screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.0)

# Resolution
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

# CPU
cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())

# OS
os = st.selectbox('Operating System', df['os'].unique())

# Predict button
if st.button('💰 Predict Price'):
    # Convert categorical options to numerical where needed
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI (Pixels Per Inch)
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res*2 + Y_res*2) ** 0.5) / screen_size

    # Final input array
    query = np.array([company, laptop_type, ram, weight, touchscreen,
                      ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)

    # Predict and show result
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.subheader(f"🟢 The predicted price is: ₹ {predicted_price:,}")

