import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open('model.pkl', 'rb'))

st.header('Car Price Prediction ML Model')

cars_data = pd.read_csv(r"D:\Data Is Good\Projects\car price prediction\Car details v3.csv")

# Defining function to extract car brand name:
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Dropdown for Car brand
st.selectbox("Select Car Brand", cars_data['name'].unique())

# Slider for manufacture year:
st.slider("Car Manufacture year", 1994,2024)

# Slider for kms driven:
st.slider("Number of kms Driven", 11,200000)

# Dropdown for Fuel type
st.selectbox("Select Fuel type", cars_data['fuel'].unique())

# Dropdown for Seller type
st.selectbox("Select seller type", cars_data['seller_type'].unique())

# Slider for car mileage
st.slider("Car Mileage", 10, 40)

# Slider for engine capacity
st.slider("Engine Capacity",500,5000)

# Slider for engine power:
st.slider("BHP",0,200)

# Number of seats
st.slider("Seating capacity",5,10)