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
name = st.selectbox("Select Car Brand", cars_data['name'].unique())

# Slider for manufacture year:
year = st.slider("Car Manufacture year", 1994,2024)

# Slider for kms driven:
km_driven = st.slider("Number of kms Driven", 11,200000)

# Dropdown for Fuel type
fuel = st.selectbox("Select Fuel type", cars_data['fuel'].unique())

# Dropdown for Seller type
seller_type = st.selectbox("Select seller type", cars_data['seller_type'].unique())

# Dropdown for transmission
transmission = st.selectbox("Select transmission type:", cars_data['transmission'].unique())

# Dropdown for owner:
owner = st.selectbox("Select ownership level",cars_data['owner'].unique())

# Slider for car mileage
mileage = st.slider("Car Mileage", 10, 40)

# Slider for engine capacity
engine = st.slider("Engine Capacity",500,5000)

# Slider for engine power:
max_power = st.slider("BHP",0,200)

# Number of seats
seats = st.slider("Seating capacity",5,10)

# Creating button:

sample_values = [name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]

columns_df = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']



if st.button("Predict Car Price"): # this if conditionn runs only if the button is clicked
    input_data_model = pd.DataFrame([sample_values], columns = columns_df)
    # st.write(input_data_model)

    # we also have to fix the categorical columns
    cols1 = ['name','fuel','seller_type', 'transmission', 'owner'] # I'm being lazy here, please manually hardcode this
    for i in cols1:
        input_data_model[i].replace(cars_data[i].unique(),
                                    list(range(1, cars_data[i].nunique()+1)), 
                                    inplace = True)
    
    # Predicting car price:
    car_price = model.predict(input_data_model)
    # car_price = round(car_price)
    
    # st.markdown(f"Car price is Rs.{car_price:,}.")
    st.markdown("Car price is Rs.{}.".format(car_price))

