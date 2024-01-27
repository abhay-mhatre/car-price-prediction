import pandas as pd
import numpy as np 
import pickle as pk 
import streamlit as st 

model = pk.load(open('model.pkl', 'wb'))

st.header('Car Price Prediction ML Model')