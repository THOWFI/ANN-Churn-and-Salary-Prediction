import streamlit as st
import numpy as mp
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf

## Loading Train Model
model=tf.keras.models.load_model('rgmodel.h5')

## Loading Scaler,Encoder pickel
with open('ohe_geo_salary.pkl','rb') as file:
    ohe_geo=pickle.load(file)
with open('le_gender_salary.pkl','rb') as file:
    le_gender=pickle.load(file)
with open('scaler_salary.pkl','rb') as file:
    scaler=pickle.load(file)

## Streamlit app
st.title('Customer Salary Prediction')

## User Inputs
geography = st.selectbox('Geography',ohe_geo.categories_[0])
gender = st.selectbox('Gender',le_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited',[0,1])
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Product',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

# User Data into Dict and into Table

input_data = {
    'CreditScore': [credit_score],
    'Gender': [le_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
}

input_df = pd.DataFrame(input_data)

# OHE 'Geography'

## OHE in Geography
geo_encoded=ohe_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=['France', 'Germany', 'Spain']
)

## Concating the Geography with tabel data
input_df=pd.concat([input_df.reset_index(drop=True),geo_encoded_df],axis=1)

# Scaling the input data
input_df_scaled = scaler.transform(input_df)

# Predicting Churn
prediction = model.predict(input_df_scaled)
prediction_proba = prediction[0][0]

st.write(f'Estimated Salary : {prediction_proba:.2f}')