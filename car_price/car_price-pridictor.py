import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
model=pk.load(open('model.pkl','rb'))
st.header(' 🚗 car price pridiction ml model')
dataset = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
  car_name = car_name.split(' ')[0]
  return car_name.strip('')
dataset['name'] =dataset['name'].apply(get_brand_name)
name=st.selectbox('select car name',dataset['name'].unique())
year=st.slider('car manufacturing year',1995,2026)
km_driven=st.slider('No of kms driven',11,2000000)
fuel=st.selectbox('fuel_type',dataset['fuel'].unique())
transmission=st.selectbox('transmission_type',dataset['transmission'].unique())
seller_type=st.selectbox('seller_type',dataset['seller_type'].unique())
owner=st.selectbox('owner type',dataset['owner'].unique())
mileage=st.slider('car Mileage',10,40)
engine=st.slider('Engine',700,5000)
max_power=st.slider('max power',0,200)
seats=st.slider('car seats',5,10)


if st.button('predict'):
  input_data_model = pd.DataFrame(
   [[	name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats
   ]] ,
  columns=["name","year","km_driven","fuel","seller_type","transmission","owner","mileage","engine","max_power","seats"]
  )
  input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner','Fourth & Above Owner', 'Test Drive Car'],[1,2,3,4,5],inplace=True)
  input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4],inplace=True)
  input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3],inplace=True)
  input_data_model['transmission'].replace(['Manual', 'Automatic'],[1,2],inplace=True)
  input_data_model['name'].replace(['Maruti','Skoda','Honda','Hyundai','Toyota','Ford' ,'Renault','Mahindra'
  , 'Tata','Chevrolet','Datsun','Jeep','Mercedes-Benz','Mitsubishi','Audi',
   'Volkswagen','BMW' ,'Nissan','Lexus','Jaguar','Land','MG','Volvo','Daewoo',
   'Kia','Fiat','Force','Ambassador','Ashok','Isuzu','Opel'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],inplace=True)

  car_price=model.predict(input_data_model)
  car_price_value = abs(car_price[0])  
  st.markdown('car price is going to be '+ str(car_price_value))
