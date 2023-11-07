import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

model = load_model("ml_airbnb")
st.title("Sistema de predicción de precios Upgrade-Hub")

neighbourhood = st.selectbox('Barrio', options=[
    'Alna', 'Frogner', 'Sagene', 'Gamle Oslo',
    'St. Hanshaugen', 'Nordstrand', 'Grünerløkka', 'Nordre Aker', 'Vestre Aker',
    'Bjerke', 'Ullern', 'Sentrum', 'Stovner', 'Østensjø',
    'Marka', 'Søndre Nordstrand', 'Grorud'
])

property_type = st.selectbox('Tipo de Propiedad', options=[
    'Entire rental unit', 'Private room in rental unit', 'Entire condo', 'Entire home', 'Boat',
    'Casa particular', 'Entire bungalow', 'Entire cabin', 'Entire chalet',
    'Entire cottage', 'Entire guest suite', 'Entire guesthouse', 'Entire home/apt', 'Entire loft',
    'Entire place', 'Tiny house', 'Entire rental unit', 'Entire serviced apartment', 'Entire townhouse',
    'Entire vacation home', 'Entire villa', 'Houseboat', 'Private room', 'Tent',
    'Private room in bed and breakfast', 'Private room in boat', 'Private room in casa particular', 'Camper/RV',
    'Private room in condo', 'Private room in guest suite', 'Private room in guesthouse', 'Private room in home', 'Private room in loft',
    'Private room in serviced apartment', 'Private room in townhouse', 'Private room in vacation home', 'Private room in villa',
    'Room in aparthotel', 'Room in boutique hotel', 'Room in hotel', 'Room in serviced apartment', 'Shared room in condo', 'Shared room in farm stay',
    'Shared room in hotel', 'Shared room in loft', 'Shared room in rental unit', 'Shared room in townhouse', 'Shared room in vacation home'
])

accommodates = st.slider('Número de Personas', min_value=1, max_value=17, value=1)
room_type = st.selectbox('Tipo de Habitación', options=['Private room', 'Entire home/apt', 'Shared room'])
maximum_nights = st.slider('Noches Máximas', min_value=1, max_value=100, value=1)
minimum_nights = st.slider('Noches Mínimas', min_value=1, max_value=10, value=1)

input_data = pd.DataFrame([[
    neighbourhood, property_type, accommodates, room_type,
    maximum_nights, minimum_nights
]], columns=['neighbourhood', 'property_type', 'accommodates', 'room_type', 'maximum_nights', 'minimum_nights'])


if st.button('¡Descubre el precio!'):
    prediction = predict_model(model, data=input_data)
    st.write(str(prediction["prediction_label"].values[0]) + ' euros')