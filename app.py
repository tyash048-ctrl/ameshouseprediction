import streamlit as st
import joblib
import pandas as pd

# Load the brain, the translator, and the map
model = joblib.load('models/lasso_model.pkl')
scaler = joblib.load('models/scaler.pkl')
features = joblib.load('models/feature_names.pkl')

#UI 
st.title("Ames House Predictor")
quality=st.slider("Overall House Quality",1,10,5)
living_area=st.number_input("Total Living Area(sq ft)",value=1500)
year_built = st.slider("Year Built", 1880, 2010, 1990)
garage_cars = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4])
total_rooms = st.number_input("Total Rooms Above Ground", value=6)


if st.button("Predict Price"):
    # 1. Create a row of zeros with the EXACT columns from your map
    input_df = pd.DataFrame(0, index=[0], columns=features)

    # 2. Assign the values
    input_df['Overall Qual'] = quality
    input_df['Gr Liv Area'] = living_area
    input_df['Year Built'] = year_built
    input_df['Garage Cars'] = garage_cars
    input_df['TotRms AbvGrd'] = total_rooms

   
    input_df = input_df[features]

    # 4. Transform and Predict
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)[0]

    st.success(f"Estimated Price: ${prediction:,.2f}")