"""
app.py

Streamlit interface to load melb_price_model.h5
and predict property prices interactively.
"""

import pandas as pd
import joblib
import streamlit as st

MODEL_FILE = "melb_price_model.h5"
CSV_PATH = "melb_data_cleaned.csv"   # To fetch suburb list


# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load(MODEL_FILE)

# Load dataset just to get suburb list
df = pd.read_csv(CSV_PATH)
suburb_list = sorted(df["Suburb"].dropna().unique())


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="🏡 Melbourne House Price Predictor", layout="centered")

st.title("🏡 Melbourne Property Price Prediction")
st.write("Enter property details below to predict the price.")

st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This app uses a **Random Forest Regressor** trained on `melb_data_cleaned.csv`
to predict Melbourne property prices.
""")

# Input form
with st.form("prediction_form"):
    landsize = st.number_input("Land size (m²)", min_value=0, max_value=10000, value=500)
    building_area = st.number_input("Building area (m²)", min_value=0, max_value=1000, value=150)
    bedroom2 = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=3)
    bathroom = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2)
    car = st.number_input("Car spaces", min_value=0, max_value=10, value=1)

    type_choice = st.selectbox("Property Type", ["h", "u", "t"])  # h=house, u=unit, t=townhouse
    suburb = st.selectbox("Suburb", suburb_list, index=suburb_list.index("Richmond") if "Richmond" in suburb_list else 0)
    submitted = st.form_submit_button("Predict Price 💰")

# Prediction
if submitted:
    input_data = pd.DataFrame([{
        "Landsize": landsize,
        "Type": type_choice,
        "Bedroom2": bedroom2,
        "Suburb": suburb,
        "Car": car,
        "Bathroom": bathroom,
        "BuildingArea": building_area
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"🏠 Estimated Property Price: **${prediction:,.0f}**")
