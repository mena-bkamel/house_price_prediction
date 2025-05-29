import numpy as np
import streamlit as st
import joblib
import pandas as pd
import os

# Load model
MODEL_PATH = "models/tuned_gradient_boosting_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

st.title("🏠 House Price Prediction")

# Input form
st.sidebar.header("Property Details")

area = st.sidebar.number_input("Area (sqft)", min_value=500, max_value=20000, value=2000)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=6, value=2)
stories = st.sidebar.number_input("Stories", min_value=1, max_value=5, value=2)
mainroad = st.sidebar.selectbox("Near Main Road", ['yes', 'no'], index=0)
guestroom = st.sidebar.selectbox("Guest Room", ['yes', 'no'], index=1)
basement = st.sidebar.selectbox("Basement", ['yes', 'no'], index=1)
hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ['yes', 'no'], index=1)
airconditioning = st.sidebar.selectbox("Air Conditioning", ['yes', 'no'], index=0)
parking = st.sidebar.number_input("Parking Spaces", min_value=0, max_value=4, value=1)
prefarea = st.sidebar.selectbox("Preferred Area", ['yes', 'no'], index=0)
furnishingstatus = st.sidebar.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'], index=1)


if st.sidebar.button("Predict Price"):
    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus,
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Convert binary features
    binary_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

    for feature in binary_features:
        df[feature] = df[feature].map({'yes': 1, 'no': 0})

    # Feature engineering
    df['price_per_sqft'] = 0 # Placeholder
    df['room_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 1)

    # One-hot encode furnishingstatus
    df = pd.get_dummies(df, columns=['furnishingstatus'], prefix_sep='furnish')

    # Ensure all expected columns exist
    expected_columns = [
        'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
        'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
        'parking', 'prefarea', 'price_per_sqft', 'room_ratio',
        'furnish_furnished', 'furnish_semi-furnished', 'furnish_unfurnished'
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df =df[expected_columns]

    try:
        prediction = model.predict(df)
        price = prediction[0]
        price_per_sqft = price / area

        st.success(f"### Predicted Price: {price: ,.2f}")
        st.info(f"**Price per sqft: {price_per_sqft: ,.2f}")

        # Feature importance
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            st.subheader("Feature Importance")
            importances = model.named_steps['model'].feature_importances_
            importances_df = pd.DataFrame({
                'Feature': expected_columns,
                'importance': importances
            }).sort_values(by='importance', ascending=False)
            st.bar_chart(importances_df.set_index('Feature'))

    except Exception as e:
        st.error(f" An error occurred: {str(e)}")


# Add documentation
st.markdown("""
### About This Model
This tool predicts house prices based on:
- **Property characteristics**: Area, bedrooms, bathrooms, stories
- **Amenities**: Parking, AC, water heating, basement, guest room
- **Location**: Main road access, preferred area
- **Furnishing status**: Furnished, semi-furnished, unfurnished

The model was trained using Gradient Boosting algorithm.
""")



