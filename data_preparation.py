import numpy as np
import pandas as pd
from datetime import datetime as dt
import os


def load_and_prepare_data(data_path="./data/housing.csv"):
    """Load and prepare housing data"""

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    try:
        # Try to load real data
        df = pd.read_csv(data_path)
        print("Load data from CSV file")

        # Convert binary categorical features to numerical
        binary_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                           'airconditioning', 'prefarea']

        for feature in binary_features:
            df[feature] = df[feature].map({'yes': 1, 'no': 0})

        # One-hot encode furnishingstatus
        df = pd.get_dummies(df, columns=["furnishingstatus"], prefix="furnish")

        # Feature engineering
        df["price_per_sqft"] = df["price"] / df["area"]
        df["room_ratio"] = df["bedrooms"] / df["bathrooms"].replace(0, 1) # Avoid division by 0

        return df

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None

if __name__ == '__main__':
    housing_data = load_and_prepare_data()
    if housing_data is not None:
        print(housing_data.head())
        print("\nColumns:", housing_data.columns.tolist())
