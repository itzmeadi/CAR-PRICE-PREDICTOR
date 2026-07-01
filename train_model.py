"""
train_model.py
----------------
Mirrors the training logic from the notebook
(CAR_PRICE_PREDICTOR_ARAVAPALLI_M_ADITYA_SRIRAM_23MIM10130.ipynb).

If you have the real CarDekho 'car data.csv' file, put it in this folder
and this script will use it. Otherwise it falls back to a synthetically
generated dataset with the same schema so you still get a working
car_price_model.pkl for deployment/testing.

Run:  python train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

CSV_PATH = "car data.csv"

if os.path.exists(CSV_PATH):
    print(f"Found {CSV_PATH}, training on real CarDekho data...")
    df = pd.read_csv(CSV_PATH)
    df['Car_Age'] = 2025 - df['Year']
    df.drop(['Car_Name', 'Year'], axis=1, inplace=True)
    df['Fuel_Type'] = df['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
    df['Seller_Type'] = df['Seller_Type'].map({'Dealer': 0, 'Individual': 1})
    df['Transmission'] = df['Transmission'].map({'Manual': 0, 'Automatic': 1})
else:
    print(f"{CSV_PATH} not found — generating synthetic training data instead.")
    print("(Download the real dataset from Kaggle 'vehicle-dataset-from-cardekho'")
    print(" and re-run this script for a production-accurate model.)")
    rng = np.random.default_rng(42)
    n = 800
    present_price = rng.uniform(0.5, 35, n)
    car_age = rng.integers(0, 18, n)
    kms_driven = rng.integers(500, 150000, n)
    fuel_type = rng.integers(0, 3, n)
    seller_type = rng.integers(0, 2, n)
    transmission = rng.integers(0, 2, n)
    owner = rng.integers(0, 3, n)

    depreciation = 0.92 - 0.015 * car_age
    depreciation = np.clip(depreciation, 0.15, 0.92)
    km_penalty = 1 - (kms_driven / 500000)
    fuel_adj = np.where(fuel_type == 1, 1.03, np.where(fuel_type == 2, 0.9, 1.0))
    trans_adj = np.where(transmission == 1, 1.05, 1.0)
    owner_penalty = 1 - (owner * 0.05)

    selling_price = (
        present_price * depreciation * km_penalty * fuel_adj * trans_adj * owner_penalty
        + rng.normal(0, 0.4, n)
    )
    selling_price = np.clip(selling_price, 0.1, None)

    df = pd.DataFrame({
        'Present_Price': present_price,
        'Kms_Driven': kms_driven,
        'Fuel_Type': fuel_type,
        'Seller_Type': seller_type,
        'Transmission': transmission,
        'Owner': owner,
        'Car_Age': car_age,
        'Selling_Price': selling_price,
    })

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R2 score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

pickle.dump(model, open('car_price_model.pkl', 'wb'))
print("Saved car_price_model.pkl")
