import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder


# -----------------------------
# Load trained models
# -----------------------------

yield_model = joblib.load("models/yield_model.pkl")
price_model = joblib.load("models/price_model.pkl")

print("Models loaded successfully")


# -----------------------------
# Example farmer input
# -----------------------------

state = "Odisha"
season = "Kharif"
area = 5
temperature = 26
rainfall = 1500


# candidate crops to evaluate
candidate_crops = [
    "Rice",
    "Maize",
    "Wheat",
    "Potato",
    "Ragi"
]


# -----------------------------
# Encode categorical features
# -----------------------------

le_state = LabelEncoder()
le_crop = LabelEncoder()
le_season = LabelEncoder()

# these lists should include all possible categories
states = ["Odisha","Andhra Pradesh","Tamil Nadu","Karnataka"]
crops = candidate_crops
seasons = ["Kharif","Rabi","Whole Year","Summer"]

le_state.fit(states)
le_crop.fit(crops)
le_season.fit(seasons)

state_encoded = le_state.transform([state])[0]
season_encoded = le_season.transform([season])[0]


results = []


# -----------------------------
# Evaluate each crop
# -----------------------------

for crop in candidate_crops:

    crop_encoded = le_crop.transform([crop])[0]

    # Yield model input
    yield_input = pd.DataFrame([{
        "State": state_encoded,
        "Crop": crop_encoded,
        "Season": season_encoded,
        "Area": area,
        "Rainfall": rainfall,
        "Temperature": temperature
    }])

    # Predict yield
    yield_log = yield_model.predict(yield_input)
    predicted_yield = max(0.1, np.expm1(yield_log)[0])


    # Price model input
    price_input = pd.DataFrame([{
        "Year": 2024,
        "Month": 6,
        "State": state_encoded,
        "Crop": crop_encoded,
        "Temperature (Celsis)": temperature,
        "Rainfall in mm": rainfall
    }])

    price_log = price_model.predict(price_input)
    predicted_price = max(1, np.expm1(price_log)[0])


    # Profit calculation
    profit = predicted_yield * predicted_price * area


    results.append({
        "Crop": crop,
        "Predicted Yield": predicted_yield,
        "Predicted Price": predicted_price,
        "Expected Profit": profit
    })


# -----------------------------
# Show recommendation
# -----------------------------

df_results = pd.DataFrame(results)

df_results = df_results.sort_values(
    by="Expected Profit",
    ascending=False
)

print("\nCrop Recommendation Results\n")
print(df_results)

print("\nBest Crop to Grow:")
print(df_results.iloc[0]["Crop"])