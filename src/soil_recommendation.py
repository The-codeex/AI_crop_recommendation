''' 

import joblib
import numpy as np

# -------------------------------
# Load model and encoders
# -------------------------------

model = joblib.load("models/soil_model.pkl")

scaler = joblib.load("models/soil_scaler.pkl")

le_state = joblib.load("models/soil_state_encoder.pkl")
le_soil = joblib.load("models/soil_encoder.pkl")
le_crop = joblib.load("models/soil_crop_encoder.pkl")

print("Soil model loaded successfully")

# -------------------------------
# Example user input
# -------------------------------

state = "Andaman and Nicobar"
soil_type = "Sandy soil"

n = 90
p = 42
k = 43

temperature = 21
humidity = 80
ph = 6.5
rainfall = 200

# -------------------------------
# Encode input
# -------------------------------

state_encoded = le_state.transform([state])[0]

soil_encoded = le_soil.transform([soil_type])[0]

input_data = np.array([[
    state_encoded,
    soil_encoded,
    n,
    p,
    k,
    temperature,
    humidity,
    ph,
    rainfall
]])

# -------------------------------
# Scale input
# -------------------------------

input_scaled = scaler.transform(input_data)

# -------------------------------
# Prediction
# -------------------------------

probs = model.predict_proba(input_scaled)[0]

# -------------------------------
# Top crops
# -------------------------------

top_indices = np.argsort(probs)[::-1][:5]

recommended_crops = le_crop.inverse_transform(top_indices)

print("\nRecommended Crops Based on Soil:\n")

for i, crop in enumerate(recommended_crops, 1):
    print(f"{i}. {crop}")
    
'''

import joblib
import numpy as np

# -------------------------------
# Load model and encoders
# -------------------------------

model    = joblib.load("models/soil_model.pkl")
scaler   = joblib.load("models/soil_scaler.pkl")
le_state = joblib.load("models/soil_state_encoder.pkl")
le_soil  = joblib.load("models/soil_encoder.pkl")
le_crop  = joblib.load("models/soil_crop_encoder.pkl")

CONFUSABLE_VEGS  = joblib.load("models/soil_confusable_vegs.pkl")
CONFIDENCE_THRESHOLD = 0.40

print("Soil model loaded successfully")

# -------------------------------
# Example user input
# -------------------------------

state      = "Andaman and Nicobar"
soil_type  = "Sandy soil"

n           = 90
p           = 42
k           = 43
temperature = 21
humidity    = 80
ph          = 6.5
rainfall    = 200
crop_price  = 1500    # average/expected market price for the region

# -------------------------------
# Encode categorical inputs
# -------------------------------

state_encoded = le_state.transform([state])[0]
soil_encoded  = le_soil.transform([soil_type])[0]

# -------------------------------
# Engineered features
# (must match exactly what was used during training)
# -------------------------------

n_p_ratio  = n / (p + 1)
n_k_ratio  = n / (k + 1)
p_k_ratio  = p / (k + 1)
npk_total  = n + p + k
temp_humid = temperature * humidity / 100
rain_humid = rainfall    * humidity / 100

# -------------------------------
# Build input array — 16 features
# Order must match FEATURES list in train_soil_model.py
# -------------------------------

input_data = np.array([[
    state_encoded,   # STATE
    soil_encoded,    # SOIL_TYPE
    n,               # N_SOIL
    p,               # P_SOIL
    k,               # K_SOIL
    temperature,     # TEMPERATURE
    humidity,        # HUMIDITY
    ph,              # ph
    rainfall,        # RAINFALL
    crop_price,      # CROP_PRICE
    n_p_ratio,       # N_P_ratio
    n_k_ratio,       # N_K_ratio
    p_k_ratio,       # P_K_ratio
    npk_total,       # NPK_total
    temp_humid,      # temp_humid
    rain_humid,      # rain_humid
]])

# -------------------------------
# Scale input
# -------------------------------

input_scaled = scaler.transform(input_data)

# -------------------------------
# Get probabilities
# -------------------------------

probs      = model.predict_proba(input_scaled)[0]
top_idxs   = np.argsort(probs)[::-1][:5]
top_crop   = le_crop.classes_[top_idxs[0]]
top_conf   = probs[top_idxs[0]]

# -------------------------------
# Two-Tier Recommendation
# -------------------------------

print(f"\nInput Summary:")
print(f"  State: {state} | Soil: {soil_type}")
print(f"  N={n}, P={p}, K={k} | Temp={temperature}°C | Humidity={humidity}%")
print(f"  pH={ph} | Rainfall={rainfall}mm | Crop Price=₹{crop_price}")

print(f"\nModel top prediction: {top_crop}  (confidence: {top_conf*100:.1f}%)")

if top_conf < CONFIDENCE_THRESHOLD and top_crop in CONFUSABLE_VEGS:
    # Tier 2: low confidence on a confusable vegetable
    # Return the full cluster — all are equally valid
    print("\n⚠  Low confidence on similar-condition crop.")
    print("   The following vegetables all grow well in these conditions:\n")

    cluster = []
    for veg in sorted(CONFUSABLE_VEGS):
        if veg in le_crop.classes_:
            idx = le_crop.transform([veg])[0]
            cluster.append((veg, probs[idx]))

    cluster.sort(key=lambda x: -x[1])
    for i, (veg, conf) in enumerate(cluster, 1):
        print(f"  {i}. {veg:<18}  (model score: {conf*100:.1f}%)")

else:
    # Tier 1: confident prediction — show top 5
    print("\n✓  Recommended Crops Based on Soil (Top 5):\n")
    for i, idx in enumerate(top_idxs, 1):
        crop = le_crop.classes_[idx]
        conf = probs[idx]
        bar  = "█" * int(conf * 30)
        print(f"  {i}. {crop:<18}  {conf*100:.1f}%  {bar}")