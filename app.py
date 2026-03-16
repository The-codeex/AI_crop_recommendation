''' 
import streamlit as st
import pandas as pd
import numpy as np
import joblib

tab1, tab2 = st.tabs([
    "💰 Profit Based Recommendation",
    "🌱 Soil Based Recommendation"
])


# -------------------------------
# Load trained models
# -------------------------------

yield_model = joblib.load("models/yield_model.pkl")
price_model = joblib.load("models/price_model.pkl")

# Load encoders
le_state = joblib.load("models/state_encoder.pkl")
le_crop = joblib.load("models/crop_encoder.pkl")
le_season = joblib.load("models/season_encoder.pkl")

#load  soil model
soil_model = joblib.load("models/soil_model.pkl")
soil_state_encoder = joblib.load("models/soil_state_encoder.pkl")
soil_type_encoder = joblib.load("models/soil_encoder.pkl")
soil_crop_encoder = joblib.load("models/soil_crop_encoder.pkl")

# price_dataset = pd.read_csv("data/price_data.csv")

# Load cost dataset
cost_df = pd.read_csv("data/crop_cost.csv")

# Convert to dictionary
crop_cost = dict(zip(cost_df["Crop"], cost_df["Cost"]))

price_df = pd.read_csv("data/crop_price_avg.csv")
crop_avg_price = dict(zip(price_df["Crop"], price_df["Price"]))

# -------------------------------
# Streamlit UI
# -------------------------------
with tab1:
    st.title("AI-Based Crop Recommendation System")

    st.write("Predict the most profitable crop")

    # User Inputs
    state = st.selectbox("Select State", le_state.classes_)
    season = st.selectbox("Select Season", le_season.classes_)
    district = st.text_input("Type District")
    area = st.number_input("Area (hectares)", min_value=1.0)
    year = st.number_input("Year", min_value=2000, max_value=2035, value=2024)

    temperature = st.number_input("Temperature (°C)")
    rainfall = st.number_input("Rainfall (mm)")



# -------------------------------
# Prediction Button for main model
# -------------------------------

    if st.button("Predict Best Crop"):

        state_encoded = le_state.transform([state])[0]
        season_encoded = le_season.transform([season])[0]

        # price_crops = price_dataset["Crop"].unique()

        # if crop in price_crops
        candidate_crops = [
            crop for crop in le_crop.classes_
        
        ]

        results = []

        for crop in candidate_crops:

            crop_encoded = le_crop.transform([crop])[0]

            # -------------------------------
            # Yield Prediction
            # -------------------------------

            yield_input = pd.DataFrame([{
                "State": state_encoded,
                "Crop": crop_encoded,
                "Season": season_encoded,
                "Area": area,
                "Rainfall": rainfall,
                "Temperature": temperature
            }])

            yield_log = yield_model.predict(yield_input)

            predicted_yield = max(0.1, np.expm1(yield_log)[0])

            # -------------------------------
            # Price Prediction
            # -------------------------------

            price_input = pd.DataFrame([{
            "Year": year,
            "Month": 6,
            "State": state_encoded,
            "Crop": crop_encoded,
            "Temperature (Celsis)": temperature,
            "Rainfall in mm": rainfall
            # "TEMPERATURE":temperature,
            # "RAINFALL": rainfall
            }])

            # price_log = price_model.predict(price_input)
                # predicted_price = max(1, np.expm1(price_log)[0])
            price_log = price_model.predict(price_input)
            model_price = max(1, np.expm1(price_log)[0])
            # Get historical crop price
            dataset_price = crop_avg_price.get(crop, model_price)
            # Final price (weighted average)
            predicted_price = (model_price * 0.7) + (dataset_price * 0.3)

            # -------------------------------
            # Profit Calculation
            # -------------------------------

            # profit = predicted_yield * predicted_price * area
            # Get cost of cultivation
            default_cost = cost_df["Cost"].mean()
            cost = crop_cost.get(crop, default_cost)
            cost = cost * area
            profit_per_hectare = ((predicted_yield * 10 )* predicted_price) - cost
            profit = (profit_per_hectare * area)/100000 

            results.append({
            "Crop": crop,
            "Predicted Yield [metric tones/hec] ": predicted_yield,
            "Predicted Price [(₹)/Q] ": predicted_price,
            "Cost/hec [(₹)]": cost,
            "Expected Profit (₹ Lakh)": round(profit, 2) 
            })

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(by="Expected Profit (₹ Lakh)", ascending=False)

        # -------------------------------
        # Display Results
        # -------------------------------

        st.subheader("Crop Profit Prediction")

        st.dataframe(results_df)

        best_crop = results_df.iloc[0]["Crop"]

        st.success(f"Recommended Crop: {best_crop}")
        
        # ---------------------------
        # plot part
        # ---------------------------
        import matplotlib.pyplot as plt

        top10_df = results_df.head(10)

        fig, ax = plt.subplots()

        ax.bar(
            top10_df["Crop"],
            top10_df["Expected Profit (₹ Lakh)"]
        )

        ax.set_ylabel("Profit (₹ Lakh)")
        ax.set_title("Top 10 Most Profitable Crops")

        plt.xticks(rotation=45)
        st.pyplot(fig)




# ----------------------------------------------
# soil Base  crop 
# ----------------------------------------------
with tab2:
    st.title("Soil Based Crop Recommendation")

    state = st.selectbox(
        "Select State",
        soil_state_encoder.classes_
    )

    soil_type = st.selectbox(
        "Select Soil Type",
        soil_type_encoder.classes_
    )

    n = st.number_input("Nitrogen (N)", min_value=0.0)
    p = st.number_input("Phosphorus (P)", min_value=0.0)
    k = st.number_input("Potassium (K)", min_value=0.0)

    temperature = st.number_input("Temperature (°C)", min_value=0.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    ph = st.number_input("pH Value", min_value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
    # -------------------------------------
    # prediction button for soil
    # -------------------------------------

    if st.button("Recommend Crops (Soil Model)"):

        state_encoded = soil_state_encoder.transform([state])[0]
        soil_encoded = soil_type_encoder.transform([soil_type])[0]

        input_data = pd.DataFrame([{

            "STATE": state_encoded,
            "SOIL_TYPE": soil_encoded,
            "N_SOIL": n,
            "P_SOIL": p,
            "K_SOIL": k,
            "TEMPERATURE": temperature,
            "HUMIDITY": humidity,
            "ph": ph,
            "RAINFALL": rainfall

        }])

        probs = soil_model.predict_proba(input_data)[0]
        top_indices = np.argsort(probs)[::-1][:]
        crops = soil_crop_encoder.inverse_transform(top_indices)
        st.success("Top Suitable Crops Based on Soil")
        for i, crop in enumerate(crops, 1):
            st.write(f"{i}. {crop}")
            
'''

import streamlit as st
import pandas as pd
import numpy as np
import joblib

tab1, tab2 = st.tabs([
    "💰 Profit Based Recommendation",
    "🌱 Soil Based Recommendation"
])


# -------------------------------
# Load trained models
# -------------------------------

yield_model = joblib.load("models/yield_model.pkl")
price_model = joblib.load("models/price_model.pkl")

# Load encoders
le_state = joblib.load("models/state_encoder.pkl")
le_crop = joblib.load("models/crop_encoder.pkl")
le_season = joblib.load("models/season_encoder.pkl")

# Load soil model
soil_model           = joblib.load("models/soil_model.pkl")
soil_state_encoder   = joblib.load("models/soil_state_encoder.pkl")
soil_type_encoder    = joblib.load("models/soil_encoder.pkl")
soil_crop_encoder    = joblib.load("models/soil_crop_encoder.pkl")
soil_confusable_vegs = joblib.load("models/soil_confusable_vegs.pkl")
soil_scaler          = joblib.load("models/soil_scaler.pkl")

CONFIDENCE_THRESHOLD = 0.40

# Load cost dataset
cost_df = pd.read_csv("data/crop_cost.csv")
crop_cost = dict(zip(cost_df["Crop"], cost_df["Cost"]))

price_df = pd.read_csv("data/crop_price_avg.csv")
crop_avg_price = dict(zip(price_df["Crop"], price_df["Price"]))

# -------------------------------
# Streamlit UI
# -------------------------------
with tab1:
    st.title("AI-Based Crop Recommendation System")
    st.write("Predict the most profitable crop")

    state = st.selectbox("Select State", le_state.classes_)
    season = st.selectbox("Select Season", le_season.classes_)
    district = st.text_input("Type District")
    area = st.number_input("Area (hectares)", min_value=1.0)
    year = st.number_input("Year", min_value=2000, max_value=2035, value=2024)
    temperature = st.number_input("Temperature (°C)")
    rainfall = st.number_input("Rainfall (mm)")

    if st.button("Predict Best Crop"):

        state_encoded = le_state.transform([state])[0]
        season_encoded = le_season.transform([season])[0]

        candidate_crops = [crop for crop in le_crop.classes_]
        results = []

        for crop in candidate_crops:

            crop_encoded = le_crop.transform([crop])[0]

            yield_input = pd.DataFrame([{
                "State": state_encoded,
                "Crop": crop_encoded,
                "Season": season_encoded,
                "Area": area,
                "Rainfall": rainfall,
                "Temperature": temperature
            }])

            yield_log = yield_model.predict(yield_input)
            predicted_yield = max(0.1, np.expm1(yield_log)[0])

            price_input = pd.DataFrame([{
                "Year": year,
                "Month": 6,
                "State": state_encoded,
                "Crop": crop_encoded,
                "Temperature (Celsis)": temperature,
                "Rainfall in mm": rainfall
            }])

            price_log = price_model.predict(price_input)
            model_price = max(1, np.expm1(price_log)[0])
            dataset_price = crop_avg_price.get(crop, model_price)
            predicted_price = (model_price * 0.7) + (dataset_price * 0.3)

            default_cost = cost_df["Cost"].mean()
            cost = crop_cost.get(crop, default_cost)
            cost = cost * area
            profit_per_hectare = ((predicted_yield * 10) * predicted_price) - cost
            profit = (profit_per_hectare * area) / 100000

            results.append({
                "Crop": crop,
                "Predicted Yield [metric tones/hec] ": predicted_yield,
                "Predicted Price [(₹)/Q] ": predicted_price,
                "Cost/hec [(₹)]": cost,
                "Expected Profit (₹ Lakh)": round(profit, 2)
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="Expected Profit (₹ Lakh)", ascending=False)

        st.subheader("Crop Profit Prediction")
        st.dataframe(results_df)

        best_crop = results_df.iloc[0]["Crop"]
        st.success(f"Recommended Crop: {best_crop}")

        import matplotlib.pyplot as plt
        top10_df = results_df.head(10)
        fig, ax = plt.subplots()
        ax.bar(top10_df["Crop"], top10_df["Expected Profit (₹ Lakh)"])
        ax.set_ylabel("Profit (₹ Lakh)")
        ax.set_title("Top 10 Most Profitable Crops")
        plt.xticks(rotation=45)
        st.pyplot(fig)


# ----------------------------------------------
# Soil Based Crop Recommendation
# ----------------------------------------------
with tab2:
    st.title("Soil Based Crop Recommendation")

    soil_state = st.selectbox(
        "Select State",
        soil_state_encoder.classes_,
        key="soil_state"
    )

    soil_type = st.selectbox(
        "Select Soil Type",
        soil_type_encoder.classes_
    )

    n           = st.number_input("Nitrogen (N)",      min_value=0.0, key="soil_n")
    p           = st.number_input("Phosphorus (P)",    min_value=0.0, key="soil_p")
    k           = st.number_input("Potassium (K)",     min_value=0.0, key="soil_k")
    temperature = st.number_input("Temperature (°C)",  min_value=0.0, key="soil_temp")
    humidity    = st.number_input("Humidity (%)",      min_value=0.0, key="soil_hum")
    ph          = st.number_input("pH Value",          min_value=0.0, key="soil_ph")
    rainfall    = st.number_input("Rainfall (mm)",     min_value=0.0, key="soil_rain")
    crop_price  = st.number_input(
        "Expected Crop Price (₹/Quintal)",
        min_value=0.0,
        value=1500.0,
        help="Enter the average market price you expect for crops in your region"
    )

    if st.button("Recommend Crops (Soil Model)"):

        state_encoded = soil_state_encoder.transform([soil_state])[0]
        soil_encoded  = soil_type_encoder.transform([soil_type])[0]

        n_p_ratio  = n / (p + 1)
        n_k_ratio  = n / (k + 1)
        p_k_ratio  = p / (k + 1)
        npk_total  = n + p + k
        temp_humid = temperature * humidity / 100
        rain_humid = rainfall    * humidity / 100

        input_data = np.array([[
            state_encoded, soil_encoded,
            n, p, k,
            temperature, humidity, ph, rainfall,
            crop_price,
            n_p_ratio, n_k_ratio, p_k_ratio,
            npk_total, temp_humid, rain_humid
        ]])

        input_scaled = soil_scaler.transform(input_data)
        probs        = soil_model.predict_proba(input_scaled)[0]

        top_crop = soil_crop_encoder.classes_[np.argmax(probs)]
        top_conf = probs[np.argmax(probs)]

        if top_conf < 0.40 and top_crop in soil_confusable_vegs:
            # st.warning("⚠ Similar growing conditions detected for multiple crops.")
            st.write("All of the following vegetables grow well in these conditions:")
            cluster = []
            for veg in sorted(soil_confusable_vegs):
                if veg in soil_crop_encoder.classes_:
                    idx = np.where(soil_crop_encoder.classes_ == veg)[0][0]
                    cluster.append((veg, probs[idx]))
            cluster.sort(key=lambda x: -x[1])
            for i, (veg, conf) in enumerate(cluster, 1):
                st.write(f"{i}. **{veg}**  —  score: {conf*100:.1f}%")
        else:
            st.success("✅ Top Suitable Crops Based on Soil")
            for i, idx in enumerate(np.argsort(probs)[::-1][:5], 1):
                st.write(f"{i}. **{soil_crop_encoder.classes_[idx]}**  —  {probs[idx]*100:.1f}%")