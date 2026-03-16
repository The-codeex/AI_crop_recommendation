import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ================================
# Load Models
# ================================

yield_model = joblib.load("models/yield_model.pkl")
price_model = joblib.load("models/price_model.pkl")
soil_model = joblib.load("models/soil_model.pkl")

print("\nModels Loaded Successfully")

# ================================
# Load Dataset
# ================================

df = pd.read_csv("data/final_dataset.csv")

# ================================
# Prepare Yield Prediction Data
# ================================

X = df[[
    "State",
    "Crop",
    "Season",
    "Area",
    "Rainfall",
    "Temperature"
]]

y = df["Yield"]

pred = yield_model.predict(X)

print("\n===== Yield Model Evaluation =====")

r2 = r2_score(y, pred)
mae = mean_absolute_error(y, pred)
mse = mean_squared_error(y, pred)
rmse = np.sqrt(mse)
std = np.std(pred)

print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("Standard Deviation:", std)

# ================================
# Soil Classification Evaluation
# ================================

soil_df = pd.read_csv("data/soil_crop_dataset.csv")

X_soil = soil_df[[
    "STATE",
    "SOIL_TYPE",
    "N_SOIL",
    "P_SOIL",
    "K_SOIL",
    "TEMPERATURE",
    "HUMIDITY",
    "ph",
    "RAINFALL"
]]

y_soil = soil_df["CROP"]

pred_soil = soil_model.predict(X_soil)

print("\n===== Soil Model Evaluation =====")

acc = accuracy_score(y_soil, pred_soil)
precision = precision_score(y_soil, pred_soil, average="weighted")
recall = recall_score(y_soil, pred_soil, average="weighted")
f1 = f1_score(y_soil, pred_soil, average="weighted")

print("Accuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_soil, pred_soil))