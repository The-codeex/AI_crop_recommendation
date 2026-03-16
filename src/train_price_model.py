import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

import joblib


# ------------------------------
# Load dataset
# ------------------------------

# df = pd.read_csv("data/price_data.csv")
df = pd.read_csv("data/price_data.csv")

# remove hidden spaces in column names
df.columns = df.columns.str.strip()

print(df.columns)

print("Original dataset shape:", df.shape)


# ------------------------------
# Data Cleaning
# ------------------------------

df = df.dropna()
df = df.drop_duplicates()

print("Dataset after cleaning:", df.shape)


# ------------------------------
# Rename columns for convenience
# ------------------------------

df = df.rename(columns={
    "Wholesale_Price[Rs. Per Quintal]": "Price"
    
    
})

# df = df.rename(columns={
#     "CROP_PRICE": "Price",
#      "STATE" : "State",
#      "CROP" : "Crop"
     
    
    
# })


# ------------------------------
# Log transform target variable
# ------------------------------

df["Price"] = np.log1p(df["Price"])


# ------------------------------
# Encode categorical variables
# ------------------------------

le_state = LabelEncoder()
le_crop = LabelEncoder()

df["State"] = le_state.fit_transform(df["State"])
df["Crop"] = le_crop.fit_transform(df["Crop"])



# ------------------------------
# Feature Selection
# ------------------------------

features = [
    "Year",
    "Month",
    "State",
    "Crop",
    "Temperature (Celsis)",
    "Rainfall in mm"
    # "TEMPERATURE",
    # "RAINFALL"
]

X = df[features]
y = df["Price"]


# ------------------------------
# Train Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ------------------------------
# Model Definition
# ------------------------------

model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)


# ------------------------------
# Model Training
# ------------------------------

print("\nTraining Price Prediction Model...")

model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nModel Performance")
print("RMSE:", rmse)
print("R2 Score:", r2)


# ------------------------------
# Cross Validation
# ------------------------------

scores = cross_val_score(
    model,
    X,
    y,
    cv=3,
    scoring="r2"
)

print("\nCross Validation R2:", scores.mean())


# ------------------------------
# Save model
# ------------------------------

joblib.dump(model, "models/price_model.pkl")

print("\nPrice model saved at models/price_model.pkl")