'''
# ---------------- old code-----------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

import joblib

df = pd.read_csv("data/final_dataset.csv")

# Remove missing values
df = df.dropna()

# Safety check
print("Missing values:")
print(df.isnull().sum())

df = df.drop_duplicates()

# Sample dataset to reduce size (optional but recommended)
# df = df.sample(100000, random_state=42)

print("Dataset size after sampling:", df.shape)

# Apply log transformation to target variable
df["Yield"] = np.log1p(df["Yield"])


print(df.head())

le_state = LabelEncoder()
le_crop = LabelEncoder()
le_season = LabelEncoder()

df["State"] = le_state.fit_transform(df["State"])
df["Crop"] = le_crop.fit_transform(df["Crop"])
df["Season"] = le_season.fit_transform(df["Season"])

df = df.drop(columns=["District_Name"])

features = [
    "State",
    "Crop",
    "Season",
    "Area",
    "Rainfall",
    "Temperature"
]

X = df[features]
y = df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {

    # "Linear Regression": LinearRegression(),

    # "Random Forest": RandomForestRegressor(
    #      n_estimators=500,
    # max_depth=25,
    # min_samples_split=5,
    # min_samples_leaf=2,
    # n_jobs=-1,
    # random_state=42
    # ),

    # "Gradient Boosting": GradientBoostingRegressor(),

    "XGBoost": XGBRegressor(
        n_estimators=800,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
    )
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results[name] = (rmse, r2)

    print(f"{name}")
    print("RMSE:", rmse)
    print("R2:", r2)
    print("---------------------")
    
for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2"
    )

    print(name, "CV R2:", scores.mean()) 
    
best_model_name = max(results, key=lambda x: results[x][1])
best_model = models[best_model_name]

print("Best Model:", best_model_name)

joblib.dump(best_model, "models/yield_model.pkl")  

print(df["Crop"].nunique())
    
 '''

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

import joblib
from sklearn.model_selection import GroupKFold


# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("data/final_dataset.csv")

print("Original dataset size:", df.shape)


# -------------------------------
# Data Cleaning
# -------------------------------

df = df.dropna()
df = df.drop_duplicates()

print("Dataset after cleaning:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# -------------------------------
# Log transform target variable
# -------------------------------

df["Yield"] = np.log1p(df["Yield"])


# -------------------------------
# Encode categorical variables
# -------------------------------

le_state = LabelEncoder()
le_crop = LabelEncoder()
le_season = LabelEncoder()

df["State"] = le_state.fit_transform(df["State"])
df["Crop"] = le_crop.fit_transform(df["Crop"])
df["Season"] = le_season.fit_transform(df["Season"])


# Drop high-cardinality column
df = df.drop(columns=["District_Name"])


# -------------------------------
# Feature Selection
# -------------------------------

features = [
    "State",
    "Crop",
    "Season",
    "Area",
    "Rainfall",
    "Temperature"
]

X = df[features]
y = df["Yield"]


# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# Model Definition
# -------------------------------

models = {

    "XGBoost": XGBRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    )
}


# -------------------------------
# Model Training
# -------------------------------

results = {}
trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    trained_models[name] = model

    predictions = model.predict(X_test)

    # -------------------------------
# Model Training
# -------------------------------

results = {}
trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    trained_models[name] = model

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results[name] = (rmse, r2)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results[name] = (rmse, r2)

    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)


# -------------------------------
# Cross Validation
# -------------------------------

print("\nGroup Cross Validation Results")

gkf = GroupKFold(n_splits=5)

for name, model in models.items():

    scores = cross_val_score(
    model,
    X,
    y,
    cv=gkf.split(X, y, groups=df["State"]),
    scoring="r2"
    )

    print(name, "Group CV R2:", scores.mean())


# -------------------------------
# Select Best Model
# -------------------------------

best_model_name = max(results, key=lambda x: results[x][1])
best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)


# -------------------------------
# Save Model
# -------------------------------

joblib.dump(best_model, "models/yield_model.pkl")

print("Model saved successfully at models/yield_model.pkl")

# -------------------------------
# Save Encoders
# -------------------------------

joblib.dump(le_state, "models/state_encoder.pkl")
joblib.dump(le_crop, "models/crop_encoder.pkl")
joblib.dump(le_season, "models/season_encoder.pkl")

print("Encoders saved successfully")


# -------------------------------
# Dataset Information
# -------------------------------

print("\nTotal unique crops:", df["Crop"].nunique())
    