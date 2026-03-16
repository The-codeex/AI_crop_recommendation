import pandas as pd

df = pd.read_csv("data/Cost_of_Cultivation.csv")

# Check columns
print(df.columns)

# Select correct columns
df = df[["Crop Name (crop_name)", "Production Cost C2 (prod_cost_c2)"]]

# Rename columns for simplicity
df = df.rename(columns={
    "Crop Name (crop_name)": "Crop",
    "Production Cost C2 (prod_cost_c2)": "Cost"
})

# Remove missing values
df = df.dropna()

# Average cost per crop
crop_cost = df.groupby("Crop")["Cost"].mean().round(0)

print(crop_cost)

# Save cost(ruppes/ acers) file 
crop_cost.to_csv("data/crop_cost.csv")

print("Crop cost file created successfully")