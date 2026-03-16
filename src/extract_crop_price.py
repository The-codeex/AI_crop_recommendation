
import pandas as pd

df = pd.read_csv("data/price_data.csv")
# df = pd.read_csv("data/soil_crop_dataset.csv")

# Check columns
print(df.columns)

# Rename for simplicity
df = df.rename(columns={
    "Wholesale_Price[Rs. Per Quintal]": "Price"
    # "CROP_PRICE": "Price",
    # "CROP" : "Crop"
})

# Average price per crop
crop_price = df.groupby("Crop")["Price"].mean().round(2)

print(crop_price)

# Save
crop_price.to_csv("data/crop_price_avg.csv")

print("Crop price file created successfully")