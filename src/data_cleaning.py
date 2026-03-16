import pandas as pd

yield_df = pd.read_csv("data/yield_data.csv")

yield_df = yield_df.dropna()

yield_df = yield_df[yield_df["Area"] > 0]

yield_df.to_csv("data/clean_yield_data.csv", index=False)

print("Cleaning Complete")