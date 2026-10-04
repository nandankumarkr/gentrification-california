import pandas as pd

df = pd.read_csv("data/final_dataset.csv")

# Features available before the prediction period
features = [
    "occupied_units_2011",
    "owner_units_2011",
    "renter_units_2011",
    "median_monthly_cost_2010",
    "median_monthly_cost_2011",
    "renter_share_2011",
    "housing_cost_change_x",
    "housing_cost_pct_change_x",
]

target = "gentrification"

clean = df[features + [target]].copy()

print("Before cleaning:")
print(clean.shape)

print("\nMissing values:")
print(clean.isna().sum())

# Remove rows with missing predictor values
clean = clean.dropna()

print("\nAfter cleaning:")
print(clean.shape)

print("\nMissing values after cleaning:")
print(clean.isna().sum())

print("\nTarget distribution:")
print(clean[target].value_counts())

print("\nTarget percentage:")
print(clean[target].value_counts(normalize=True) * 100)

clean.to_csv(
    "data/analysis_dataset.csv",
    index=False
)

print("\nSaved:")
print("data/analysis_dataset.csv")
