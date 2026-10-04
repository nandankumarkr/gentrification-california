import pandas as pd
import numpy as np

INPUT = "data/clean_predictors.csv"
OUTPUT = "data/final_predictors.csv"

df = pd.read_csv(INPUT)

# -----------------------------------
# 1. Replace Census sentinel values
# -----------------------------------

sentinel = -666666666

df["median_monthly_cost_2010"] = (
    df["median_monthly_cost_2010"]
    .replace(sentinel, np.nan)
)

df["median_monthly_cost_2011"] = (
    df["median_monthly_cost_2011"]
    .replace(sentinel, np.nan)
)

# -----------------------------------
# 2. Treat zero price-asked values
#    as unavailable
# -----------------------------------

df["price_asked_2010"] = (
    df["price_asked_2010"].replace(0, np.nan)
)

df["price_asked_2011"] = (
    df["price_asked_2011"].replace(0, np.nan)
)

# -----------------------------------
# 3. Recalculate derived features
# -----------------------------------

df["renter_share_2011"] = (
    df["renter_units_2011"] /
    df["occupied_units_2011"].replace(0, np.nan)
)

df["housing_cost_change"] = (
    df["median_monthly_cost_2011"] -
    df["median_monthly_cost_2010"]
)

df["housing_cost_pct_change"] = (
    df["housing_cost_change"] /
    df["median_monthly_cost_2010"]
)

df["price_asked_change"] = (
    df["price_asked_2011"] -
    df["price_asked_2010"]
)

df["price_asked_pct_change"] = (
    df["price_asked_change"] /
    df["price_asked_2010"]
)

# -----------------------------------
# 4. Remove impossible infinite values
# -----------------------------------

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# -----------------------------------
# 5. Save
# -----------------------------------

df.to_csv(OUTPUT, index=False)

print("CLEAN DATASET CREATED")
print("Shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nSummary:")
print(df.describe().round(2))

