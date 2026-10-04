import pandas as pd
import glob
import os

# CPI-U annual averages
CPI_2012 = 229.594
CPI_2016 = 240.007

# Correct S2503 variables
COL_2012 = "S2503_C01_028E"
COL_2016 = "S2503_C01_024E"


def load_year(year, column):
    files = glob.glob(f"data/raw/outcome/{year}/*.csv")

    dfs = []

    for file in files:
        df = pd.read_csv(file)

        if "GEO_ID" not in df.columns:
            continue

        temp = df[["GEO_ID", column]].copy()
        temp["tract_id"] = (
            temp["GEO_ID"]
            .astype(str)
            .str.replace("1400000US", "", regex=False)
        )

        temp[column] = pd.to_numeric(
            temp[column], errors="coerce"
        )

        # Census missing-value sentinel
        temp[column] = temp[column].replace(
            -666666666, pd.NA
        )

        dfs.append(temp[["tract_id", column]])

    return pd.concat(dfs, ignore_index=True)


print("Loading 2012...")
d12 = load_year(2012, COL_2012)

print("Loading 2016...")
d16 = load_year(2016, COL_2016)

# Rename
d12 = d12.rename(columns={
    COL_2012: "housing_cost_2012"
})

d16 = d16.rename(columns={
    COL_2016: "housing_cost_2016"
})

# Merge
target = d12.merge(
    d16,
    on="tract_id",
    how="inner"
)

# Inflation-adjust 2012 values into 2016 dollars
target["housing_cost_2012_real_2016"] = (
    target["housing_cost_2012"]
    * CPI_2016
    / CPI_2012
)

# Change in inflation-adjusted housing cost
target["housing_cost_change"] = (
    target["housing_cost_2016"]
    - target["housing_cost_2012_real_2016"]
)

# Percentage change
target["housing_cost_pct_change"] = (
    target["housing_cost_change"]
    / target["housing_cost_2012_real_2016"]
)

# Binary target:
# 1 = inflation-adjusted housing cost increased
# 0 = otherwise
target["gentrification"] = pd.NA

valid = target["housing_cost_change"].notna()

target.loc[valid, "gentrification"] = (
    target.loc[valid, "housing_cost_change"] > 0
).astype(int)

# Save
target = target.dropna(
    subset=[
        "housing_cost_2012",
        "housing_cost_2016",
        "gentrification"
    ]
)
os.makedirs("data", exist_ok=True)

target.to_csv(
    "data/housing_cost_target.csv",
    index=False
)

print("\nTarget dataset:")
print("Shape:", target.shape)

print("\nClass distribution:")
print(target["gentrification"].value_counts())

print("\nClass percentages:")
print(
    target["gentrification"]
    .value_counts(normalize=True)
    .mul(100)
)

print("\nMissing values:")
print(target.isna().sum())

print("\nFirst rows:")
print(target.head())

