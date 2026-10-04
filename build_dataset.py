import pandas as pd

BASE = "data/raw/"

# Load data
s2502 = pd.read_csv(BASE + "2011_S2502.csv")
s2503_2010 = pd.read_csv(BASE + "2010_S2503.csv")
s2503_2011 = pd.read_csv(BASE + "2011_S2503.csv")
b25085_2010 = pd.read_csv(BASE + "2010_B25085.csv")
b25085_2011 = pd.read_csv(BASE + "2011_B25085.csv")


# Create common Census tract ID
def add_id(df):
    df["tract_id"] = (
        df["state"].astype(str).str.zfill(2) + "_" +
        df["county"].astype(str).str.zfill(3) + "_" +
        df["tract"].astype(str)
    )
    return df


for df in [
    s2502,
    s2503_2010,
    s2503_2011,
    b25085_2010,
    b25085_2011
]:
    add_id(df)


# -----------------------------
# 2011 S2502
# -----------------------------

s2502_small = s2502[
    [
        "tract_id",
        "S2502_C01_001E",
        "S2502_C02_001E",
        "S2502_C03_001E"
    ]
].copy()

s2502_small.rename(columns={
    "S2502_C01_001E": "occupied_units_2011",
    "S2502_C02_001E": "owner_units_2011",
    "S2502_C03_001E": "renter_units_2011"
}, inplace=True)


# -----------------------------
# S2503 — median monthly
# housing costs
# -----------------------------

s2503_2010_small = s2503_2010[
    ["tract_id", "S2503_C01_028E"]
].copy()

s2503_2010_small.rename(
    columns={
        "S2503_C01_028E": "median_monthly_cost_2010"
    },
    inplace=True
)

s2503_2011_small = s2503_2011[
    ["tract_id", "S2503_C01_028E"]
].copy()

s2503_2011_small.rename(
    columns={
        "S2503_C01_028E": "median_monthly_cost_2011"
    },
    inplace=True
)


# -----------------------------
# B25085 — price asked
# -----------------------------

b25085_2010_small = b25085_2010[
    ["tract_id", "B25085_001E"]
].copy()

b25085_2010_small.rename(
    columns={
        "B25085_001E": "price_asked_2010"
    },
    inplace=True
)

b25085_2011_small = b25085_2011[
    ["tract_id", "B25085_001E"]
].copy()

b25085_2011_small.rename(
    columns={
        "B25085_001E": "price_asked_2011"
    },
    inplace=True
)


# -----------------------------
# Merge
# -----------------------------

df = s2502_small

df = df.merge(
    s2503_2010_small,
    on="tract_id",
    how="inner"
)

df = df.merge(
    s2503_2011_small,
    on="tract_id",
    how="inner"
)

df = df.merge(
    b25085_2010_small,
    on="tract_id",
    how="inner"
)

df = df.merge(
    b25085_2011_small,
    on="tract_id",
    how="inner"
)


# -----------------------------
# Derived features
# -----------------------------

df["renter_share_2011"] = (
    df["renter_units_2011"] /
    df["occupied_units_2011"]
)

df["housing_cost_change"] = (
    df["median_monthly_cost_2011"] -
    df["median_monthly_cost_2010"]
)

df["housing_cost_pct_change"] = (
    df["housing_cost_change"] /
    df["median_monthly_cost_2010"].replace(0, pd.NA)
)

df["price_asked_change"] = (
    df["price_asked_2011"] -
    df["price_asked_2010"]
)

df["price_asked_pct_change"] = (
    df["price_asked_change"] /
    df["price_asked_2010"].replace(0, pd.NA)
)


# -----------------------------
# Save
# -----------------------------

df.to_csv("data/clean_predictors.csv", index=False)

print("\nDATASET CREATED")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isna().sum())
