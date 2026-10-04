import pandas as pd

# Load datasets
predictors = pd.read_csv("data/clean_predictors.csv")
target = pd.read_csv("data/housing_cost_target.csv")
def normalize_predictor_id(x):
    parts = str(x).split("_")

    state = parts[0].zfill(2)
    county = parts[1].zfill(3)
    tract = str(int(parts[2])).zfill(6)

    return state + county + tract


predictors["tract_id"] = (
    predictors["tract_id"]
    .apply(normalize_predictor_id)
)

target["tract_id"] = (
    target["tract_id"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(11)
)

print("Predictors:", predictors.shape)
print("Target:", target.shape)

# Merge using Census tract ID
final = predictors.merge(
    target[
        [
            "tract_id",
            "housing_cost_2012",
            "housing_cost_2016",
            "housing_cost_change",
            "housing_cost_pct_change",
            "gentrification"
        ]
    ],
    on="tract_id",
    how="inner"
)

print("\nFinal dataset:")
print("Shape:", final.shape)

print("\nMissing values:")
print(final.isna().sum())

print("\nClass distribution:")
print(final["gentrification"].value_counts())

print("\nClass percentages:")
print(
    final["gentrification"]
    .value_counts(normalize=True)
    .mul(100)
)

# Save
final.to_csv("data/final_dataset.csv", index=False)

print("\nSaved to:")
print("data/final_dataset.csv")

