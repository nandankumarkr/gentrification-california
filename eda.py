import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/final_predictors.csv")

os.makedirs("results/eda", exist_ok=True)

# --------------------------------
# 1. Median monthly housing cost
# --------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["median_monthly_cost_2010"].dropna(),
    bins=40
)

plt.xlabel("Median Monthly Housing Cost ($)")
plt.ylabel("Number of Census Tracts")
plt.title("Distribution of Median Monthly Housing Cost — 2010")

plt.tight_layout()
plt.savefig("results/eda/housing_cost_2010.png", dpi=200)
plt.close()


# --------------------------------
# 2. Housing cost percentage change
# --------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["housing_cost_pct_change"].dropna(),
    bins=50
)

plt.xlabel("Housing Cost Percentage Change (2010–2011)")
plt.ylabel("Number of Census Tracts")
plt.title("Change in Median Monthly Housing Cost")

plt.tight_layout()
plt.savefig("results/eda/housing_cost_change.png", dpi=200)
plt.close()


# --------------------------------
# 3. Renter share
# --------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["renter_share_2011"].dropna(),
    bins=40
)

plt.xlabel("Renter Share")
plt.ylabel("Number of Census Tracts")
plt.title("Distribution of Renter Share — 2011")

plt.tight_layout()
plt.savefig("results/eda/renter_share.png", dpi=200)
plt.close()


# --------------------------------
# 4. Housing cost: 2010 vs 2011
# --------------------------------

plot_df = df[
    ["median_monthly_cost_2010",
     "median_monthly_cost_2011"]
].dropna()

plt.figure(figsize=(7, 7))

plt.scatter(
    plot_df["median_monthly_cost_2010"],
    plot_df["median_monthly_cost_2011"],
    alpha=0.3,
    s=10
)

plt.xlabel("Median Monthly Cost — 2010")
plt.ylabel("Median Monthly Cost — 2011")
plt.title("Housing Costs: 2010 vs 2011")

plt.tight_layout()
plt.savefig("results/eda/housing_cost_2010_vs_2011.png", dpi=200)
plt.close()


print("EDA COMPLETE")
print("\nFiles created:")

for f in os.listdir("results/eda"):
    print(" -", f)

print("\nKey statistics:")
print(
    df[
        [
            "median_monthly_cost_2010",
            "median_monthly_cost_2011",
            "housing_cost_pct_change",
            "renter_share_2011"
        ]
    ].describe().round(3)
)
