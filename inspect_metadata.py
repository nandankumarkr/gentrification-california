import requests
import pandas as pd

YEAR = 2011
TABLE = "S2502"

url = f"https://api.census.gov/data/{YEAR}/acs/acs5/subject/groups/{TABLE}.json"

data = requests.get(url).json()
variables = data["variables"]

rows = []

for name, info in variables.items():

    if not name.startswith(TABLE + "_"):
        continue

    # Keep only estimate variables
    if not name.endswith("E"):
        continue

    rows.append({
        "variable": name,
        "label": info.get("label", ""),
        "concept": info.get("concept", "")
    })

df = pd.DataFrame(rows)

df = df.sort_values("variable")

df.to_csv("data/raw/2011_S2502_metadata.csv", index=False)

print("Saved:", len(df), "estimate variables")
print()
print(df.to_string(index=False))
