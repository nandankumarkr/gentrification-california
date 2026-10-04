import os
import requests
import pandas as pd
from getpass import getpass
import time

API_KEY = getpass("Enter your Census API key: ")

YEAR = 2011
TABLE = "S2502"

os.makedirs("data/raw", exist_ok=True)

# Get all California counties
county_url = (
    f"https://api.census.gov/data/{YEAR}/acs/acs5/subject"
    f"?get=NAME"
    f"&for=county:*"
    f"&in=state:06"
    f"&key={API_KEY}"
)

print("Getting California counties...")

response = requests.get(county_url, timeout=60)
response.raise_for_status()

counties = response.json()

# Skip header row
county_list = [row[2] for row in counties[1:]]

print(f"Found {len(county_list)} counties")

all_data = []

for i, county in enumerate(county_list, start=1):

    print(f"Downloading county {i}/{len(county_list)}: {county}")

    url = (
        f"https://api.census.gov/data/{YEAR}/acs/acs5/subject"
        f"?get=NAME,group({TABLE})"
        f"&for=tract:*"
        f"&in=state:06"
        f"&in=county:{county}"
        f"&key={API_KEY}"
    )

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        data = response.json()

        df = pd.DataFrame(data[1:], columns=data[0])

        all_data.append(df)

        print(f"  -> {len(df)} tracts")

    except Exception as e:
        print(f"  ERROR: {e}")

    time.sleep(0.2)

# Combine all counties
print("Combining counties...")

final_df = pd.concat(all_data, ignore_index=True)

output = f"data/raw/{YEAR}_{TABLE}.csv"

final_df.to_csv(output, index=False)

print()
print("DOWNLOAD COMPLETE")
print("-----------------")
print("Rows:", len(final_df))
print("Columns:", len(final_df.columns))
print("Saved to:", output)
