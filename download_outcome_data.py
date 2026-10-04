import requests
import pandas as pd
import os
import time
from getpass import getpass

API_KEY = getpass("Enter your Census API key: ")

YEARS = [2012, 2013, 2014, 2015, 2016]
TABLE = "S2503"

os.makedirs("data/raw/outcome", exist_ok=True)

# Get California counties
county_url = (
    "https://api.census.gov/data/2012/acs/acs5/subject"
    "?get=NAME&for=county:*&in=state:06"
    f"&key={API_KEY}"
)

response = requests.get(county_url, timeout=60)
response.raise_for_status()

counties = response.json()
county_list = [row[2] for row in counties[1:]]

print(f"Found {len(county_list)} California counties")

for year in YEARS:

    year_dir = f"data/raw/outcome/{year}"
    os.makedirs(year_dir, exist_ok=True)

    print(f"\n========== {year} ==========")

    for i, county in enumerate(county_list, start=1):

        output = f"{year_dir}/{county}.csv"

        if os.path.exists(output):
            print(f"[{i}/58] {county} already exists")
            continue

        url = (
            f"https://api.census.gov/data/{year}/acs/acs5/subject"
            f"?get=NAME,group({TABLE})"
            f"&for=tract:*"
            f"&in=state:06"
            f"&in=county:{county}"
            f"&key={API_KEY}"
        )

        for attempt in range(3):

            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()

                data = r.json()

                df = pd.DataFrame(
                    data[1:],
                    columns=data[0]
                )

                df.to_csv(output, index=False)

                print(
                    f"[{i}/58] {county} -> "
                    f"{len(df)} tracts"
                )

                break

            except Exception as e:

                print(
                    f"[{i}/58] {county} "
                    f"attempt {attempt + 1}/3 failed: {e}"
                )

                if attempt < 2:
                    time.sleep(5)

        time.sleep(0.5)

print("\nDOWNLOAD COMPLETE")
