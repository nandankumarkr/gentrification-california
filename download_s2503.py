import requests
import pandas as pd
import os
import time
from getpass import getpass

API_KEY = getpass("Enter your Census API key: ")

YEAR = 2011
TABLE = "S2503"

os.makedirs("data/raw/s2503_counties", exist_ok=True)

# Get California counties
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
county_list = [row[2] for row in counties[1:]]

print(f"Found {len(county_list)} counties")

for i, county in enumerate(county_list, start=1):

    output = f"data/raw/s2503_counties/{county}.csv"

    # Don't download again if already completed
    if os.path.exists(output):
        print(f"[{i}/58] County {county} already downloaded")
        continue

    print(f"[{i}/58] Downloading county {county}...")

    url = (
        f"https://api.census.gov/data/{YEAR}/acs/acs5/subject"
        f"?get=NAME,group({TABLE})"
        f"&for=tract:*"
        f"&in=state:06"
        f"&in=county:{county}"
        f"&key={API_KEY}"
    )

    success = False

    for attempt in range(3):

        try:
            response = requests.get(url, timeout=180)
            response.raise_for_status()

            data = response.json()

            df = pd.DataFrame(
                data[1:],
                columns=data[0]
            )

            df.to_csv(output, index=False)

            print(f"    SUCCESS -> {len(df)} tracts")
            success = True
            break

        except Exception as e:

            print(
                f"    Attempt {attempt + 1}/3 failed: {e}"
            )

            if attempt < 2:
                print("    Retrying in 5 seconds...")
                time.sleep(5)

    if not success:
        print(f"    FAILED county {county}")

    time.sleep(1)

print()
print("County downloads finished.")
