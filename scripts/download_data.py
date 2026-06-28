import os
import requests

os.makedirs("tml-data", exist_ok=True)

files = requests.get(
    "https://stats.tennismylife.org/api/data-files"
).json()["files"]

for file in files:
    print(f"Downloading {file['name']}")
    data = requests.get(file["url"])
    with open(f"tml-data/{file['name']}", "wb") as f:
        f.write(data.content)

print("Done")
