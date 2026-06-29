import os
import requests

os.makedirs("tml-data", exist_ok=True)

files = requests.get(
    "https://stats.tennismylife.org/api/data-files"
).json()["files"]

for file in files:
    name = file["name"]
    print(f"Downloading {name}")
    data = requests.get(file["url"])
    out = f"tml-data/{name}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(data.content)

print("Done")
