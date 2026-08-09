# Do anything in this file. Temporary code for fast sanity checks/experiments,etc.

import json

with open("meta/vt_cache_benign.json", "r") as f:
    data = json.loads(f.read())

dates = []
for entry in data:
    entry_data = data[entry]
    date = entry_data["First Seen on VirusTotal"]
    dates.append(int(date[:4]))

print(f"Sample range: {min(dates)}-{max(dates)}")
print(
    f"Samples w.r.t 2019: -{len(list(filter(lambda x: x <= 2019,dates)))} +{len(list(filter(lambda x: x > 2019,dates)))}"
)
