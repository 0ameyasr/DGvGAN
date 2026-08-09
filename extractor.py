import os
import json

REPORTS = "reports/raw"
API = "meta/api.json"
PROCESSED_DIR = "reports/processed"

json_reports = os.listdir(REPORTS)

with open(API, "r") as f:
    labels = json.loads(f.read())

labels = {val: int(key) for key, val in labels.items()}


def extract():
    print("Extracting all raw behavioral reports, processing...")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed = set(os.listdir(PROCESSED_DIR))

    def _gather_apis(node):
        if isinstance(node, dict):
            if "api" in node:
                val = node["api"]
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            yield item
                elif isinstance(val, str):
                    yield val

            for v in node.values():
                yield from _gather_apis(v)

        elif isinstance(node, list):
            for item in node:
                yield from _gather_apis(item)

    for report in json_reports:
        md5 = report[:-5]

        if f"{md5}.json" in processed:
            print(f"Skipped sample {md5} (Already processed)")
            continue

        with open(f"{REPORTS}/{report}", "r") as f:
            data = json.load(f)

        raw_api_calls = list(_gather_apis(data))

        if not raw_api_calls:
            print(f"Skipped sample {md5} (No API signatures found)")
            continue

        mapped_calls = [labels.get(call, None) for call in raw_api_calls][:100]

        sample = {"md5": md5, "api": mapped_calls}

        with open(f"{PROCESSED_DIR}/{md5}.json", "w") as f:
            json.dump(sample, f, indent=2)

    print("DONE.")


if __name__ == "__main__":
    extract()
