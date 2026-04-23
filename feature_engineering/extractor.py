import json

def extract_features_from_report(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    api_calls = data.get("api", [])

    if not isinstance(api_calls, list):
        api_calls = []

    # ✅ Basic features
    # total_calls = len(api_calls)
    # unique_calls = len(set(api_calls))
    # max_call = max(api_calls) if api_calls else 0

    # # Optional extra features (better ML performance)
    # avg_call = sum(api_calls) / total_calls if total_calls > 0 else 0

    return api_calls
