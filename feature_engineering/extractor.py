import json

def extract_features(report_path):
    with open(report_path, "r") as f:
        data = json.load(f)

    features = []

    # simple numeric features
    features.append(len(data["behavior"]["processes"]))
    features.append(len(data["network"]["domains"]))
    features.append(len(data["network"]["http"]))

    return features