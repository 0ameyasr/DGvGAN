import os
from feature_engineering.extractor import extract_features_from_report
from pipeline.predictor import (
    predict_sgan,
    predict_dgcnn,
    predict_cnn,
    predict_hybrid
)
from models.ensemble import ensemble

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "sandbox", "reports")

def run():
    if not os.path.exists(REPORTS_DIR):
        print("❌ Reports folder not found:", REPORTS_DIR)
        return

    files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".json")]

    if not files:
        print("❌ No JSON files found!")
        return

    for file in files:
        path = os.path.join(REPORTS_DIR, file)

        features = extract_features_from_report(path)

        p1 = predict_sgan(features)
        p2 = predict_dgcnn(features)
        p3 = predict_cnn(features)
        p4 = predict_hybrid(features)

        final = ensemble([p1, p2, p3, p4])

        label = "MALWARE" if final > 0.5 else "BENIGN"

        print("\n==============================")
        print("📄 File:", file)
        print("📊 Features:", features)
        print("🤖 Predictions:", p1, p2, p3, p4)
        print("🎯 Final Score:", round(final, 3))
        print("🚨 Prediction:", label)
        print("==============================")

if __name__ == "__main__":
    run()