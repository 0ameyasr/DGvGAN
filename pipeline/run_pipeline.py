import os

from feature_engineering.extractor import extract_features_from_report
from pipeline.predictor import (
    predict_sgan,
    predict_dgcnn,
    predict_cnn,
    predict_hybrid,
)
from models.ensemble import ensemble

print("✅ run_pipeline.py is being executed")


def safe_predict(func, features, name):
    try:
        return func(features)
    except Exception as e:
        print(f"⚠️ {name} failed:", e)
        return None


def run():
    # REPORTS_DIR = r"C:\DGvGAN\sandbox\reports\processed"
    REPORTS_DIR = "sandbox/reports/processed"

    print("\n📂 REPORTS_DIR:", REPORTS_DIR)
    print("📂 EXISTS?:", os.path.exists(REPORTS_DIR))

    if not os.path.exists(REPORTS_DIR):
        print("❌ Reports directory not found!")
        return

    files = [
        f for f in os.listdir(REPORTS_DIR)
        if os.path.isfile(os.path.join(REPORTS_DIR, f))
        and f.lower().endswith(".json")
    ]

    print("📄 JSON FILES:", files)

    if not files:
        print("❌ No JSON files found!")
        return

    for file in files:
        path = os.path.join(REPORTS_DIR, file)

        try:
            features = extract_features_from_report(path)

            # 🔥 Run ALL models safely
            p1 = safe_predict(predict_sgan, features, "SGAN")
            p2 = safe_predict(predict_dgcnn, features, "DGCNN")
            p3 = safe_predict(predict_cnn, features, "CNN")
            p4 = safe_predict(predict_hybrid, features, "HYBRID")


            # keep only working ones
            preds = [p for p in [p1, p2, p3, p4] if p is not None]

            if not preds:
                print("❌ All models failed for:", file)
                continue

            # 🔥 Ensemble (fallback to average if needed)
            try:
                final = ensemble(preds)
            except:
                final = sum(preds) / len(preds)

            label = "MALWARE" if final > 0.5 else "BENIGN"

            print("\n==============================")
            print("📄 File:", file)
            print("📊 Features:", features[:10], "...")  # avoid huge print
            print("🤖 SGAN:", p1)
            print("🤖 DGCNN:", p2)
            print("🤖 CNN:", p3)
            print("🤖 HYBRID:", p4)
            print("🎯 Final Score:", round(final, 3))
            print("🚨 Prediction:", label)
            print("==============================")

        except Exception as e:
            print(f"\n❌ Error processing {file}: {e}")


if __name__ == "__main__":
    run()
