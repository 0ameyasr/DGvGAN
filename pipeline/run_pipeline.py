import os
from feature_engineering.extractor import extract_features_from_report
from pipeline.predictor import (
    predict_sgan,
    predict_dgcnn,
    predict_cnn,
    predict_hybrid,
    predict_gat
)
from models.ensemble import ensemble
import extractor
import pandas

def safe_predict(func, features, name):
    try:
        return func(features)
    except Exception as e:
        print(f"{name} failed:", e)
        return None


def run():
    results_list = []
    REPORTS_DIR = "sandbox/reports/processed"
    
    if not os.path.exists(REPORTS_DIR):
        print("Reports directory not found!")
        return

    files = [
        f for f in os.listdir(REPORTS_DIR)
        if os.path.isfile(os.path.join(REPORTS_DIR, f))
        and f.lower().endswith(".json")
    ]

    if not files:
        print("No JSON files found!")
        return

    for file in files:
        path = os.path.join(REPORTS_DIR, file)

        try:
            features = extract_features_from_report(path)

            tp1, p1 = safe_predict(predict_sgan, features, "SGAN")
            tp2, p2 = safe_predict(predict_dgcnn, features, "DGCNN")
            tp3, p3 = safe_predict(predict_cnn, features, "CNN")
            tp4, p4 = safe_predict(predict_hybrid, features, "HYBRID")
            tp5, p5 = safe_predict(predict_gat, features, "GAT")

            preds = [p for p in [p1, p2, p3, p4, p5] if p is not None]

            if not preds:
                print("All models failed for:", file)
                continue

            try:
                final = ensemble(preds)
            except:
                final = sum(preds) / len(preds)

            label = "MALWARE" if final > 0.5 else "BENIGN"

            results = {}
            results['sample'] = file
            results['cnn'] = round(p3,3)
            results['sgan'] = round(p1,3)
            results['dgcnn'] = round(p2,3)
            results['hybrid'] = round(p4,3)
            results['gat'] = round(p5,3)
            results['final'] = round(final,3)
            results['malware'] = label
            results['t_cnn'] = round(tp3,9)
            results['t_sgan'] = round(tp1,9)
            results['t_dgcnn'] = round(tp2,9)
            results['t_hybrid'] = round(tp4,9)
            results['t_gat'] = round(tp5,9)
            results_list.append(results)
            
        except Exception as e:
            print(f"\nError processing {file}: {e}")
    
    print(pandas.DataFrame(results_list))


if __name__ == "__main__":
    extractor.extract()
    print()
    run()
