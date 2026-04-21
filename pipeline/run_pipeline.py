from feature_engineering.extractor import extract_features
from pipeline.predictor import (
    predict_sgan,
    predict_dgcnn,
    predict_cnn,
    predict_hybrid
)
from models.ensemble import ensemble

def run():
    report_path = "sandbox/report.json"

    features = extract_features(report_path)

    p1 = predict_sgan(features)
    p2 = predict_dgcnn(features)
    p3 = predict_cnn(features)
    p4 = predict_hybrid(features)

    final = ensemble([p1, p2, p3, p4])

    print("Features:", features)
    print("Predictions:", p1, p2, p3, p4)
    print("Final Score:", final)

if __name__ == "__main__":
    run()