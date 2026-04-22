import os
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_model(path):
    os.makedirs(path, exist_ok=True)

    X = np.array([
        [1, 0, 1],
        [2, 1, 0],
        [3, 1, 2],
        [5, 2, 1]
    ])
    y = [0, 1, 0, 1]

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, os.path.join(path, "model.pkl"))

create_model(os.path.join(BASE_DIR, "saves/sgan"))
create_model(os.path.join(BASE_DIR, "saves/dgcnn"))
create_model(os.path.join(BASE_DIR, "saves/cnn"))
create_model(os.path.join(BASE_DIR, "saves/hybrid"))

print("✅ Dummy models created!")