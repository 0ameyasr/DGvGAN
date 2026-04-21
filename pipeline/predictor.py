import numpy as np

# Dummy models (replace later with real ones)
def predict_sgan(features):
    return float(sum(features) * 0.1)

def predict_dgcnn(features):
    return float(sum(features) * 0.2)

def predict_cnn(features):
    return float(sum(features) * 0.3)

def predict_hybrid(features):
    return float(sum(features) * 0.4)