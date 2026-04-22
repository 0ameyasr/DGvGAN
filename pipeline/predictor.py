import os
import joblib
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# Dummy models (for demo)
# -----------------------------
sgan_model = joblib.load(os.path.join(BASE_DIR, "models/saves/sgan/model.pkl"))
dgcnn_model = joblib.load(os.path.join(BASE_DIR, "models/saves/dgcnn/model.pkl"))
hybrid_model = joblib.load(os.path.join(BASE_DIR, "models/saves/hybrid/model.pkl"))

# -----------------------------
# CNN MODEL (REAL)
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(307, 128)

        self.conv = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.fc(x)
        return x

# Load CNN model
cnn_model = Discriminator()
cnn_model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "models/saves/cnn/cnn_seed_50.pt"), map_location="cpu")
)
cnn_model.eval()

# -----------------------------
# Prediction functions
# -----------------------------

def predict_sgan(features):
    return float(sgan_model.predict([features])[0])

def predict_dgcnn(features):
    return float(dgcnn_model.predict([features])[0])

def predict_hybrid(features):
    return float(hybrid_model.predict([features])[0])

def predict_cnn(features):
    # Convert small feature vector → sequence (length 100)
    x = features * 34
    x = x[:100]

    x = torch.tensor([x])
    with torch.no_grad():
        output = cnn_model(x)
        prob = torch.softmax(output, dim=1)[0][1].item()
    return prob