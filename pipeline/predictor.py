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

VOCAB_SIZE = 307
EMB_DIM = 128
NUM_CLASSES = 2
class SGANDiscriminator(nn.Module):
  def __init__(self):
    super().__init__()

    self.embedding = nn.Embedding(VOCAB_SIZE, EMB_DIM)

    self.conv = nn.Sequential(
        nn.Conv1d(EMB_DIM, 128, 5, padding=2),
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
        nn.Linear(128, NUM_CLASSES + 1)
    )

  def forward(self, x, embedded=False, return_features=False):
    if not embedded:
        x = x.long()
        x = self.embedding(x)

    x = x.permute(0, 2, 1) # (BatchSize, EmbeddingSpace, SequenceLength)
    features = self.conv(x)
    logits = self.fc(features)

    if return_features:
        return logits, features

    return logits

class GNN_Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.gcn1 = GraphConvLayer(SEQ_LEN, 64)
        self.gcn2 = GraphConvLayer(64, 32)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(NUM_API_CALLS * 32, 3)

    def forward(self, adj, X, return_features=False):

        Z = self.gcn1(adj, X)
        Z = F.relu(Z)

        Z = self.gcn2(adj, Z)
        Z = F.relu(Z)

        Z = self.dropout(Z)

        features = Z.reshape(Z.size(0), -1)

        logits = self.fc(features)

        if return_features:
            return logits, features

        return logits

# ----------------------------
# Graph Convolutional Layer
# ----------------------------

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features) * 0.01
        )

    def forward(self, adj, X):
        B, N, _ = adj.size()
        I = torch.eye(N, device=adj.device).unsqueeze(0)
        A_hat = adj + I
        D = torch.sum(A_hat, dim=2)
        D_inv = torch.diag_embed(1.0 / (D + 1e-6))
        A_norm = D_inv @ A_hat
        Z = A_norm @ X
        Z = Z @ self.weight
        return Z
    
# Load CNN model
cnn_model = Discriminator()
cnn_model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "models/saves/cnn/cnn_seed_50.pt"), map_location="cpu")
)
cnn_model.eval()


sgan_model = SGANDiscriminator()
sgan_model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "models/saves/sgan/sgan_seed-50.pt"), map_location="cpu")
)
sgan_model.eval()

# -----------------------------
# Prediction functions
# -----------------------------

def predict_sgan(features):
    x = features
    x = torch.tensor([x])
    with torch.no_grad():
        output = cnn_model(x)
        prob = torch.softmax(output, dim=1)[0][1].item()
    return prob

def predict_dgcnn(features):
    return float(dgcnn_model.predict([features])[0])

def predict_hybrid(features):
    return float(hybrid_model.predict([features])[0])

def predict_cnn(features):
    # Convert small feature vector → sequence (length 100)
    # x = features * 34
    # x = x[:100]
    
    x = features
    x = torch.tensor([x])
    with torch.no_grad():
        output = cnn_model(x)
        prob = torch.softmax(output, dim=1)[0][1].item()
    return prob