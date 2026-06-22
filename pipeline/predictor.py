import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_SIZE = 307
EMB_DIM = 128
NUM_CLASSES = 2
NUM_API_CALLS = 307
LATENT_DIM = 128
SEQ_LEN = 100
HIDDEN_DIM = 256

# -----------------------------
# CNN
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

# -------
# SGAN
# -------
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

        x = x.permute(0, 2, 1)
        features = self.conv(x)
        logits = self.fc(features)

        if return_features:
            return logits, features
        return logits

# ----------------------------
# DGCNN
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
    
class DGCNN(nn.Module):
    def __init__(self, out_channels=31, dropout=0.6):
        super().__init__()
        self.gcn = GraphConvLayer(SEQ_LEN, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(NUM_API_CALLS * out_channels, 1)

    def forward(self, adj, X):
        Z = self.gcn(adj, X)
        Z = F.relu(Z)
        Z = self.dropout(Z)
        Z = Z.reshape(Z.size(0), -1)
        out = self.fc(Z)
        return out.squeeze(1)

# -----------------
# DGCNN-SGAN
# -----------------
class DGCNN_Discriminator(nn.Module):
    def __init__(self, out_channels=31):
        super().__init__()
        self.gcn = GraphConvLayer(SEQ_LEN, out_channels)
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(NUM_API_CALLS * out_channels, 3)

    def forward(self, adj, X, return_features=False):
        Z = self.gcn(adj, X)
        Z = F.relu(Z)
        Z = self.dropout(Z)
        features = Z.reshape(Z.size(0), -1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits

# -----------
# GAT-SGAN
# -----------
class GATHybridLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4):
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.W = nn.Parameter(torch.randn(heads, in_features, out_features) * 0.01)
        self.a_src = nn.Parameter(torch.randn(heads, out_features, 1) * 0.01)
        self.a_dst = nn.Parameter(torch.randn(heads, out_features, 1) * 0.01)

    def forward(self, adj, X):
        B, N, _ = adj.size()
        outputs = []

        for h in range(self.heads):
            Wh = X @ self.W[h]
            f1 = Wh @ self.a_src[h]
            f2 = Wh @ self.a_dst[h]
            e = f1 + f2.transpose(1, 2)
            e = F.leaky_relu(e)

            # FIXED: Safe masking boundary from -9e15 to -1e9 to avoid NaN underflows
            zero_vec = -1e9 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=2)
            
            # Catch completely empty graph rows to avoid passing downstream NaNs
            if torch.isnan(attention).any():
                attention = torch.zeros_like(attention)

            h_out = attention @ Wh
            outputs.append(h_out)

        return torch.cat(outputs, dim=2)

class GAT_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat1 = GATHybridLayer(SEQ_LEN, 64, heads=4)
        self.gat2 = GATHybridLayer(64 * 4, 32, heads=4)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(NUM_API_CALLS * 32 * 4, 3)

    def forward(self, adj, X, return_features=False):
        Z = F.elu(self.gat1(adj, X))
        Z = F.elu(self.gat2(adj, Z))
        Z = self.dropout(Z)
        features = Z.reshape(Z.size(0), -1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits

# ------
# GAT
# ------
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4):
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.W = nn.Parameter(torch.randn(heads, in_features, out_features) * 0.01)
        self.a_src = nn.Parameter(torch.randn(heads, out_features, 1) * 0.01)
        self.a_dst = nn.Parameter(torch.randn(heads, out_features, 1) * 0.01)

    def forward(self, adj, X):
        B, N, _ = X.size() 
        Wh = torch.matmul(X.unsqueeze(1), self.W) 
        f1 = torch.matmul(Wh, self.a_src) 
        f2 = torch.matmul(Wh, self.a_dst) 
        
        e = F.leaky_relu(f1 + f2.transpose(-2, -1))
        # FIXED: Changed -9e15 thresholding to stable -1e9
        mask = torch.where(adj.unsqueeze(1) > 0, e, torch.full_like(e, -1e9))
        attention = F.softmax(mask, dim=-1)

        if torch.isnan(attention).any():
            attention = torch.zeros_like(attention)

        h_out = torch.matmul(attention, Wh)
        h_out = h_out.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.out_features)
        return h_out

class GATClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.gat1 = GATLayer(in_features=SEQ_LEN, out_features=64, heads=4)
        self.gat2 = GATLayer(in_features=64 * 4, out_features=32, heads=4)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(NUM_API_CALLS * 32 * 4, num_classes)

    def forward(self, adj, X, return_features=False):
        Z = F.elu(self.gat1(adj, X))
        Z = F.elu(self.gat2(adj, Z))
        Z = self.dropout(Z)
        features = Z.reshape(Z.size(0), -1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits

# -------
# LSTM
# -------
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMB_DIM)
        self.lstm = nn.LSTM(
            input_size=EMB_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        x = hidden[-1]
        x = self.fc(x)
        return x
    
# Seq2Graph
def seq_to_graph(seq_batch):
    B = seq_batch.size(0)
    src = seq_batch[:, :-1]
    dst = seq_batch[:, 1:]

    adj = torch.zeros((B, NUM_API_CALLS, NUM_API_CALLS), device=seq_batch.device)
    batch_index = torch.arange(B, device=seq_batch.device).unsqueeze(1)
    adj[batch_index, src, dst] += 1

    X = F.one_hot(seq_batch, NUM_API_CALLS).float().permute(0, 2, 1)
    return adj, X

# -----------------------------
# Safe Prediction wrappers
# -----------------------------
def predict_sgan(features, seed=10):
    if not features: return 0.0, 0.5
    sgan_model = SGANDiscriminator()
    sgan_model.load_state_dict(torch.load(os.path.join(BASE_DIR, f"models/saves/sgan/sgan_seed-{seed}.pt"), map_location="cpu"))
    sgan_model.eval()

    start = time.time()
    x = torch.tensor([features])
    with torch.no_grad():
        output = sgan_model(x)
        prob = torch.softmax(output, dim=1)[0][1].item()
    return time.time() - start, prob

def predict_lstm(features, seed=10):
    if not features: return 0.0, 0.5
    lstm_model = LSTMClassifier()
    lstm_model.load_state_dict(torch.load(os.path.join(BASE_DIR, f"models/saves/lstm/lstm_seed_{seed}.pt"), map_location="cpu"))
    lstm_model.eval()

    start = time.time()
    x = torch.tensor([features])
    with torch.no_grad():
        output = lstm_model(x)
        prob = torch.softmax(output, dim=1)[0][1].item()
    return time.time() - start, prob
    
def predict_dgcnn(features, seed=10):
    if not features or len(features) < 2: return 0.0, 0.5
    dgcnn_model = DGCNN()
    dgcnn_model.load_state_dict(torch.load(os.path.join(BASE_DIR, f"models/saves/dgcnn/dgcnn_seed_{seed}.pt"), map_location="cpu"))
    dgcnn_model.eval()
    
    features = [int(x) for x in features]
    start = time.time()

    with torch.no_grad():
        adj = torch.zeros((NUM_API_CALLS, NUM_API_CALLS))
        for i in range(len(features) - 1):
            src, dst = features[i], features[i + 1]
            if 0 <= src < NUM_API_CALLS and 0 <= dst < NUM_API_CALLS:
                adj[src, dst] = 1

        X = F.one_hot(torch.tensor(features, dtype=torch.long), num_classes=NUM_API_CALLS).float().permute(1, 0)
        
        # Guard feature sizing boundary anomalies
        if X.size(1) != SEQ_LEN:
            X = F.pad(X, (0, max(0, SEQ_LEN - X.size(1))))[:, :SEQ_LEN]

        adj, X = adj.unsqueeze(0), X.unsqueeze(0)
        logits = dgcnn_model(adj, X)
        prob = torch.sigmoid(logits).item()
        
        # Post-execution sanity check against NaNs
        if np.isnan(prob): prob = 0.5
    return time.time() - start, float(prob)
    
def predict_dgcnn_sgan(features, seed=10):
    if not features or len(features) < 2: return 0.0, 0.5
    hdgcnn_model = DGCNN_Discriminator()
    hdgcnn_model.load_state_dict(torch.load(os.path.join(BASE_DIR, f"models/saves/hybrid/dgvgan_seed_{seed}.pt"), map_location="cpu"))
    hdgcnn_model.eval()
    
    features = [int(x) for x in features]
    start = time.time()

    with torch.no_grad():
        adj = torch.zeros((NUM_API_CALLS, NUM_API_CALLS))
        for i in range(len(features) - 1):
            src, dst = features[i], features[i + 1]
            if 0 <= src < NUM_API_CALLS and 0 <= dst < NUM_API_CALLS:
                adj[src, dst] = 1

        X = F.one_hot(torch.tensor(features, dtype=torch.long), num_classes=NUM_API_CALLS).float().permute(1, 0)
        if X.size(1) != SEQ_LEN:
            X = F.pad(X, (0, max(0, SEQ_LEN - X.size(1))))[:, :SEQ_LEN]

        adj, X = adj.unsqueeze(0), X.unsqueeze(0)
        logits = hdgcnn_model(adj, X)
        probs = torch.softmax(logits[:, :2], dim=1)
        malware_prob = probs[0, 1].item()
        
        if np.isnan(malware_prob): malware_prob = 0.5
    return time.time() - start, malware_prob

def predict_cnn(features, seed=10):
    if not features: return 0.0, 0.5
    cnn_model = Discriminator()
    cnn_model.load_state_dict(torch.load(os.path.join(BASE_DIR, f"models/saves/cnn/cnn_seed_{seed}.pt"), map_location="cpu"))
    cnn_model.eval()

    start = time.time()
    x = torch.tensor([features])
    with torch.no_grad():
        output = cnn_model(x)
        prob = torch.softmax(output, dim=1)[0][1].item()
    return time.time() - start, prob

def predict_gat_sgan(features, seed=10):
    if not features or len(features) < 2: return 0.0, 0.5
    gat_sgan_model = GAT_Discriminator()
    checkpoint = torch.load(os.path.join(BASE_DIR, f"models/saves/hybrid_sgan_gat/state_dict_seed_{seed}.pth"), map_location="cpu")
    gat_sgan_model.load_state_dict(checkpoint["D_state_dict"])

    start = time.time()
    gat_sgan_model.eval()
    features = [int(feat) for feat in features]
    
    # Pad out input features if they fall short of expected static dimensions
    if len(features) < SEQ_LEN:
        features += [0] * (SEQ_LEN - len(features))
    features = features[:SEQ_LEN]

    with torch.no_grad():
        seq = torch.tensor(features, dtype=torch.long).unsqueeze(0)
        adj, X = seq_to_graph(seq)
        logits = gat_sgan_model(adj, X)
        probs = torch.softmax(logits[:, :2], dim=1)
        malware_prob = probs[0, 1].item()
        
        if np.isnan(malware_prob): malware_prob = 0.5
    return time.time() - start, malware_prob

def predict_gat(features, seed=10):
    if not features or len(features) < 2: return 0.0, 0.5
    import __main__
    __main__.GATClassifier = GATClassifier
    __main__.GATLayer = GATLayer

    gat_model = torch.load(os.path.join(BASE_DIR, f"models/saves/gat/gat_seed_{seed}.pt"), map_location="cpu", weights_only=False)
    start = time.time()
    gat_model.eval()
    
    features = [int(feat) for feat in features]
    if len(features) < SEQ_LEN:
        features += [0] * (SEQ_LEN - len(features))
    features = features[:SEQ_LEN]

    with torch.no_grad():
        seq = torch.tensor(features, dtype=torch.long).unsqueeze(0)
        adj, X = seq_to_graph(seq)
        logits = gat_model(adj, X)
        probs = torch.softmax(logits[:, :2], dim=1)
        malware_prob = probs[0, 1].item()
        
        if np.isnan(malware_prob): malware_prob = 0.5
    return time.time() - start, malware_prob