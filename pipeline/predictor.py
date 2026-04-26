import os
import joblib
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
VOCAB_SIZE = 307
EMB_DIM = 128
NUM_CLASSES = 2
NUM_API_CALLS = 307
LATENT_DIM = 128

SEQ_LEN = 100

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

class GATLayer(nn.Module):

    def __init__(self,in_features,out_features,heads=4):
        super().__init__()

        self.heads = heads
        self.out_features = out_features

        self.W = nn.Parameter(
            torch.randn(heads,in_features,out_features)*0.01
        )

        self.a_src = nn.Parameter(
            torch.randn(heads,out_features,1)*0.01
        )

        self.a_dst = nn.Parameter(
            torch.randn(heads,out_features,1)*0.01
        )

    def forward(self,adj,X):

        B,N,_ = adj.size()

        outputs = []

        for h in range(self.heads):

            Wh = X @ self.W[h]

            f1 = Wh @ self.a_src[h]
            f2 = Wh @ self.a_dst[h]

            e = f1 + f2.transpose(1,2)

            e = F.leaky_relu(e)

            zero_vec = -9e15*torch.ones_like(e)

            attention = torch.where(adj>0,e,zero_vec)

            attention = F.softmax(attention,dim=2)

            h_out = attention @ Wh

            outputs.append(h_out)

        H = torch.cat(outputs,dim=2)

        return H

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.init_fc = nn.Linear(LATENT_DIM,256)

        self.rnn = nn.GRU(
            input_size=EMB_DIM,
            hidden_size=256,
            batch_first=True
        )

        self.token_proj = nn.Linear(256,NUM_API_CALLS)

        self.start_token = nn.Parameter(torch.zeros(1,1,EMB_DIM))

    def forward(self,z):

        B = z.size(0)

        h0 = torch.tanh(self.init_fc(z)).unsqueeze(0)

        inputs = self.start_token.repeat(B,SEQ_LEN,1)

        outputs,_ = self.rnn(inputs,h0)

        logits = self.token_proj(outputs)

        probs = F.gumbel_softmax(logits,tau=0.5,hard=True)

        tokens = torch.argmax(probs,dim=-1)

        return tokens
    
class GAT_Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.gat1 = GATLayer(SEQ_LEN,64,heads=4)
        self.gat2 = GATLayer(64*4,32,heads=4)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(NUM_API_CALLS*32*4,3)

    def forward(self,adj,X,return_features=False):

        Z = self.gat1(adj,X)
        Z = F.elu(Z)

        Z = self.gat2(adj,Z)
        Z = F.elu(Z)

        Z = self.dropout(Z)

        features = Z.reshape(Z.size(0),-1)

        logits = self.fc(features)

        if return_features:
            return logits,features

        return logits

# -----------------
# GNN-SGAN
# -----------------


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

dgcnn_model = DGCNN()
dgcnn_model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "models/saves/dgcnn/dgcnn_seed_50.pt"), map_location="cpu")
)
dgcnn_model.eval()

hdgcnn_model = DGCNN_Discriminator()
hdgcnn_model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "models/saves/hybrid/dgvgan_seed_50.pt"), map_location="cpu")
)
hdgcnn_model.eval()

def seq_to_graph(seq_batch):

    B = seq_batch.size(0)

    src = seq_batch[:,:-1]
    dst = seq_batch[:,1:]

    adj = torch.zeros((B,NUM_API_CALLS,NUM_API_CALLS),device=seq_batch.device)

    batch_index = torch.arange(B,device=seq_batch.device).unsqueeze(1)

    adj[batch_index,src,dst]+=1

    X = F.one_hot(seq_batch,NUM_API_CALLS).float().permute(0,2,1)

    return adj,X

# -----------------------------
# Prediction functions
# -----------------------------

def predict_sgan(features):
    start = time.time()
    x = features
    x = torch.tensor([x])
    with torch.no_grad():
        output = sgan_model(x)
        prob = torch.softmax(output, dim=1)[0][1].item()
    end = time.time()
    print(f"sgan (s): {end-start}")
    return prob
    
def predict_dgcnn(features):
    start = time.time()
    dgcnn_model.eval()

    with torch.no_grad():
        adj = torch.zeros((NUM_API_CALLS, NUM_API_CALLS))

        for i in range(len(features) - 1):
            src = int(features[i])
            dst = int(features[i + 1])

            if 0 <= src < NUM_API_CALLS and 0 <= dst < NUM_API_CALLS:
                adj[src, dst] = 1

        X = F.one_hot(
            torch.tensor(features, dtype=torch.long),
            num_classes=NUM_API_CALLS
        ).float().permute(1, 0)

        adj = adj.unsqueeze(0)
        X = X.unsqueeze(0)

        logits = dgcnn_model(adj, X)
        prob = torch.sigmoid(logits)
        end = time.time()
        
        print(f"dgcnn (s): {end-start}")
        return float(prob.item())
    
def predict_hybrid(features):
    start = time.time()
    hdgcnn_model.eval()

    with torch.no_grad():
        adj = torch.zeros((NUM_API_CALLS, NUM_API_CALLS))

        for i in range(len(features) - 1):
            src = int(features[i])
            dst = int(features[i + 1])

            if 0 <= src < NUM_API_CALLS and 0 <= dst < NUM_API_CALLS:
                adj[src, dst] = 1

        X = F.one_hot(
            torch.tensor(features, dtype=torch.long),
            num_classes=NUM_API_CALLS
        ).float().permute(1, 0)

        adj = adj.unsqueeze(0)
        X = X.unsqueeze(0)

        logits = hdgcnn_model(adj, X)
        probs = torch.softmax(logits[:, :2], dim=1)

        malware_prob = probs[0, 1].item()

    end = time.time()
    print(f"hybrid (s): {end-start}")

    return malware_prob

def predict_cnn(features):
    start = time.time()
    x = features
    x = torch.tensor([x])
    with torch.no_grad():
        output = cnn_model(x)
        prob = torch.softmax(output, dim=1)[0][1].item()
    end = time.time()
    print(f"cnn (s): {end-start}")
    return prob