import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── 1. DATA LOADING & PREPROCESSING ───────────────────────────────────────

def load_and_preprocess(csv_path: str,
                        feature_cols=None,
                        rolling_window: int = 10,
                        train_frac: float = 0.8):
    """
    1. Reads CSV
    2. Selects features, drops NA
    3. Applies rolling mean, differencing, min-max scaling
    4. Splits into train/test tensors
    """
    df = pd.read_csv(csv_path)
    feature_cols = feature_cols or ['Open','Close','High','Low','Volume']
    df = df[feature_cols].dropna()

    # smooth
    df_sm = df.rolling(rolling_window).mean().dropna()
    # diff + normalize
    df_diff = df_sm.diff().dropna()
    normed = (df_diff - df_diff.min()) / (df_diff.max() - df_diff.min())

    data = normed.values.astype(np.float32)
    split = int(len(data)*train_frac)
    train, test = data[:split], data[split:]
    return (torch.from_numpy(train), torch.from_numpy(test))

class TimeSeriesDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int=8):
        """
        data: [T, features]
        returns seq_len inputs → next-step target
        """
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, i):
        x = self.data[i:i+self.seq_len]
        y = self.data[i+1:i+self.seq_len+1, 1]
        return x, y.unsqueeze(-1)

# ─── 2. MODEL DEFINITIONS ───────────────────────────────────────────────────

class Time2Vec(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # one linear + sin projections
        self.w0 = nn.Parameter(torch.randn(in_dim, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w  = nn.Parameter(torch.randn(in_dim, out_dim-1))
        self.b  = nn.Parameter(torch.randn(out_dim-1))

    def forward(self, x):
        # x: [B, T, in_dim]
        # sin part
        sin_proj = torch.sin(x @ self.w + self.b)
        lin_proj = x @ self.w0 + self.b0
        return torch.cat([sin_proj, lin_proj], dim=-1)

class StockTransformer(nn.Module):
    def __init__(self, feat_dim: int, time2vec_dim: int, nhead: int, nlayers: int, hidden: int):
        super().__init__()
        self.time2vec = Time2Vec(feat_dim, time2vec_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim+time2vec_dim,
            nhead=nhead,
            dim_feedforward=hidden,
            dropout=0.1,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Sequential(
            nn.Linear(feat_dim+time2vec_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        # x: [B, T, feat_dim]
        tv = self.time2vec(x)
        cat = torch.cat([x, tv], dim=-1)              # [B, T, D]
        # Transformer expects [T, B, D]
        out = self.encoder(cat.permute(1,0,2))
        out = out.permute(1,2,0)                      # [B, D, T]
        pooled = self.pool(out).squeeze(-1)           # [B, D]
        return self.fc(pooled)                        # [B, 1]

# ─── 3. TRAIN / EVAL LOOPS ─────────────────────────────────────────────────

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred.squeeze(-1), y.squeeze(-1))
        loss.backward()
        opt.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total += loss_fn(pred.squeeze(-1), y.squeeze(-1)).item() * X.size(0)
    return total / len(loader.dataset)

# ─── 4. VISUALIZATION ───────────────────────────────────────────────────────

def plot_series(series: np.ndarray, title="Series", ylabel="Value"):
    plt.figure(figsize=(8,4))
    plt.plot(series, label=ylabel)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# ─── 5. MAIN ────────────────────────────────────────────────────────────────

def main():
    # hyperparams
    CSV_PATH = "TATASTEEL.csv"
    SEQ_LEN  = 8
    BATCH    = 32
    EPOCHS   = 10
    LR       = 1e-3
    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load
    train_t, test_t = load_and_preprocess(CSV_PATH)
    train_ds = TimeSeriesDataset(train_t, seq_len=SEQ_LEN)
    test_ds  = TimeSeriesDataset(test_t,  seq_len=SEQ_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH)

    # model
    feat_dim = train_t.shape[1]
    model = StockTransformer(
        feat_dim=feat_dim,
        time2vec_dim=feat_dim,   # e.g. same as input
        nhead=1,
        nlayers=2,
        hidden=64
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # train loop
    for epoch in range(1, EPOCHS+1):
        tr_loss = train_epoch(model, train_dl, opt, loss_fn, DEVICE)
        val_loss = evaluate(model, test_dl, loss_fn, DEVICE)
        print(f"Epoch {epoch:2d} – train: {tr_loss:.4f} │ val: {val_loss:.4f}")

    # plot raw vs predicted on test set
    model.eval()
    X_test, y_test = next(iter(test_dl))
    with torch.no_grad():
        preds = model(X_test.to(DEVICE)).cpu().squeeze(-1).numpy()
    plot_series(y_test[:, -1].numpy(), title="True Close Prices", ylabel="Normalized ΔPrice")
    plot_series(preds, title="Predicted ΔPrice", ylabel="ΔPrice Prediction")

if __name__ == "__main__":
    main()