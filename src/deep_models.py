from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from .evaluation import metric_dict


class CNN1D(nn.Module):
    def __init__(self, channels: int, classes: int, hidden: int = 96, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, hidden, 5, stride=2, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden, classes),
        )
    def forward(self, x): return self.net(x)


class BiLSTM(nn.Module):
    def __init__(self, channels: int, classes: int, hidden: int = 96, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, classes))
    def forward(self, x):
        y, _ = self.lstm(x.transpose(1, 2)); return self.head(y[:, -1])


class Transformer1D(nn.Module):
    def __init__(self, channels: int, classes: int, hidden: int = 96, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Conv1d(channels, hidden, kernel_size=16, stride=16)
        enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden * 4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Linear(hidden, classes)
    def forward(self, x):
        z = self.proj(x).transpose(1, 2); z = self.encoder(z); return self.head(z.mean(dim=1))


def build_deep(name: str, channels: int, classes: int, hidden: int, dropout: float) -> nn.Module:
    if name == "cnn": return CNN1D(channels, classes, hidden, dropout)
    if name == "bilstm": return BiLSTM(channels, classes, hidden, dropout)
    if name == "transformer": return Transformer1D(channels, classes, hidden, dropout)
    raise ValueError(name)


def train_one(model: nn.Module, Xtr, ytr, Xva, yva, cfg: dict, device: str) -> nn.Module:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    counts = np.bincount(ytr, minlength=int(np.max(ytr)) + 1)
    weights = counts.sum() / np.maximum(counts, 1)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)), batch_size=int(cfg["batch_size"]), shuffle=True)
    best, best_f1, stale = None, -1.0, 0
    for _ in range(int(cfg["epochs"])):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xva, device=device)).argmax(1).cpu().numpy()
        f1 = f1_score(yva, pred, average="macro", zero_division=0)
        if f1 > best_f1 + 1e-4:
            best_f1, best, stale = f1, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]): break
    model.load_state_dict(best)
    return model


def deep_loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: dict, seed: int) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for subject in np.unique(groups):
        te = groups == subject; pool = ~te
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_rel, va_rel = next(gss.split(X[pool], y[pool], groups[pool]))
        pool_idx = np.where(pool)[0]; tr, va = pool_idx[tr_rel], pool_idx[va_rel]
        for name in cfg["models"]:
            torch.manual_seed(seed)
            model = build_deep(name, X.shape[1], len(np.unique(y)), int(cfg["hidden_dim"]), float(cfg["dropout"]))
            model = train_one(model, X[tr], y[tr], X[va], y[va], cfg, device)
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(X[te], device=device)).argmax(1).cpu().numpy()
            rows.append({"seed": seed, "method": "raw", "model": name.upper(), "subject_id": subject, **metric_dict(y[te], pred)})
    return pd.DataFrame(rows)
