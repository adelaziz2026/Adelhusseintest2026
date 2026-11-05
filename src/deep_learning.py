"""A small PyTorch feed-forward network for tabular binary classification."""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU, batch norm and dropout."""

    def __init__(self, n_features: int, hidden=(64, 32), dropout: float = 0.2):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchClassifier:
    """Minimal sklearn-style wrapper: fit / predict / predict_proba."""

    def __init__(self, hidden=(64, 32), dropout=0.2, lr=1e-3, epochs=100,
                 batch_size=32, random_state=42):
        self.hidden, self.dropout, self.lr = hidden, dropout, lr
        self.epochs, self.batch_size, self.random_state = epochs, batch_size, random_state
        self.scaler = StandardScaler()
        self.history = []

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = torch.tensor(self.scaler.fit_transform(np.asarray(X)), dtype=torch.float32)
        y = torch.tensor(np.asarray(y), dtype=torch.float32)
        self.model = MLP(X.shape[1], self.hidden, self.dropout)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        self.history = []
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(X))
            total = 0.0
            for i in range(0, len(X), self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:  # BatchNorm needs more than one sample
                    continue
                opt.zero_grad()
                loss = loss_fn(self.model(X[idx]), y[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.history.append(total / len(X))
        return self

    def predict_proba(self, X):
        self.model.eval()
        X = torch.tensor(self.scaler.transform(np.asarray(X)), dtype=torch.float32)
        with torch.no_grad():
            p = torch.sigmoid(self.model(X)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
