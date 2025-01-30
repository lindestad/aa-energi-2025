import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


class PytorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_dim=10,
        output_dim=4,
        hidden_size=32,
        n_hidden_layers=1,
        dropout=0.0,
        lr=1e-3,
        max_epochs=20,
        batch_size=256,
        device="cpu",
        verbose=False,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device

    def _build_model(self):
        layers = []
        in_dim = self.input_dim
        for _ in range(self.n_hidden_layers):
            layers.append(nn.Linear(in_dim, self.hidden_size))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_size
        layers.append(nn.Linear(in_dim, self.output_dim))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)  # Move to device
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)  # Move to device

        self.model_ = self._build_model().to(self.device)  # Ensure model is on device
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=self.lr)
        self.criterion_ = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model_.train()

        for epoch in tqdm(
            range(self.max_epochs),
            desc="Training Progress",
            unit="epoch",
            leave=True,  # Keeps the bar after completion, but prevents excessive new lines
            dynamic_ncols=True,  # Adjusts width dynamically
        ):

            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer_.zero_grad()
                preds = self.model_(batch_X)
                loss = self.criterion_(preds, batch_y)
                loss.backward()
                self.optimizer_.step()
                epoch_loss += loss.item() * batch_X.size(0)

            if self.verbose:
                avg_loss = epoch_loss / len(dataloader.dataset)
                tqdm.write(f"Epoch {epoch+1}/{self.max_epochs}, Loss: {avg_loss:.4f}")

        return self

    def predict(self, X):
        self.model_.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)  # Move to device
        with torch.no_grad():
            preds = self.model_(X_t).cpu().numpy()  # Move back to CPU for NumPy
        return preds

    def score(self, X, y):
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)
