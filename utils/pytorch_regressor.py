import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


class PytorchRegressor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible PyTorch regressor that can handle multi-output.

    Example usage with RandomizedSearchCV or GridSearchCV:

    from sklearn.model_selection import RandomizedSearchCV

    param_dist = {
        'hidden_size': [16, 32, 64],
        'n_hidden_layers': [1, 2],
        'dropout': [0.0, 0.2],
        'lr': [1e-2, 1e-3],
        'max_epochs': [20, 40],
    }

    reg = PytorchRegressor(input_dim=10, output_dim=4)
    search = RandomizedSearchCV(reg, param_dist, n_iter=5, cv=3, scoring='neg_mean_squared_error')
    search.fit(X_train, y_train)
    """

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
        """
        Parameters
        ----------
        input_dim : int
            Number of input features.
        output_dim : int
            Number of output dimensions (multi-output).
        hidden_size : int
            Number of neurons in each hidden layer.
        n_hidden_layers : int
            Number of hidden layers.
        dropout : float
            Dropout ratio.
        lr : float
            Learning rate for the optimizer.
        max_epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        device : str
            'cpu' or 'cuda' if GPU is available.
        verbose : bool
            Print training progress or not.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        # We'll define our model and other attributes in fit()

    def _build_model(self):
        """Build a simple MLP with the specified architecture."""
        layers = []
        in_dim = self.input_dim
        for _ in range(self.n_hidden_layers):
            layers.append(nn.Linear(in_dim, self.hidden_size))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_size
        layers.append(nn.Linear(in_dim, self.output_dim))
        model = nn.Sequential(*layers)
        return model

    def fit(self, X, y):
        # Convert X, y to PyTorch tensors
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # Build model
        self.model_ = self._build_model()
        self.model_.to(self.device)

        # Define optimizer and loss
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=self.lr)
        self.criterion_ = nn.MSELoss()

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model_.train()

        # Wrap the epoch loop in tqdm for progress tracking
        for epoch in tqdm(
            range(self.max_epochs), desc="Training Progress", unit="epoch"
        ):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer_.zero_grad()
                preds = self.model_(batch_X)
                loss = self.criterion_(preds, batch_y)
                loss.backward()
                self.optimizer_.step()
                epoch_loss += loss.item() * batch_X.size(0)

            if self.verbose:
                avg_loss = epoch_loss / len(dataloader.dataset)
                print(f"Epoch {epoch+1}/{self.max_epochs}, Loss: {avg_loss:.4f}")

        return self

    def predict(self, X):
        # Inference mode
        self.model_.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model_(X_t).cpu().numpy()
        return preds

    def score(self, X, y):
        # By default, let's return negative MSE for consistency with scikit-learn
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        # Higher score = better, so let's do negative MSE
        return -mse
