import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------
# Load & preprocess data
# ----------------------
file_path = "data/tahps_data.txt"
df = pd.read_csv(file_path, delimiter="\t", skiprows=1)
df.columns = ["date", "x1", "x2", "x3", "y1"]
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

# Optional: create some time-based features
df["dayofyear"] = df["date"].dt.dayofyear
df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# Sort by date (in case it's not sorted)
df = df.sort_values("date").reset_index(drop=True)

# Choose features and target
feature_cols = ["x1", "x2", "x3", "dayofyear", "dayofweek", "month"]
target_col = ["y1"]

# Convert to numpy
X = df[feature_cols].values
y = df[target_col].values

# 90/10 train/test split
n = len(df)
train_size = int(n * 0.9)  # last 10% as test
X_train_raw, X_test_raw = X[:train_size], X[train_size:]
y_train_raw, y_test_raw = y[:train_size], y[train_size:]

dates_train = df["date"][:train_size]
dates_test = df["date"][train_size:]

# Scale features & target with MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# Windowing: create sequences of length `history_size`
history_size = 30


def create_sequences(X_data, y_data, seq_len=history_size):
    sequences = []
    targets = []
    for i in range(len(X_data) - seq_len):
        seq_x = X_data[i : i + seq_len]
        seq_y = y_data[i + seq_len]
        sequences.append(seq_x)
        targets.append(seq_y)
    return np.array(sequences), np.array(targets)


X_train_seq, y_train_seq = create_sequences(
    X_train_scaled, y_train_scaled, history_size
)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, history_size)

print("X_train_seq.shape:", X_train_seq.shape)  # (num_samples, 30, num_features)
print("y_train_seq.shape:", y_train_seq.shape)  # (num_samples, 1)
print("X_test_seq.shape:", X_test_seq.shape)
print("y_test_seq.shape:", y_test_seq.shape)


# ----------------------
# Create PyTorch Dataset & DataLoaders
# ----------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ----------------------
#  Define the LSTM Model
#
# We’ll define a simple LSTM-based regressor:
#
#     An LSTM with hidden dimension 64 (configurable).
#     Possibly a nn.Dropout for regularization.
#     A final linear layer to output a single value (y1).
# ----------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(
            x
        )  # lstm_out: (batch_size, seq_len, hidden_dim)
        # We can take the last time step's output
        out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.fc(out)  # (batch_size, 1)
        return out


# ----------------------
# Training & Evaluation Loop
#
# A reusable function to train a given model:
# ----------------------


def train_model(model, train_loader, test_loader, num_epochs=20, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X).squeeze()  # shape: (batch_size,)
            loss = criterion(preds, batch_y.squeeze())
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * batch_X.size(0)

        epoch_train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                preds = model(batch_X).squeeze()
                loss = criterion(preds, batch_y.squeeze())
                epoch_val_loss += loss.item() * batch_X.size(0)
        epoch_val_loss /= len(test_loader.dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}"
            )

    return train_losses, val_losses


# ----------------------
# Instantiate and Train the LSTM
# ----------------------

input_dim = X_train_seq.shape[
    2
]  # number of features (e.g., 6 if x1,x2,x3,dayofyear,dayofweek,month)
hidden_dim = 64
num_layers = 2
dropout = 0.2
num_epochs = 20
learning_rate = 1e-3

lstm_model = LSTMModel(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
train_losses, val_losses = train_model(
    lstm_model, train_loader, test_loader, num_epochs, learning_rate
)

# Plot training & validation loss
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("LSTM Model Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()


# ----------------------
# Evaluate on the Test Set & Visualize
#
# To get predictions, we’ll run through the test_loader once more, storing predictions in order. Then we can invert the scaling and compare to the ground truth.
# ----------------------

# Switch to eval mode, gather predictions
lstm_model.eval()
test_preds_scaled = []
test_targets_scaled = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        preds = lstm_model(batch_X).squeeze()
        test_preds_scaled.append(preds.cpu().numpy())
        test_targets_scaled.append(batch_y.squeeze().cpu().numpy())

# Concatenate all batches
test_preds_scaled = np.concatenate(test_preds_scaled, axis=0)
test_targets_scaled = np.concatenate(test_targets_scaled, axis=0)

# Invert scaling
test_preds = scaler_y.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
test_targets = scaler_y.inverse_transform(test_targets_scaled.reshape(-1, 1)).flatten()

# Align dates for plotting
test_dates_for_plot = dates_test.iloc[history_size:].values  # offset by history_size

mae_lstm = mean_absolute_error(test_targets, test_preds)
mse_lstm = mean_squared_error(test_targets, test_preds)
rmse_lstm = np.sqrt(mse_lstm)

print(f"LSTM Test MAE: {mae_lstm:.3f}")
print(f"LSTM Test RMSE: {rmse_lstm:.3f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates_for_plot, test_targets, label="Actual", color="blue")
plt.plot(test_dates_for_plot, test_preds, label="LSTM Predicted", color="red")
plt.title("LSTM Predictions vs Actual")
plt.xlabel("Date")
plt.ylabel("y1")
plt.legend()
plt.show()

# Residuals
residuals_lstm = test_targets - test_preds
plt.figure(figsize=(10, 4))
plt.hist(residuals_lstm, bins=30, alpha=0.7, color="gray")
plt.title("LSTM: Distribution of Residuals")
plt.xlabel("Error (y_true - y_pred)")
plt.ylabel("Frequency")
plt.show()
