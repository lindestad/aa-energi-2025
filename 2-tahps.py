import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# For the AutoReg baseline
from statsmodels.tsa.ar_model import AutoReg

# ----------------------
# 1. Load & Preprocess Data
# ----------------------
file_path = "data/tahps_data.txt"
df = pd.read_csv(file_path, delimiter="\t", skiprows=1)
df.columns = ["date", "x1", "x2", "x3", "y1"]
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

# Optional time-based features
df["dayofyear"] = df["date"].dt.dayofyear
df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

df = df.sort_values("date").reset_index(drop=True)

# --- Add y-lag features: e.g., up to 5 days of lag ---
max_lag = 5
for lag in range(1, max_lag + 1):
    df[f"y_lag_{lag}"] = df["y1"].shift(lag)

# Drop the first 'max_lag' rows that have NaNs due to shifting
df = df.dropna().reset_index(drop=True)

# Now define features to include the lags
feature_cols = ["x1", "x2", "x3", "dayofyear", "dayofweek", "month"] + [
    f"y_lag_{i}" for i in range(1, max_lag + 1)
]
target_col = "y1"  # predict today's y1

# Build feature matrix (X) and target (y)
X = df[feature_cols].values  # shape: (n, num_features + y_lag_features)
y = df[[target_col]].values  # shape: (n, 1)

# Train/test split (90% train, last 10% test)
n = len(df)
train_size = int(n * 0.9)
X_train_raw, X_test_raw = X[:train_size], X[train_size:]
y_train_raw, y_test_raw = y[:train_size], y[train_size:]
dates_train = df["date"][:train_size]
dates_test = df["date"][train_size:]

# For the neural nets, scale the features & target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# ----------------------
# 2. Baseline: AutoReg using only y (with 5 lags)
#
#   Even though we've built lag columns for the NN,
#   the classic AutoReg from statsmodels just uses y's own history.
#   We'll keep it simple: y(t) = f( y(t-1),..., y(t-5) ).
# ----------------------
# Flatten y for AutoReg
y_full_ar = df["y1"].values  # after dropping NaNs
y_train_ar = y_full_ar[:train_size]
y_test_ar = y_full_ar[train_size:]

# Fit AR(5)
ar_lags = 5
ar_model = AutoReg(y_train_ar, lags=ar_lags).fit()
ar_preds = ar_model.predict(start=train_size, end=train_size + len(y_test_ar) - 1)

# Evaluate AR
ar_mae = mean_absolute_error(y_test_ar, ar_preds)
ar_mse = mean_squared_error(y_test_ar, ar_preds)
ar_rmse = np.sqrt(ar_mse)

# ----------------------
# 3. Prepare Sequences for LSTM/Transformer
#    We'll still do "history_size" windowing, but
#    each time-step's input includes the lag features as well.
# ----------------------
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

test_dates_for_plot = dates_test.iloc[history_size:].values  # offset for final plotting


# Make Dataset & Dataloaders
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
# 4. Define LSTM & Transformer
# ----------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
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
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take last time-step
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerTimeSeries, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_fc(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)  # add positional encoding
        x = self.transformer_encoder(x)  # pass through Transformer
        x = x[:, -1, :]  # final time step
        out = self.fc_out(x)  # (batch_size, 1)
        return out


# ----------------------
# 5. Training with Early Stopping
# ----------------------
def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3, patience=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_weights = None
    no_improve_count = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X).squeeze()
            loss = criterion(preds, batch_y.squeeze())
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)
        epoch_train_loss /= len(train_loader.dataset)

        # ---- Validate ----
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                preds = model(batch_X).squeeze()
                loss = criterion(preds, batch_y.squeeze())
                epoch_val_loss += loss.item() * batch_X.size(0)
        epoch_val_loss /= len(val_loader.dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Print progress every few epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, "
                f"Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}"
            )

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return train_losses, val_losses


# ----------------------
# 6. Train & Evaluate LSTM
# ----------------------
input_dim = X_train_seq.shape[2]  # includes exogenous + y-lag features
hidden_dim = 128
num_layers = 3
dropout = 0.2
num_epochs = 100
learning_rate = 1e-3
patience = 15

lstm_model = LSTMModel(input_dim, hidden_dim, num_layers, dropout)
train_losses_lstm, val_losses_lstm = train_model(
    lstm_model,
    train_loader,
    test_loader,
    num_epochs=num_epochs,
    lr=learning_rate,
    patience=patience,
)

# Inference on test set
lstm_model.eval()
lstm_preds_scaled = []
lstm_targets_scaled = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        preds = lstm_model(batch_X).squeeze()
        lstm_preds_scaled.append(preds.cpu().numpy())
        lstm_targets_scaled.append(batch_y.squeeze().cpu().numpy())

lstm_preds_scaled = np.concatenate(lstm_preds_scaled, axis=0)
lstm_targets_scaled = np.concatenate(lstm_targets_scaled, axis=0)

lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()
lstm_targets = scaler_y.inverse_transform(lstm_targets_scaled.reshape(-1, 1)).flatten()

lstm_mae = mean_absolute_error(lstm_targets, lstm_preds)
lstm_mse = mean_squared_error(lstm_targets, lstm_preds)
lstm_rmse = np.sqrt(lstm_mse)

# ----------------------
# 7. Train & Evaluate Transformer
# ----------------------
transformer_model = TransformerTimeSeries(
    input_dim, d_model=256, nhead=8, num_layers=2, dropout=0.2
)
train_losses_t, val_losses_t = train_model(
    transformer_model,
    train_loader,
    test_loader,
    num_epochs=num_epochs,
    lr=learning_rate,
    patience=patience,
)

transformer_model.eval()
trans_preds_scaled = []
trans_targets_scaled = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        preds = transformer_model(batch_X).squeeze()
        trans_preds_scaled.append(preds.cpu().numpy())
        trans_targets_scaled.append(batch_y.squeeze().cpu().numpy())

trans_preds_scaled = np.concatenate(trans_preds_scaled, axis=0)
trans_targets_scaled = np.concatenate(trans_targets_scaled, axis=0)

trans_preds = scaler_y.inverse_transform(trans_preds_scaled.reshape(-1, 1)).flatten()
trans_targets = scaler_y.inverse_transform(
    trans_targets_scaled.reshape(-1, 1)
).flatten()

trans_mae = mean_absolute_error(trans_targets, trans_preds)
trans_mse = mean_squared_error(trans_targets, trans_preds)
trans_rmse = np.sqrt(trans_mse)

# ----------------------
# 8. Final Visualization & Metrics
# ----------------------
plt.figure(figsize=(12, 6))
# For the timeline, we offset AR predictions by the same 30 days to align
# with the neural net's test window. (Though AR was done on the full test set).
# We'll slice AR predictions if needed:
ar_pred_aligned = (
    ar_preds[history_size:] if len(ar_preds) >= len(trans_preds) else ar_preds
)

# Plot
plt.plot(
    test_dates_for_plot,
    lstm_targets,
    label="Actual",
    color="black",
    linewidth=2,
    linestyle=":",
)
plt.plot(
    test_dates_for_plot,
    ar_pred_aligned,
    label="AutoReg(5)",
    color="orange",
    linestyle="--",
    alpha=0.7,
)
plt.plot(test_dates_for_plot, lstm_preds, label="LSTM", color="red", linestyle="-")
plt.plot(
    test_dates_for_plot, trans_preds, label="Transformer", color="green", linestyle="-"
)

plt.title("Comparison: AutoReg vs. LSTM vs. Transformer (with Y-lags) vs. Actual")
plt.xlabel("Date")
plt.ylabel("y1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar chart of MAE & RMSE
model_names = ["AutoReg(5)", "LSTM", "Transformer"]
mae_values = [ar_mae, lstm_mae, trans_mae]
rmse_values = [ar_rmse, lstm_rmse, trans_rmse]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
rects1 = ax.bar(x - width / 2, mae_values, width, label="MAE", color="skyblue")
rects2 = ax.bar(x + width / 2, rmse_values, width, label="RMSE", color="salmon")

ax.set_ylabel("Error")
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_title("MAE and RMSE by Model")
ax.legend()
ax.grid(axis="y", linestyle="--")


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()

# Print final metrics
print("AR coefficients:\n", ar_model.params)
print("\n")
print("===== Final Metrics (all test data) =====")
print(f"AutoReg(5)    MAE: {ar_mae:.3f}, RMSE: {ar_rmse:.3f}")
print(f"LSTM          MAE: {lstm_mae:.3f}, RMSE: {lstm_rmse:.3f}")
print(f"Transformer   MAE: {trans_mae:.3f}, RMSE: {trans_rmse:.3f}")
