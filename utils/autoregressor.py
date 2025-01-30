import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -------------------------
# 1. LSTM & Transformer Definitions
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.2):
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
    def __init__(self, input_dim, d_model=128, nhead=16, num_layers=3, dropout=0.2):
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
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        out = self.fc_out(x)
        return out


# -------------------------
# 2. Early Stopping Routine
# -------------------------
def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3, patience=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_weights = None
    no_improve_count = 0

    for epoch in range(num_epochs):
        # --- TRAIN ---
        model.train()
        running_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X).squeeze()
            loss = criterion(preds, batch_y.squeeze())
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_X.size(0)
        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # --- VALIDATE ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                preds = model(batch_X).squeeze()
                val_loss = criterion(preds, batch_y.squeeze())
                running_val_loss += val_loss.item() * batch_X.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        # Early stop checks
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, TrainLoss={epoch_train_loss:.4f}, ValLoss={epoch_val_loss:.4f}"
            )

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)


# -------------------------
# 3. Walk-Forward Inference (One-Step-Ahead) for LSTM/Transformer
# -------------------------
def walk_forward_predict_nn(model, all_data, history_size, train_end, device="cpu"):
    """
    model: trained LSTM or Transformer
    all_data: a 2D numpy array of shape (n, input_dim+1) if we want y included in last column
              or a 2D array of shape (n, input_dim) if y is not included.
              We'll assume the final column is the target (scaled).
    history_size: how many past timesteps we feed into the model (e.g. 30)
    train_end: index separating train/test, i.e. all_data[train_end:] is test
    device: "cpu" or "cuda"

    Returns:
       preds: shape (test_length,) unscaled predictions for the test portion
    """
    model.eval()

    n = len(all_data)
    test_length = n - train_end
    preds = []

    # We'll keep a rolling buffer of the last `history_size` rows
    # Initially: the last 'history_size' portion of train
    buffer = np.copy(all_data[train_end - history_size : train_end, :])

    for i in range(test_length):
        # 1) Build input of shape (1, history_size, input_dim)
        #    We exclude the final column if it is the target, i.e. input_dim = all_data.shape[1] - 1
        X_seq = buffer[:, :-1]  # everything except the last column is input features
        # shape = (history_size, input_dim)
        X_seq_torch = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(device)

        # 2) Forward pass
        with torch.no_grad():
            y_hat_scaled = model(X_seq_torch).item()

        # 3) Store the prediction
        preds.append(y_hat_scaled)

        # 4) "Observe" the real next day => all_data[train_end + i, -1]
        #    so for day i, we see the actual y, put it in the buffer for the next iteration
        # SHIFT the buffer up by 1, drop the oldest row
        if i < test_length:
            # next actual row is all_data[train_end + i, :]
            next_row = np.copy(all_data[train_end + i, :])
            # BUT we want to replace the last column of next_row with the *actual* y.
            # It's already in the scaled all_data. So that is correct if we've pre-populated all_data with real y.
            # We then push it into the buffer.
            buffer = np.concatenate([buffer[1:], next_row.reshape(1, -1)], axis=0)

    return np.array(preds)


# -------------------------
# 4. Full Implementation
# -------------------------
def main():
    # 4.1 Load Data
    file_path = "data/tahps_data.txt"
    df = pd.read_csv(file_path, delimiter="\t", skiprows=1)
    df.columns = ["date", "x1", "x2", "x3", "y1"]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

    # Optional time-based features
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # We'll keep it simple: features + target all in one array,
    # final column is y1, so input_dim = total_cols - 1
    # This way we can do walk-forward by updating the real y each day.
    feature_cols = ["x1", "x2", "x3", "dayofyear", "dayofweek", "month"]
    df_features = df[feature_cols].values
    df_target = df["y1"].values.reshape(-1, 1)

    # Stack them: shape (n, input_dim+1)
    data_np = np.hstack([df_features, df_target])  # last column is target

    # 4.2 Scaling
    # We'll scale features and target separately, then re-stack
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    scaled_features = scaler_features.fit_transform(df_features)
    scaled_target = scaler_target.fit_transform(df_target)

    # Re-stack them
    all_scaled = np.hstack([scaled_features, scaled_target])

    # 4.3 Train/Test Split
    n = len(df)
    train_ratio = 0.9
    train_size = int(n * train_ratio)
    test_size = n - train_size
    train_end = train_size  # index separating train from test
    dates_train = df["date"][:train_size]
    dates_test = df["date"][train_size:]

    # 4.4 Build sequences for training (batch approach)
    # We'll do the standard approach for training:
    history_size = 30

    def create_sequences_for_training(data, seq_len=30, start=0, end=train_end):
        # data shape: (n, input_dim+1)
        # We'll use [:-1] columns as input, the last column as target
        X_list = []
        Y_list = []
        for i in range(start, end - seq_len):
            window = data[i : i + seq_len, :-1]
            label = data[i + seq_len, -1]
            X_list.append(window)
            Y_list.append(label)
        return np.array(X_list), np.array(Y_list)

    X_train_seq, y_train_seq = create_sequences_for_training(
        all_scaled, seq_len=history_size, start=0, end=train_end
    )

    # For validation, we can do the same approach on the latter portion of the *train* or a small chunk.
    # For simplicity, let's just do a 80/10/10 split or so. We'll define "val_start" after 80% of train:
    val_start = int(train_end * 0.8)
    X_val_seq, y_val_seq = create_sequences_for_training(
        all_scaled, seq_len=history_size, start=val_start, end=train_end
    )

    # Convert to PyTorch
    class TimeSeriesDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
    val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 4.5 Define LSTM & Transformer
    input_dim = scaled_features.shape[1]  # number of input features (i.e. 6)
    hidden_dim = 64
    model_lstm = LSTMModel(input_dim, hidden_dim=hidden_dim, num_layers=2, dropout=0.2)
    model_transformer = TransformerTimeSeries(
        input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2
    )

    # 4.6 Train LSTM
    num_epochs = 200
    lr = 1e-4
    patience = 20
    print("Training LSTM...")
    train_model(
        model_lstm,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=lr,
        patience=patience,
    )

    # 4.7 Train Transformer
    print("Training Transformer...")
    train_model(
        model_transformer,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=lr,
        patience=patience,
    )

    # 4.8 Walk-Forward: LSTM
    print("Walk-forward inference with LSTM...")
    lstm_preds_scaled = walk_forward_predict_nn(
        model_lstm, all_scaled, history_size, train_end
    )
    # 4.9 Walk-Forward: Transformer
    print("Walk-forward inference with Transformer...")
    transformer_preds_scaled = walk_forward_predict_nn(
        model_transformer, all_scaled, history_size, train_end
    )

    # 4.10 Invert scaling for predictions
    # The last column is the target
    # So we just invert using 'scaler_target'
    lstm_preds = scaler_target.inverse_transform(
        lstm_preds_scaled.reshape(-1, 1)
    ).flatten()
    transformer_preds = scaler_target.inverse_transform(
        transformer_preds_scaled.reshape(-1, 1)
    ).flatten()

    # The *actual* test portion in original scale:
    y_test_actual = df_target[train_end:]  # shape (test_size,)
    test_dates = df["date"][train_end:]

    # 4.11 Metrics
    def rmse(a, b):
        return np.sqrt(mean_squared_error(a, b))

    lstm_mae = mean_absolute_error(y_test_actual, lstm_preds)
    lstm_rmse = rmse(y_test_actual, lstm_preds)

    trans_mae = mean_absolute_error(y_test_actual, transformer_preds)
    trans_rmse = rmse(y_test_actual, transformer_preds)

    print(f"LSTM        MAE: {lstm_mae:.3f}, RMSE: {lstm_rmse:.3f}")
    print(f"Transformer MAE: {trans_mae:.3f}, RMSE: {trans_rmse:.3f}")

    # 4.12 Plot
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, y_test_actual, label="Actual", color="black")
    plt.plot(
        test_dates, lstm_preds, label="LSTM (Walk-forward)", color="red", linestyle="--"
    )
    plt.plot(
        test_dates,
        transformer_preds,
        label="Transformer (Walk-forward)",
        color="green",
        linestyle="--",
    )
    plt.title("Walk-forward LSTM vs. Transformer vs. Actual")
    plt.xlabel("Date")
    plt.ylabel("y1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
