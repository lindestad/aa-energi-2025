"""
2-tahps.py

This script:
1) Loads the data (including time-based features).
2) Implements:
   - Naive AR (y_hat(t+1) = y(t)),
   - Manual AR(5),
   - LSTM,
   - Transformer,
   all in a walk-forward or standard approach as appropriate.
3) Produces four plots:
   1) Full timeline: Actual vs. Naive AR, AR(5), LSTM, Transformer.
   2) Full timeline: Absolute error lines for each method.
   3) Bar chart: MAE.
   4) Bar chart: MSE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ================ AR(naive), AR(5) UTILS ================
def naive_predict(y_train, y_test):
    """
    Naive approach: y_hat(t+1) = y(t).
    For test step i, we use the actual y_test[i-1] if i>0, or y_train[-1] for i=0.
    """
    preds = []
    last_value = y_train[-1]
    for i in range(len(y_test)):
        preds.append(last_value)
        last_value = y_test[i]  # update with actual
    return np.array(preds)


def fit_ar_model(y_train, lags=5):
    """
    Fit AR(5) via ordinary least squares:
      y_t = c + sum_{i=1..5} phi_i * y_{t-i}.
    Returns (beta, last_obs).
    """
    T = len(y_train)
    X_list = []
    Y_list = []
    for t in range(lags, T):
        row = [1.0]  # intercept
        for lag_i in range(1, lags + 1):
            row.append(y_train[t - lag_i])
        X_list.append(row)
        Y_list.append(y_train[t])
    X = np.array(X_list)
    Y = np.array(Y_list)

    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    last_obs = y_train[-lags:].copy()
    return beta, last_obs


def predict_ar_one_step(y_test, beta, last_obs, lags=5):
    """
    Walk-forward AR(5) forecast, each day using real previous data.
    """
    c = beta[0]
    phi = beta[1:]
    buffer = list(last_obs)
    preds = []
    for i in range(len(y_test)):
        row = [1.0]
        for lag_i in range(1, lags + 1):
            row.append(buffer[-lag_i])
        row = np.array(row)

        y_hat = c
        for j in range(lags):
            y_hat += phi[j] * row[j + 1]
        preds.append(y_hat)
        # update buffer with the real y_test[i]
        buffer.append(y_test[i])
        if len(buffer) > lags:
            buffer.pop(0)
    return np.array(preds)


# ================ LSTM & Transformer MODELS ================
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
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        out = self.fc_out(x)
        return out


def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-3, patience=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_weights = None
    no_improve_count = 0

    for epoch in range(num_epochs):
        # TRAIN
        model.train()
        train_loss_sum = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X).squeeze()
            loss = criterion(preds, batch_y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_X.size(0)
        epoch_train_loss = train_loss_sum / len(train_loader.dataset)

        # VALIDATE
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                preds = model(batch_X).squeeze()
                vloss = criterion(preds, batch_y.squeeze())
                val_loss_sum += vloss.item() * batch_X.size(0)
        epoch_val_loss = val_loss_sum / len(val_loader.dataset)

        # Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Print
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} -> TrainLoss={epoch_train_loss:.4f}, ValLoss={epoch_val_loss:.4f}"
            )

    # load best
    if best_weights is not None:
        model.load_state_dict(best_weights)


# ================ MAIN SCRIPT ================
def main():
    # -------------------------
    # A) LOAD DATA
    # -------------------------
    file_path = "data/tahps_data.txt"
    df = pd.read_csv(file_path, delimiter="\t", skiprows=1)
    df.columns = ["date", "x1", "x2", "x3", "y1"]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

    # Add time-based features
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # AR data
    # We'll do naive & AR(5) on the raw y
    y_full = df["y1"].values

    # We'll do a 90/10 train test
    n = len(df)
    train_size = int(n * 0.9)
    y_train_ar = y_full[:train_size]
    y_test_ar = y_full[train_size:]
    dates_train = df["date"][:train_size]
    dates_test = df["date"][train_size:]

    # ---- AR(naive) + AR(5)
    naive_preds = naive_predict(y_train_ar, y_test_ar)
    beta, last_obs = fit_ar_model(y_train_ar, lags=5)
    ar5_preds = predict_ar_one_step(y_test_ar, beta, last_obs, lags=5)

    # Evaluate
    naive_mae = mean_absolute_error(y_test_ar, naive_preds)
    naive_mse = mean_squared_error(y_test_ar, naive_preds)

    ar5_mae = mean_absolute_error(y_test_ar, ar5_preds)
    ar5_mse = mean_squared_error(y_test_ar, ar5_preds)

    # -------------------------
    # B) LSTM & Transformer
    # We'll build a small dataset with exogenous features + y
    # We'll do standard overlap sequences for training, then do a single pass test approach
    # (Alternatively, we can do walk-forward for LSTM, but let's keep it simpler.)
    # -------------------------
    feature_cols = ["x1", "x2", "x3", "dayofyear", "dayofweek", "month"]
    df_features = df[feature_cols].values
    df_target = df["y1"].values.reshape(-1, 1)

    # Scale
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df_features)
    y_scaled = scaler_y.fit_transform(df_target)
    # We'll store scaled into arrays so we can build sequences
    X = X_scaled
    y = y_scaled

    # Train/test split
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    # We'll build sequences for the entire train set, and entire test set, then feed the test set in one pass
    history_size = 30

    def create_sequences(X_data, y_data, seq_len=30):
        X_seq = []
        y_seq = []
        for i in range(len(X_data) - seq_len):
            xw = X_data[i : i + seq_len]
            yw = y_data[i + seq_len]
            X_seq.append(xw)
            y_seq.append(yw)
        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, history_size)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, history_size)

    # Torch dataset
    class TSData(torch.utils.data.Dataset):
        def __init__(self, X, Y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.Y = torch.tensor(Y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    train_dataset = TSData(X_train_seq, y_train_seq)
    test_dataset = TSData(X_test_seq, y_test_seq)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # We'll do a small validation split from train
    val_split = int(0.8 * len(train_dataset))
    val_dataset = TSData(X_train_seq[val_split:], y_train_seq[val_split:])
    train_dataset = TSData(X_train_seq[:val_split], y_train_seq[:val_split])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Models
    input_dim = X_train_seq.shape[2]
    lstm_model = LSTMModel(input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
    transformer_model = TransformerTimeSeries(
        input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2
    )

    # Train LSTM
    print("Training LSTM...")
    train_model(
        lstm_model, train_loader, val_loader, num_epochs=500, lr=1e-4, patience=25
    )

    # Train Transformer
    print("Training Transformer...")
    train_model(
        transformer_model,
        train_loader,
        val_loader,
        num_epochs=500,
        lr=1e-4,
        patience=25,
    )

    # Evaluate on test set in one pass
    #  - We'll just feed the entire X_test_seq
    # In a "strict" sense, you might do day-by-day walk-forward, but let's keep it simpler here.

    # LSTM
    lstm_model.eval()
    lstm_preds_scaled = []
    y_test_scaled_all = []
    with torch.no_grad():
        for batch_X, batch_y in DataLoader(test_dataset, batch_size=32, shuffle=False):
            p = lstm_model(batch_X).squeeze().cpu().numpy()
            lstm_preds_scaled.append(p)
            y_test_scaled_all.append(batch_y.cpu().numpy())
    lstm_preds_scaled = np.concatenate(lstm_preds_scaled, axis=0)
    y_test_scaled_all = np.concatenate(y_test_scaled_all, axis=0)
    # invert scale
    lstm_preds_ = scaler_y.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()
    y_test_ = scaler_y.inverse_transform(y_test_scaled_all.reshape(-1, 1)).flatten()

    lstm_mae = mean_absolute_error(y_test_, lstm_preds_)
    lstm_mse = mean_squared_error(y_test_, lstm_preds_)

    # Transformer
    transformer_model.eval()
    trans_preds_scaled = []
    y_test_scaled_all2 = []
    with torch.no_grad():
        for batch_X, batch_y in DataLoader(test_dataset, batch_size=32, shuffle=False):
            p = transformer_model(batch_X).squeeze().cpu().numpy()
            trans_preds_scaled.append(p)
            y_test_scaled_all2.append(batch_y.cpu().numpy())
    trans_preds_scaled = np.concatenate(trans_preds_scaled, axis=0)
    y_test_scaled_all2 = np.concatenate(y_test_scaled_all2, axis=0)
    trans_preds_ = scaler_y.inverse_transform(
        trans_preds_scaled.reshape(-1, 1)
    ).flatten()
    trans_mae = mean_absolute_error(y_test_, trans_preds_)
    trans_mse = mean_squared_error(y_test_, trans_preds_)

    # We'll align the timeline for the neural net predictions
    # The test portion minus the first 30 days
    test_dates_for_plot = dates_test[history_size:].values
    # AR / naive are for the entire test, but let's also align them
    # We'll keep the last len(test_dates_for_plot) points of each AR series
    AR_length = len(y_test_ar)
    needed_len = len(test_dates_for_plot)  # test_size - 30
    # AR(naive) predictions: naive_preds, shape=(test_size,)
    # We'll slice from (test_size - needed_len) to end
    naive_aligned = naive_preds[-needed_len:]
    ar5_aligned = ar5_preds[-needed_len:]
    y_test_aligned = y_test_ar[-needed_len:]  # actual

    # Now let's do the final 4 metrics (for the aligned portion)
    # we already have full test metrics for AR(naive) & AR(5) above,
    # but let's do them for the aligned portion to show the final plot consistently
    naive_mae_aligned = mean_absolute_error(y_test_aligned, naive_aligned)
    naive_mse_aligned = mean_squared_error(y_test_aligned, naive_aligned)
    ar5_mae_aligned = mean_absolute_error(y_test_aligned, ar5_aligned)
    ar5_mse_aligned = mean_squared_error(y_test_aligned, ar5_aligned)
    lstm_mae_aligned = mean_absolute_error(y_test_aligned, lstm_preds_)
    lstm_mse_aligned = mean_squared_error(y_test_aligned, lstm_preds_)
    trans_mae_aligned = mean_absolute_error(y_test_aligned, trans_preds_)
    trans_mse_aligned = mean_squared_error(y_test_aligned, trans_preds_)

    # =====================
    # 1) Timeline Plot
    # =====================
    plt.figure(figsize=(12, 6))
    plt.plot(
        test_dates_for_plot,
        y_test_aligned,
        label="Actual",
        color="#333333",
        linewidth=2,
    )
    plt.plot(
        test_dates_for_plot,
        naive_aligned,
        label="Naive AR",
        color="orange",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        test_dates_for_plot,
        ar5_aligned,
        label="AR(5)",
        color="blue",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        test_dates_for_plot,
        lstm_preds_,
        label="LSTM",
        color="red",
        linestyle="-",
        linewidth=1.8,
    )
    plt.plot(
        test_dates_for_plot,
        trans_preds_,
        label="Transformer",
        color="green",
        linestyle="-",
        linewidth=1.8,
    )
    plt.title("Full Timeline: Predictions vs Actual", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("y1", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # =====================
    # 2) Absolute Error over Time
    # =====================
    ae_naive = np.abs(naive_aligned - y_test_aligned)
    ae_ar5 = np.abs(ar5_aligned - y_test_aligned)
    ae_lstm = np.abs(lstm_preds_ - y_test_aligned)
    ae_trans = np.abs(trans_preds_ - y_test_aligned)

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates_for_plot, ae_naive, label="Naive AR AE", color="orange")
    plt.plot(test_dates_for_plot, ae_ar5, label="AR(5) AE", color="blue")
    plt.plot(test_dates_for_plot, ae_lstm, label="LSTM AE", color="red")
    plt.plot(test_dates_for_plot, ae_trans, label="Transformer AE", color="green")
    plt.title("Absolute Error Over Time", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Absolute Error", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # =====================
    # 3) Bar Chart - MAE
    # =====================
    mae_vals = [naive_mae_aligned, ar5_mae_aligned, lstm_mae_aligned, trans_mae_aligned]
    model_names = ["Naive AR", "AR(5)", "LSTM", "Transformer"]
    x = np.arange(len(model_names))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, mae_vals, color=["orange", "blue", "red", "green"], alpha=0.7)
    plt.xticks(x, model_names)
    plt.ylabel("MAE", fontsize=12)
    plt.title("MAE (Aligned Test Portion)", fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for idx, b in enumerate(bars):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.01,
            f"{b.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.show()

    # =====================
    # 4) Bar Chart - MSE
    # =====================
    mse_vals = [naive_mse_aligned, ar5_mse_aligned, lstm_mse_aligned, trans_mse_aligned]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, mse_vals, color=["orange", "blue", "red", "green"], alpha=0.7)
    plt.xticks(x, model_names)
    plt.ylabel("MSE", fontsize=12)
    plt.title("MSE (Aligned Test Portion)", fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for idx, b in enumerate(bars):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.01,
            f"{b.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.tight_layout()
    plt.show()

    print("===== Final Aligned Metrics =====")
    print(f"Naive AR:      MAE={naive_mae_aligned:.3f}, MSE={naive_mse_aligned:.3f}")
    print(f"AR(5):         MAE={ar5_mae_aligned:.3f}, MSE={ar5_mse_aligned:.3f}")
    print(f"LSTM:          MAE={lstm_mae_aligned:.3f}, MSE={lstm_mse_aligned:.3f}")
    print(f"Transformer:   MAE={trans_mae_aligned:.3f}, MSE={trans_mse_aligned:.3f}")


if __name__ == "__main__":
    main()
