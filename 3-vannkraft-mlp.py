"""
3-vannkraft-mlp.py

Trains a multi-output PyTorch MLP on x1..x10 -> y1..y4, using best-found hyperparams.
Plots:
  1) Actual vs. Predicted scatter for each y
  2) Bar plots of MSE / MAE for each y

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATA
# ----------------------------------------------------------------------------
file_path = "data/vannkraft_data.txt"
df = pd.read_csv(file_path, sep="\t")

X_cols = [f"x{i}" for i in range(1, 11)]
y_cols = [f"y{i}" for i in range(1, 5)]

X = df[X_cols].values.astype(np.float32)
Y = df[y_cols].values.astype(np.float32)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Optional: scale the inputs for better NN performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# 2. DEFINE MLP MODEL WITH BEST HYPERPARAMS (FOUND EARLIER)
# ----------------------------------------------------------------------------
# Let's say you found these to be best from your hyperparam search:
BEST_HIDDEN_SIZE = 512  # e.g. 128
BEST_N_HIDDEN_LAYERS = 2  # e.g. 3
BEST_DROPOUT = 0.01  # e.g. 0.2
BEST_LR = 0.001  # e.g. 0.001
BEST_MAX_EPOCHS = 500  # Early stopping
BEST_BATCH_SIZE = 256  # e.g. 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, n_layers, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Create the model
model = MLP(
    input_dim=len(X_cols),
    output_dim=len(y_cols),
    hidden_size=BEST_HIDDEN_SIZE,
    n_layers=BEST_N_HIDDEN_LAYERS,
    dropout=BEST_DROPOUT,
).to(DEVICE)

# 3. TRAINING SETUP
# ----------------------------------------------------------------------------
optimizer = optim.Adam(model.parameters(), lr=BEST_LR)
criterion = nn.MSELoss()

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_train, device=DEVICE), torch.tensor(Y_train, device=DEVICE)
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BEST_BATCH_SIZE, shuffle=True
)

# 4. TRAIN THE MODEL WITH EARLY STOPPING AND BEST MODEL RESTORATION
# ----------------------------------------------------------------------------
model.train()

patience = 35  # Number of epochs with no improvement before stopping
best_loss = float("inf")
epochs_no_improve = 0
best_model_state = None  # Store the best model weights

for epoch in range(BEST_MAX_EPOCHS):
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)

    # Compute average training loss
    avg_loss = epoch_loss / len(train_loader.dataset)

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, device=DEVICE)
        preds_test_t = model(X_test_t)
        val_loss = criterion(preds_test_t, torch.tensor(Y_test, device=DEVICE)).item()

    model.train()  # Switch back to training mode

    # Print progress
    print(
        f"Epoch {epoch+1}/{BEST_MAX_EPOCHS}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}"
    )

    # Check for improvement
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()  # Save the best model state
    else:
        epochs_no_improve += 1

    # Early stopping check
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Restore best model before evaluation
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Best model restored.")

# 5. EVALUATION (after best model is restored)
# ----------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    X_test_t = torch.tensor(X_test, device=DEVICE)
    preds_test_t = model(X_test_t)
preds_test = preds_test_t.cpu().numpy()  # shape (N, 4)

# 6. COMPUTE METRICS (PER TARGET)
# ----------------------------------------------------------------------------
rows = []
for i, target in enumerate(y_cols):
    mse = mean_squared_error(Y_test[:, i], preds_test[:, i])
    mae = mean_absolute_error(Y_test[:, i], preds_test[:, i])
    rows.append({"target": target, "MSE": mse, "MAE": mae})

df_metrics = pd.DataFrame(rows)
df_metrics["RMSE"] = np.sqrt(df_metrics["MSE"])

print(df_metrics)

# 7. PLOTS
# ----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")

## 7A. Scatter: Actual vs. Pred for each target
plt.figure(figsize=(12, 4))
num_targets = len(y_cols)
for i, target in enumerate(y_cols):
    plt.subplot(1, num_targets, i + 1)
    y_true = Y_test[:, i]
    y_pred = preds_test[:, i]

    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4)
    # 1:1 line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")

    plt.title(f"{target}\nActual vs. Pred")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

plt.tight_layout()
plt.show()

## 7B. Bar plots of MSE & MAE
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.barplot(data=df_metrics, x="target", y="MSE", ax=axes[0], palette="Blues_d")
axes[0].set_title("MSE by Target")

sns.barplot(data=df_metrics, x="target", y="MAE", ax=axes[1], palette="Greens_d")
axes[1].set_title("MAE by Target")

plt.tight_layout()
plt.show()
