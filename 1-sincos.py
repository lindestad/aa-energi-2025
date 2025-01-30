import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import (
    Axes3D,
)  # just for side-effects so '3d' is recognized in add_subplot

# -------------------------------------------------
# 1) LOAD THE DATA
# -------------------------------------------------
# X has shape (N, 2) => two input features
# y has shape (N,) => single target
X = np.loadtxt("data/X_sincos.txt")  # e.g. lines of "x  i"
y = np.loadtxt("data/y_sincos.txt")  # e.g. lines of "Z"

# Convert to torch tensors
X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # shape (N,1)


# -------------------------------------------------
# 2) DEFINE SMALL AND LARGE MLPs
# -------------------------------------------------
class SmallMLP(nn.Module):
    def __init__(self):
        super(SmallMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.model(x)


class LargeMLP(nn.Module):
    def __init__(self):
        super(LargeMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.model(x)


# -------------------------------------------------
# 3) TRAIN FUNCTION
# -------------------------------------------------
def train_model(model, X, y, epochs=2000, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        # Forward pass
        preds = model(X)
        loss = criterion(preds, y)

        # Backward pass & update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress occasionally
        if (epoch + 1) % (epochs // 5) == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.6f}")


# -------------------------------------------------
# 4) TRAIN BOTH MLPs
# -------------------------------------------------
small_mlp = SmallMLP()
large_mlp = LargeMLP()

print("Training Small MLP:")
train_model(small_mlp, X_torch, y_torch, epochs=1000, lr=1e-3)

print("\nTraining Large MLP (overfitting):")
train_model(large_mlp, X_torch, y_torch, epochs=3000, lr=1e-3)

# -------------------------------------------------
# 5) MAKE PREDICTIONS
# -------------------------------------------------
small_mlp.eval()
large_mlp.eval()

with torch.no_grad():
    y_pred_small = small_mlp(X_torch).numpy().flatten()
    y_pred_large = large_mlp(X_torch).numpy().flatten()

# -------------------------------------------------
# 6) RESHAPE FOR 3D PLOTTING
# -------------------------------------------------
# If your data is truly a grid in x & i, you can do:
x_vals = np.unique(X[:, 0])
i_vals = np.unique(X[:, 1])

# We expect len(X) = len(x_vals)*len(i_vals)
# Make sure the data is sorted or reorder X, y accordingly.
# A quick approach: sort everything by (x, i) so that the reshape lines up
sort_idx = np.lexsort((X[:, 1], X[:, 0]))  # sort by x first, then by i
X_sorted = X[sort_idx]
y_sorted = y[sort_idx]
y_pred_small_sorted = y_pred_small[sort_idx]
y_pred_large_sorted = y_pred_large[sort_idx]

# Now we reshape
X_grid = X_sorted[:, 0].reshape(len(x_vals), len(i_vals))
I_grid = X_sorted[:, 1].reshape(len(x_vals), len(i_vals))
Z_true = y_sorted.reshape(len(x_vals), len(i_vals))
Z_small = y_pred_small_sorted.reshape(len(x_vals), len(i_vals))
Z_large = y_pred_large_sorted.reshape(len(x_vals), len(i_vals))

# -------------------------------------------------
# 7) PLOT EVERYTHING IN 3D
# -------------------------------------------------
fig = plt.figure(figsize=(18, 5))

# -- Ground Truth
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
surf1 = ax1.plot_surface(X_grid, I_grid, Z_true, cmap="viridis", edgecolor="none")
ax1.set_title("Ground Truth")
ax1.set_xlabel("X")
ax1.set_ylabel("I")
ax1.set_zlabel("Z")
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# -- Small MLP Predictions
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
surf2 = ax2.plot_surface(X_grid, I_grid, Z_small, cmap="plasma", edgecolor="none")
ax2.set_title("Small MLP Predictions")
ax2.set_xlabel("X")
ax2.set_ylabel("I")
ax2.set_zlabel("Z")
fig.colorbar(surf2, ax=ax2, shrink=0.5)

# -- Large MLP Predictions
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
surf3 = ax3.plot_surface(X_grid, I_grid, Z_large, cmap="cividis", edgecolor="none")
ax3.set_title("Large MLP Predictions")
ax3.set_xlabel("X")
ax3.set_ylabel("I")
ax3.set_zlabel("Z")
fig.colorbar(surf3, ax=ax3, shrink=0.5)

plt.tight_layout()
plt.show()
