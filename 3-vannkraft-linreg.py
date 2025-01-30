"""
3-vannkraft_linreg.py

Trains a multi-output Linear Regression model on x1..x10 -> y1..y4.
Generates:
  - Bar plots of MSE and MAE for each target.
  - (Optional) scatter plots of actual vs. predicted, if you want.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
# ----------------------------------------------------------------------------
file_path = "data/vannkraft_data.txt"
df = pd.read_csv(file_path, sep="\t")

X_cols = [f"x{i}" for i in range(1, 11)]
y_cols = [f"y{i}" for i in range(1, 5)]

X = df[X_cols].values
Y = df[y_cols].values

# Optional scaling (often helps for interpretability in LR or might not matter much)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 2. TRAIN MULTI-OUTPUT LINEAR REGRESSION
# ----------------------------------------------------------------------------
linreg = LinearRegression()
linreg.fit(X_train, Y_train)
Y_pred = linreg.predict(X_test)

# 3. EVALUATE PER TARGET
# ----------------------------------------------------------------------------
results = []
for i, target in enumerate(y_cols):
    mse = mean_squared_error(Y_test[:, i], Y_pred[:, i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test[:, i], Y_pred[:, i])
    results.append({"target": target, "MSE": mse, "RMSE": rmse, "MAE": mae})

df_metrics = pd.DataFrame(results)
print("\n=== Linear Regression Metrics ===")
print(df_metrics)

# 4. PLOTS
# ----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")

# A. Bar plots for MSE & MAE
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.barplot(data=df_metrics, x="target", y="MSE", ax=axes[0], color="#4C72B0")
axes[0].set_title("MSE by Target")

sns.barplot(data=df_metrics, x="target", y="MAE", ax=axes[1], color="#55A868")
axes[1].set_title("MAE by Target")

plt.tight_layout()
plt.show()

# B. Scatter plots
plt.figure(figsize=(12, 4))

for i, target in enumerate(y_cols):
    plt.subplot(1, len(y_cols), i + 1)
    y_true = Y_test[:, i]
    y_hat = Y_pred[:, i]

    sns.scatterplot(x=y_true, y=y_hat, color="#C44E52", alpha=0.4)

    min_val = min(y_true.min(), y_hat.min())
    max_val = max(y_true.max(), y_hat.max())
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="#FF7F0E", label="y=x")

    plt.title(f"{target}\nActual vs. Pred")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

plt.tight_layout()
plt.show()
