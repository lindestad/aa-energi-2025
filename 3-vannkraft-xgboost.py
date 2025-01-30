#!/usr/bin/env python3
"""
train_compare_xgb_linreg.py

Compares XGBoost (with best hyperparams per target) vs. Linear Regression on y1..y4.
Calculates RMSE, MAE, and R^2. Plots grouped bar charts for each metric + scatter plots (XGB).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

# 1. LOAD DATA
# ----------------------------------------------------------------------------
file_path = "data/vannkraft_data.txt"
df = pd.read_csv(file_path, sep="\t")

X_cols = [f"x{i}" for i in range(1, 11)]
y_cols = [f"y{i}" for i in range(1, 5)]

X = df[X_cols].values
Y = df[y_cols].values  # shape: (n_samples, 4)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 2. DEFINE BEST XGB PARAMS (PER TARGET) FROM YOUR SEARCH
# ----------------------------------------------------------------------------
best_params = {
    "y1": {
        "n_estimators": 500,
        "max_depth": 9,
        "learning_rate": 0.2,
        "subsample": 1.0,
        "colsample_bytree": 0.5,
        "min_child_weight": 5,
        "gamma": 0.0,
        "reg_alpha": 1.0,
        "reg_lambda": 10,
    },
    "y2": {
        "n_estimators": 300,
        "max_depth": 9,
        "learning_rate": 0.05,
        "subsample": 0.5,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.0,
        "reg_alpha": 1.0,
        "reg_lambda": 1,
    },
    "y3": {
        "n_estimators": 500,
        "max_depth": 9,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1,
    },
    "y4": {
        "n_estimators": 100,
        "max_depth": 9,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 5,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 10,
    },
}

# 3. TRAIN LINEAR REGRESSION & XGBOOST, COMPUTE METRICS
# ----------------------------------------------------------------------------
methods = ["LinearReg", "XGBoost"]
rows = []  # to hold dicts of {method, target, MSE, RMSE, MAE, R2}

# 3A. LINEAR REGRESSION
#    We'll train one linear model per target for simplicity,
#    or we can do a multi-output linear reg.
#    scikit-learn's LinearRegression can handle multi-output directly by passing Y_train, Y_test as Nx4.
linreg = LinearRegression()
linreg.fit(X_train, Y_train)
Y_pred_linreg = linreg.predict(X_test)  # shape (N,4)

# Compute metrics target-wise
for i, target in enumerate(y_cols):
    mse_lin = mean_squared_error(Y_test[:, i], Y_pred_linreg[:, i])
    rmse_lin = mse_lin**0.5
    mae_lin = mean_absolute_error(Y_test[:, i], Y_pred_linreg[:, i])
    r2_lin = r2_score(Y_test[:, i], Y_pred_linreg[:, i])

    rows.append(
        {
            "method": "LinearReg",
            "target": target,
            "MSE": mse_lin,
            "RMSE": rmse_lin,
            "MAE": mae_lin,
            "R2": r2_lin,
        }
    )

# 3B. XGBOOST per target
models_xgb = {}
preds_xgb = {}

for i, target in enumerate(y_cols):
    params = best_params[target]
    xgb_model = XGBRegressor(**params, random_state=42, tree_method="hist")
    xgb_model.fit(X_train, Y_train[:, i])
    y_pred = xgb_model.predict(X_test)
    preds_xgb[target] = y_pred
    models_xgb[target] = xgb_model

    mse_xgb = mean_squared_error(Y_test[:, i], y_pred)
    rmse_xgb = mse_xgb**0.5
    mae_xgb = mean_absolute_error(Y_test[:, i], y_pred)
    r2_xgb = r2_score(Y_test[:, i], y_pred)

    rows.append(
        {
            "method": "XGBoost",
            "target": target,
            "MSE": mse_xgb,
            "RMSE": rmse_xgb,
            "MAE": mae_xgb,
            "R2": r2_xgb,
        }
    )

# 4. CREATE A DATAFRAME FOR METRICS
# ----------------------------------------------------------------------------
df_metrics = pd.DataFrame(rows)

# Optional: print or save
print("\nOverall Metrics:\n", df_metrics)

# 5. PLOT GROUPED BAR CHARTS (MSE, MAE, R2) AVOIDING FUTUREWARNING
# ----------------------------------------------------------------------------
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot MSE
sns.barplot(
    data=df_metrics,
    x="target",
    y="MSE",
    hue="method",  # group by method
    ax=axes[0],
    palette="Set2",
)
axes[0].set_title("MSE by Target")
axes[0].set_xlabel("Target")
axes[0].set_ylabel("MSE")

# Plot MAE
sns.barplot(
    data=df_metrics, x="target", y="MAE", hue="method", ax=axes[1], palette="Set2"
)
axes[1].set_title("MAE by Target")
axes[1].set_xlabel("Target")
axes[1].set_ylabel("MAE")

# Plot R2
sns.barplot(
    data=df_metrics, x="target", y="R2", hue="method", ax=axes[2], palette="Set2"
)
axes[2].set_title("R² by Target")
axes[2].set_xlabel("Target")
axes[2].set_ylabel("R²")
axes[2].set_ylim(0, 1)  # optional if you expect all R2 in [0..1]

plt.tight_layout()
plt.show()

# 6. SCATTER PLOTS (ACTUAL VS. PRED) - XGBOOST
#    If you want to compare LinearReg as well, just replicate code with Y_pred_linreg.
# ----------------------------------------------------------------------------
plt.figure(figsize=(12, 4))
num_targets = len(y_cols)

for i, target in enumerate(y_cols):
    y_true = Y_test[:, i]
    y_pred = preds_xgb[target]

    plt.subplot(1, num_targets, i + 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, color="teal")

    # Reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")

    plt.title(f"{target} - XGBoost\nActual vs. Pred")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()

plt.tight_layout()
plt.show()
