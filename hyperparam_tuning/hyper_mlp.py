#!/usr/bin/env python3
"""
hyper_mlp_custom_wrapper.py

Hyperparam tuning for a multi-output MLP using a custom scikit-learn wrapper
(bypasses skorch).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
)
import torch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the custom PyTorch regressor
from utils.pytorch_regressor import PytorchRegressor


def main():
    ###################################
    # 1. Load data
    ###################################
    file_path = "data/vannkraft_data.txt"
    df = pd.read_csv(file_path, sep="\t")

    X_cols = [f"x{i}" for i in range(1, 11)]
    y_cols = [f"y{i}" for i in range(1, 5)]

    X = df[X_cols].values.astype(np.float32)
    Y = df[y_cols].values.astype(np.float32)

    # Optional scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    ###################################
    # 2. Define param distribution
    ###################################
    param_dist = {
        "hidden_size": [64, 128, 256, 512, 1024, 2048],
        "n_hidden_layers": [1, 2, 3, 4, 5],
        "dropout": [0.0, 0.2],
        "lr": [1e-2, 1e-3, 1e-4],
        "max_epochs": [20, 40],
        "batch_size": [256, 1024],
    }

    ###################################
    # 3. Create the regressor
    ###################################
    # Automatically select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    reg = PytorchRegressor(
        input_dim=10,  # x1..x10
        output_dim=4,  # y1..y4
        verbose=True,
        device=device,
    )

    ###################################
    # 4. Define a scoring function
    ###################################
    # By default, our regressor's score() is negative MSE,
    # but let's define a custom "negative RMSE" or "negative MAE" if we prefer.

    def rmse_multi_output(y_true, y_pred):
        # Flatten or do a mean across outputs
        return root_mean_squared_error(y_true, y_pred)

    rmse_scorer = make_scorer(rmse_multi_output, greater_is_better=False)

    ###################################
    # 5. RandomizedSearchCV
    ###################################
    from sklearn.model_selection import RandomizedSearchCV

    search = RandomizedSearchCV(
        reg,
        param_distributions=param_dist,
        n_iter=5,  # or more for thorough search
        scoring=rmse_scorer,
        cv=3,
        verbose=1,
        random_state=42,
    )

    search.fit(X_train, Y_train)

    print("\nBest score (negative RMSE):", search.best_score_)
    print("Best params:", search.best_params_)

    best_reg = search.best_estimator_

    ###################################
    # 6. Evaluate on test set
    ###################################
    Y_pred_test = best_reg.predict(X_test)

    test_rmse = rmse_multi_output(Y_test, Y_pred_test)
    test_mae = mean_absolute_error(Y_test, Y_pred_test)
    test_r2 = r2_score(Y_test, Y_pred_test, multioutput="uniform_average")
    # 'uniform_average' means average R² across all outputs

    print(f"\nTest RMSE: {test_rmse:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test R² (avg): {test_r2:.4f}")


if __name__ == "__main__":
    main()
