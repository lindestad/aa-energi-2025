"""
hyper_xgboost.py

Hyperparameter tuning for XGBoost on one target variable (e.g., y1) using RandomizedSearchCV.

Usage:
  python xgb_optimize.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, root_mean_squared_error
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


######################################
# Main loop
######################################
targets = ["y1", "y2", "y3", "y4"]
for target in targets:
    ######################################
    # 1. LOAD AND PREPARE DATA
    ######################################
    file_path = "data/vannkraft_data.txt"  # Adjust path as needed
    df = pd.read_csv(file_path, sep="\t")

    # Features: x1..x10
    X_cols = [f"x{i}" for i in range(1, 11)]
    target_col = target

    X = df[X_cols].values
    y = df[target_col].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ######################################
    # 2. DEFINE A SCORER (RMSE)
    ######################################
    def rmse(y_true, y_pred):
        """Calculate Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    ######################################
    # 3. SET UP RANDOMIZED SEARCH
    ######################################
    # Choose a parameter distribution to explore:
    param_dist = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.5, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.5, 1],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [1, 10, 50],
    }

    # Initialize a base XGBoost regressor (for single-target)
    xgb_reg = XGBRegressor(tree_method="hist", random_state=42)

    # Implement manual cross-validation search
    def random_search_cv(model, param_grid, X, y, n_iter=20, cv=3):
        best_score = float("inf")
        best_params = None

        # Generate random parameter combinations
        param_list = []
        for _ in range(n_iter):
            params = {k: np.random.choice(v) for k, v in param_grid.items()}
            param_list.append(params)

        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        for params in param_list:
            scores = []
            # Cross validation
            for train_idx, val_idx in kf.split(X):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]

                model_cv = XGBRegressor(**params, tree_method="hist", random_state=42)
                model_cv.fit(X_train_cv, y_train_cv)
                y_pred = model_cv.predict(X_val_cv)
                score = rmse(y_val_cv, y_pred)
                scores.append(score)

            mean_score = np.mean(scores)
            if mean_score < best_score:
                best_score = mean_score
                best_params = params

        return best_params, best_score

    ######################################
    # 4. RUN THE SEARCH
    ######################################
    best_params, best_score = random_search_cv(
        xgb_reg, param_dist, X_train, y_train, n_iter=20, cv=3
    )

    print("\nTARGET:", target)
    print("Best RMSE:", best_score)
    print("Best Parameters:", best_params)

    # Retrieve the best model
    best_model = XGBRegressor(**best_params, tree_method="hist", random_state=42)
    best_model.fit(X_train, y_train)

    ######################################
    # 5. EVALUATE ON TEST SET
    ######################################
    from sklearn.metrics import mean_absolute_error

    y_pred_test = best_model.predict(X_test)
    test_rmse = rmse(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"\nTest RMSE: {test_rmse:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
