"""
test_auto_reg_comparison.py

This script:
1. Loads the data from "data/tahps_data.txt"
2. Creates y-lag columns (for consistent comparison)
3. Splits into train/test
4. Uses a MANUAL AR approach (ordinary least squares on lagged y)
5. Uses statsmodels AutoReg
6. Plots predictions from both on one chart vs. the actual test data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg


##################################
# 1. Utility: Manual AR Functions
##################################
def fit_ar_model(y_train, lags=5):
    """
    Fit a basic autoregressive model:
      y_t = c + phi_1 * y_{t-1} + ... + phi_lags * y_{t-lags}
    using ordinary least squares. Returns (beta, last_obs).
    """
    T = len(y_train)
    X_list = []
    Y_list = []

    # for t in range(lags, T): we form a row with [1, y_{t-1}, ..., y_{t-lags}]
    for t in range(lags, T):
        row = [1.0]  # intercept
        for lag_i in range(1, lags + 1):
            row.append(y_train[t - lag_i])
        X_list.append(row)
        Y_list.append(y_train[t])
    X = np.array(X_list)
    Y = np.array(Y_list)

    # Solve with least squares
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)

    # last_obs -> the last lags from training
    last_obs = y_train[-lags:].copy()

    return beta, last_obs


def predict_ar(y_test, beta, last_obs, lags=5):
    """
    Iterative multi-step forecast for the test period.
    buffer = last_obs (from end of train).
    Each new forecast is appended to buffer.
    """
    c = beta[0]
    phi = beta[1:]  # shape (lags,)

    buffer = list(last_obs)  # We'll keep appending
    preds = []

    for _ in range(len(y_test)):
        y_hat = c
        # y_hat += sum_i( phi[i] * buffer[-(i+1)] )
        for i in range(lags):
            y_hat += phi[i] * buffer[-(i + 1)]
        preds.append(y_hat)
        buffer.append(y_hat)  # next step forecast depends on predicted y

    return np.array(preds)


##########################
# 2. Load & Prep the Data
##########################
def load_data(file_path="data/tahps_data.txt", max_lag=5, test_ratio=0.1):
    # Read file
    df = pd.read_csv(file_path, delimiter="\t", skiprows=1)
    df.columns = ["date", "x1", "x2", "x3", "y1"]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

    # Add y-lag columns (if we want consistent indexing)
    for lag in range(1, max_lag + 1):
        df[f"y_lag_{lag}"] = df["y1"].shift(lag)

    # Drop the first 'max_lag' rows that have NaNs
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Basic train/test split on y only (for AR approach)
    n = len(df)
    train_size = int(n * (1 - test_ratio))

    y = df["y1"].values
    y_train = y[:train_size]
    y_test = y[train_size:]
    dates = df["date"]
    dates_train = dates[:train_size]
    dates_test = dates[train_size:]

    return y_train, y_test, dates_train, dates_test


def main():
    # Load data
    y_train, y_test, dates_train, dates_test = load_data(max_lag=5)

    print(f"Train set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")

    #####################################
    # 3. Manual AR
    #####################################
    lags = 5
    beta, last_obs = fit_ar_model(y_train, lags=lags)
    manual_preds = predict_ar(y_test, beta, last_obs, lags=lags)

    #####################################
    # 4. statsmodels AutoReg
    #####################################
    # We'll fit on y_train with 5 lags
    ar_model = AutoReg(y_train, lags=lags, old_names=False).fit()
    # Predict
    # The model's internal index is 0..(len(y_train)-1).
    # So the next index is len(y_train) for the first test point.
    stats_preds = ar_model.predict(
        start=len(y_train), end=len(y_train) + len(y_test) - 1, dynamic=False
    )

    # Compute some metrics
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error

    manual_mae = mean_absolute_error(y_test, manual_preds)
    manual_rmse = root_mean_squared_error(y_test, manual_preds)

    stats_mae = mean_absolute_error(y_test, stats_preds)
    stats_rmse = root_mean_squared_error(y_test, stats_preds)

    print("==== Manual AR(5) ====")
    print(f"Coefficients: {beta}")  # c, phi1..phi5
    print(f"MAE:  {manual_mae:.3f}")
    print(f"RMSE: {manual_rmse:.3f}")
    print()

    print("==== Statsmodels AR(5) ====")
    print(ar_model.summary())
    print(f"MAE:  {stats_mae:.3f}")
    print(f"RMSE: {stats_rmse:.3f}")

    #####################################
    # 5. Plot both + actual
    #####################################
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label="Actual", color="black", linewidth=2)
    plt.plot(
        dates_test, manual_preds, label="Manual AR(5)", color="red", linestyle="--"
    )
    plt.plot(
        dates_test,
        stats_preds,
        label="Statsmodels AR(5)",
        color="orange",
        linestyle="--",
    )

    plt.title("Comparison: Manual AR(5) vs Statsmodels AR(5) vs Actual")
    plt.xlabel("Date")
    plt.ylabel("y1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
