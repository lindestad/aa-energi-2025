"""
test_auto_reg_comparison.py

This script:
1. Loads the data from "data/tahps_data.txt"
2. Creates y-lag columns (for consistent comparison)
3. Splits into train/test
4. Implements:
   - A MANUAL AR(5) approach (ordinary least squares on lagged y) with *walk-forward* one-step-ahead
   - Statsmodels AR(5) in a similar one-step-ahead manner (or dynamic=False)
   - A naive model (y_hat(t+1) = y(t))
5. Plots predictions from all three vs. the actual test data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error


# We'll define a root_mean_squared_error for convenience
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


##################################
# 1. Utility: Manual AR Functions
##################################
def fit_ar_model(y_train, lags=5):
    """
    Fit a basic autoregressive model:
      y_t = c + phi_1*y_{t-1} + ... + phi_lags*y_{t-lags}
    using ordinary least squares. Returns (beta, last_obs).
    """
    T = len(y_train)
    X_list = []
    Y_list = []

    # Build design matrix X and target Y for t in [lags..T-1]
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

    # Keep the last 'lags' points as the initial "buffer"
    last_obs = y_train[-lags:].copy()
    return beta, last_obs


def predict_ar_one_step(y_test, beta, last_obs, lags=5):
    """
    One-step-ahead forecast for the test period, using real previous-day values.

    Steps:
      - We start with 'last_obs' from the training set (the last lags).
      - For each i in [0..len(y_test)-1]:
          1. Construct a single design row [1, y_{t-1}, y_{t-2}, ..., y_{t-lags}]
             using the *latest known real values* (initially from train, then updated with test).
          2. Predict y_hat(i).
          3. Append the *actual* y_test[i] to our buffer (so the next iteration sees real data).
    """
    c = beta[0]
    phi = beta[1:]  # shape (lags,)

    # We'll keep track of the last 'lags' real data points in a buffer
    buffer = list(last_obs)
    preds = []

    for i in range(len(y_test)):
        # 1) Build row = [1, buffer[-1], buffer[-2], ...] i.e. last known real values
        row = [1.0]
        for lag_i in range(1, lags + 1):
            row.append(buffer[-lag_i])

        row = np.array(row)

        # 2) Make a 1-step forecast
        y_hat = row[0] * c  # intercept
        for j in range(lags):
            y_hat += phi[j] * row[j + 1]

        preds.append(y_hat)

        # 3) Update the buffer with the *actual* new test data
        buffer.append(y_test[i])  # we do NOT append the predicted value
        # If we want to keep the buffer length = lags, remove the oldest
        if len(buffer) > lags:
            buffer.pop(0)

    return np.array(preds)


###################################
# 2. Naive Baseline Implementation
###################################
def naive_predict(y_train, y_test):
    """
    Naive approach: y_hat(t+1) = y(t).
    For the test step i, we use the actual y_test[i-1] if i>0, or y_train[-1] for i=0.
    """
    preds = []
    last_value = y_train[-1]

    for i in range(len(y_test)):
        preds.append(last_value)
        last_value = y_test[i]  # update to the newly observed real value

    return np.array(preds)


##########################
# 3. Load & Prep the Data
##########################
def load_data(file_path="data/tahps_data.txt", max_lag=5, test_ratio=0.1):
    # Read file
    df = pd.read_csv(file_path, delimiter="\t", skiprows=1)
    df.columns = ["date", "x1", "x2", "x3", "y1"]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

    # Add y-lag columns (for consistent indexing across methods)
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
    # 1) Manual AR(5) with One-Step-Ahead
    #####################################
    lags = 5
    beta, last_obs = fit_ar_model(y_train, lags=lags)
    manual_preds = predict_ar_one_step(y_test, beta, last_obs, lags=lags)

    #####################################
    # 2) statsmodels AutoReg(5) Rolling
    #####################################
    # By default, "dynamic=False" uses the real data for prior steps,
    # but only from training data. If we want a truly step-by-step approach that
    # uses newly observed test values at each step, we can do a manual loop:
    ar_model = AutoReg(y_train, lags=lags, old_names=False).fit()

    # We'll walk forward the test set, each time appending the real observation
    # to an "all_data" array so statsmodels can see it.
    # A simpler approach is just calling predict with dynamic=False for the entire test set,
    # but here's the step-by-step for demonstration:
    all_data = np.concatenate([y_train])  # we will expand this with each real y_test
    stats_preds_rolling = []

    for i in range(len(y_test)):
        # next index is len(all_data)
        start_index = len(all_data)
        end_index = len(all_data)  # same because it's 1-step forecast
        # forecast next step
        forecast = ar_model.predict(start=start_index, end=end_index, dynamic=False)
        y_hat = forecast[0]  # get the single predicted value
        stats_preds_rolling.append(y_hat)

        # append the actual new data
        all_data = np.append(all_data, y_test[i])

    stats_preds_rolling = np.array(stats_preds_rolling)

    #####################################
    # 3) Naive Forecast
    #####################################
    naive_preds = naive_predict(y_train, y_test)

    #####################################
    # 4) Compute Metrics
    #####################################
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    def rmse(a, b):
        return np.sqrt(mean_squared_error(a, b))

    # Manual AR
    manual_mae = mean_absolute_error(y_test, manual_preds)
    manual_rmse = rmse(y_test, manual_preds)

    # Statsmodels AR (step-by-step)
    stats_mae = mean_absolute_error(y_test, stats_preds_rolling)
    stats_rmse = rmse(y_test, stats_preds_rolling)

    # Naive
    naive_mae = mean_absolute_error(y_test, naive_preds)
    naive_rmse = rmse(y_test, naive_preds)

    print("==== Manual AR(5) One-step-ahead ====")
    print(f"Coefficients: {beta}")  # c, phi1..phi5
    print(f"MAE:  {manual_mae:.3f},   RMSE: {manual_rmse:.3f}")
    print()

    print("==== Statsmodels AR(5) Rolling One-step ====")
    print("(We do a stepwise update of the data for each test point, dynamic=False)")
    print(ar_model.summary())
    print(f"MAE:  {stats_mae:.3f},   RMSE: {stats_rmse:.3f}")
    print()

    print("==== Naive Baseline ====")
    print(f"MAE:  {naive_mae:.3f},   RMSE: {naive_rmse:.3f}")
    print()

    #####################################
    # 5) Plot: Manual AR, Statsmodels AR, Naive vs Actual
    #####################################
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label="Actual", color="black", linewidth=2)

    plt.plot(
        dates_test, manual_preds, label="Manual AR(5)", color="red", linestyle="--"
    )
    plt.plot(
        dates_test,
        stats_preds_rolling,
        label="Statsmodels AR(5)",
        color="orange",
        linestyle="--",
    )
    plt.plot(dates_test, naive_preds, label="Naive", color="green", linestyle="--")

    plt.title(
        "Comparison: Manual AR(5) (1-step), Statsmodels AR(5) (1-step), Naive vs Actual"
    )
    plt.xlabel("Date")
    plt.ylabel("y1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
