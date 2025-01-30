import numpy as np


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
