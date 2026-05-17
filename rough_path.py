import numpy as np
from scipy.linalg import cholesky
from scipy.linalg import solve_triangular
from sklearn.linear_model import LinearRegression

def fbm_covariance(T, H, n):
    t = np.linspace(0, T, n)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = 0.5 * (abs(t[i])**(2*H) + abs(t[j])**(2*H) - abs(t[i]-t[j])**(2*H))
    # Add small diagonal regularization to ensure positive definiteness
    cov += np.eye(n) * 1e-8
    return cov

def simulate_fbm(T, H, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    cov = fbm_covariance(T, H, n)
    # Attempt Cholesky; if fails due to rounding, try with larger regularization
    try:
        L = cholesky(cov, lower=True)
    except np.linalg.LinAlgError:
        # Add larger regularization
        cov += np.eye(n) * 1e-6
        L = cholesky(cov, lower=True)
    Z = np.random.randn(n)
    return L @ Z

def rough_path_prediction(returns_series, window, H=0.3, steps_per_day=10, order=2):
    if len(returns_series) < window + 1:
        return 0.0
    # Use last `window` days of returns
    returns = returns_series.iloc[-window:].values
    # Remove NaN
    valid = ~np.isnan(returns)
    returns = returns[valid]
    if len(returns) < 5:
        return 0.0
    # Simulate fBM increments for the same length
    T = len(returns)
    n = len(returns) + 1
    fbm = simulate_fbm(T, H, n, seed=42)
    increments = np.diff(fbm)   # length T
    # Align lengths
    inc = increments[:len(returns)]
    # Linear regression
    model = LinearRegression()
    model.fit(inc.reshape(-1, 1), returns)
    # Predict next step using next fBM increment
    next_fbm = simulate_fbm(T+1, H, n+1, seed=42)
    next_inc = next_fbm[-1] - next_fbm[-2]
    pred = model.predict([[next_inc]])[0]
    return pred
