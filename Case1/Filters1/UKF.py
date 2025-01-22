import numpy as np
from scipy.linalg import cholesky
from f import f
from h import h

def UKF(y_esti, P_esti, z, Q, R, k, T):
    n = len(y_esti)
    tspan = [0, T]
    
    alpha = 1e-4
    beta = 2
    kappa = 0
    lambda_ = alpha**2 * (n + kappa) - n
    
    # Sigma points calculation
    y_sigma = np.zeros((n, 2*n + 1))
    y_sigma[:, 0] = y_esti
    sqrt_P = cholesky(P_esti, lower=True)
    for i in range(1, n + 1):
        y_sigma[:, i] = y_esti + np.sqrt(n + lambda_) * sqrt_P[:, i - 1]
    for i in range(n + 1, 2*n + 1):
        y_sigma[:, i] = y_esti - np.sqrt(n + lambda_) * sqrt_P[:, i - n - 1]
    
    y_pred_sigma = f(y_sigma, tspan)
    
    # Weights for mean and covariance
    w0 = lambda_ / (n + lambda_)
    w1 = 1 / (2 * (n + lambda_))
    
    y_pred = w0 * y_pred_sigma[:, 0]
    for i in range(1, 2*n + 1):
        y_pred += w1 * y_pred_sigma[:, i]
    
    P_pred = (w0 + 1 - alpha**2 + beta) * np.outer(y_pred_sigma[:, 0] - y_pred, y_pred_sigma[:, 0] - y_pred)
    for i in range(1, 2*n + 1):
        P_pred += w1 * np.outer(y_pred_sigma[:, i] - y_pred, y_pred_sigma[:, i] - y_pred)
    P_pred += Q
    
    # Resampling
    y_pred_sigma_new = np.zeros((n, 2*n + 1))
    y_pred_sigma_new[:, 0] = y_pred
    sqrt_P_new = cholesky(P_pred, lower=True)
    for i in range(1, n + 1):
        y_pred_sigma_new[:, i] = y_pred + np.sqrt(n + lambda_) * sqrt_P_new[:, i - 1]
    for i in range(n + 1, 2*n + 1):
        y_pred_sigma_new[:, i] = y_pred - np.sqrt(n + lambda_) * sqrt_P_new[:, i - n - 1]
    
    # Nonlinear transformation
    z_pred_sigma = h(y_pred_sigma_new)
    
    # Compute statistics
    z_pred = w0 * z_pred_sigma[:, 0]
    for i in range(1, 2*n + 1):
        z_pred += w1 * z_pred_sigma[:, i]
    
    P_xz = w0 * np.outer(y_pred_sigma_new[:, 0] - y_pred, z_pred_sigma[:, 0] - z_pred)
    for i in range(1, 2*n + 1):
        P_xz += w1 * np.outer(y_pred_sigma_new[:, i] - y_pred, z_pred_sigma[:, i] - z_pred)
    
    P_zz = w0 * np.outer(z_pred_sigma[:, 0] - z_pred, z_pred_sigma[:, 0] - z_pred)
    for i in range(1, 2*n + 1):
        P_zz += w1 * np.outer(z_pred_sigma[:, i] - z_pred, z_pred_sigma[:, i] - z_pred)
    P_zz += (1 - alpha**2 + beta) * np.outer(z_pred_sigma[:, 0] - z_pred, z_pred_sigma[:, 0] - z_pred)
    P_zz += R
    
    K = np.linalg.solve(P_zz, P_xz.T).T
    y_esti = y_pred + K @ (z[:, k] - z_pred)
    P_esti = P_pred - K @ P_zz @ K.T
    
    return y_esti, P_esti