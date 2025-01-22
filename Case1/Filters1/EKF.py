import numpy as np
from f_EKF import f_EKF
from h_EKF import h_EKF

def EKF(x_esti, P_esti, z, Q, R, k, T):
    n = len(x_esti)
    tspan = [0, T]
    
    x_pred, Phi = f_EKF(x_esti, tspan)
    P_pred = Phi @ P_esti @ Phi.T + Q
    
    z_pred, H = h_EKF(x_pred)
    
    H_T = H.T
    K = P_pred @ H_T @ np.linalg.inv(R + H @ P_pred @ H_T)
    x_esti = x_pred + K @ (z[:, k] - z_pred)
    I_KH = np.eye(n) - K @ H
    P_esti = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
    
    return x_esti, P_esti