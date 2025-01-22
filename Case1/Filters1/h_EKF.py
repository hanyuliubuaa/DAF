import numpy as np

def h_EKF(x):
    H = np.eye(3)
    return x, H