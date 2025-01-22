import numpy as np
from J_f import J_f
from J_h import J_h

def CRLB(J, x_k_1, x_k, Q, R, T):
    tspan = [0, T]
    Phi = J_f(x_k_1, tspan)
    H = J_h(x_k)
    
    D_11 = Phi.T @ np.linalg.inv(Q) @ Phi
    D_22 = np.linalg.inv(Q) + H.T @ np.linalg.inv(R) @ H
    D_12 = Phi.T @ np.linalg.inv(Q)
    D_21 = D_12.T
    
    J = D_22 - D_21 @ np.linalg.inv(J + D_11) @ D_12
    J_inv = np.linalg.inv(J)
    
    return J, J_inv