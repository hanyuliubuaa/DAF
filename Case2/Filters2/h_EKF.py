import numpy as np

def h_EKF(x):
    # 初始化 z
    z = np.zeros((2))
    z[0] = x[0] ** 2
    z[1] = x[1]
    
    # 创建 H 矩阵
    H = np.array([[2 * x[0], 0, 0],
                  [0, 1, 0]])
    
    return z, H