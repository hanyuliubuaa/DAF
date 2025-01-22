import numpy as np

def h(x):
    if np.size(x.shape) == 1:
        z = np.zeros((2))
        z[0] = x[0] ** 2
        z[1] = x[1]
    else:
        N = x.shape[1]
        z = np.zeros((2, N))
        for i in range(N):
            z[0, i] = x[0, i] ** 2
            z[1, i] = x[1, i]
    return z