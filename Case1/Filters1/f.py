import numpy as np
from integ_Lorenz import integ_Lorenz

def f(x0, tspan):
    if np.size(x0.shape) == 1:
        x = integ_Lorenz(x0, tspan)
    else:
        n, N = x0.shape
        x = np.zeros((n, N))
        for i in range(N):
            x[:, i] = integ_Lorenz(x0[:, i], tspan)
    return x