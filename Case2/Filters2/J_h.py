import numpy as np

def J_h(x):
    H = np.array([[2 * x[0], 0, 0], 
                  [0, 1, 0]])
    return H