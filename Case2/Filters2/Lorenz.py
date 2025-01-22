import numpy as np

def Lorenz(t, y):
    # Lorenz Model
    sigma = 10
    rho = 28
    beta = 8 / 3
    
    # Initialize y_dot
    y_dot = np.zeros(3)
    
    # Extract variables
    x1 = y[0]
    x2 = y[1]
    x3 = y[2]
    
    # Compute derivatives
    x1_dot = sigma * (x2 - x1)
    x2_dot = rho * x1 - x2 - x1 * x3
    x3_dot = x1 * x2 - beta * x3
    
    # Assign derivatives to y_dot
    y_dot[:] = [x1_dot, x2_dot, x3_dot]
    
    return y_dot