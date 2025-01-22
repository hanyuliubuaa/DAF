import numpy as np

def Lorenz_Phi(t, y):
    # Lorenz Model parameters
    sigma = 10
    rho = 28
    beta = 8 / 3
    nx = 3

    # Initialize y_dot
    y_dot = np.zeros(nx + nx**2)
    
    # Extract variables
    x1, x2, x3 = y[0], y[1], y[2]
    
    # Compute derivatives
    x1_dot = sigma * (x2 - x1)
    x2_dot = rho * x1 - x2 - x1 * x3
    x3_dot = x1 * x2 - beta * x3
    y_dot[0:nx] = [x1_dot, x2_dot, x3_dot]
    
    # Compute the Jacobian matrix A
    A = np.array([
        [-sigma, sigma, 0],
        [rho - x3, -1, -x1],
        [x2, x1, -beta]
    ])
    
    # Reshape Phi
    Phi = y[nx:].reshape(nx, nx)
    
    # Compute Phi_dot
    Phi_dot = A @ Phi
    y_dot[nx:] = Phi_dot.flatten()
    
    return y_dot