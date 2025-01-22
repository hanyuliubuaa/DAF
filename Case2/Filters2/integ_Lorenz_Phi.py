import numpy as np
from scipy.integrate import solve_ivp
from Lorenz_Phi import Lorenz_Phi

def integ_Lorenz_Phi(x0, tspan):
    nx = 3

    y0 = np.zeros(nx + nx**2)
    y0[0:nx] = x0
    Phi0 = np.eye(nx)
    y0[nx:] = Phi0.flatten()

    tol = 1e-13

    sol = solve_ivp(Lorenz_Phi, tspan, y0, method='RK45', rtol=tol, atol=tol)

    x = sol.y[0:nx, -1]
    Phi = sol.y[nx:, -1].reshape(nx, nx)
    
    return x, Phi