from scipy.integrate import solve_ivp
from Lorenz import Lorenz

def integ_Lorenz(x0, tspan):
    # Define the tolerance options
    tol = 1e-13

    # Solve the differential equation using solve_ivp
    sol = solve_ivp(Lorenz, tspan, x0, method='RK45', rtol=tol, atol=tol)

    # Extract the final value
    x = sol.y[:, -1]
    
    return x