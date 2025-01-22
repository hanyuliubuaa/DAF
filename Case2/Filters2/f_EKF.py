from integ_Lorenz_Phi import integ_Lorenz_Phi

def f_EKF(x0, tspan):
    x, Phi = integ_Lorenz_Phi(x0, tspan)
    return x, Phi