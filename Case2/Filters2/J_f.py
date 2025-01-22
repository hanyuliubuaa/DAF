from integ_Lorenz_Phi import integ_Lorenz_Phi

def J_f(x, tspan):
    _, Phi = integ_Lorenz_Phi(x, tspan)
    return Phi