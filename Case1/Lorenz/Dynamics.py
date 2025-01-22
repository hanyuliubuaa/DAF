import torch
import torch.autograd
import torch.nn as nn
from torchdiffeq import odeint

torch.set_default_dtype(torch.float64)

class ODEfunc(nn.Module):
    def __init__(self):
        super(ODEfunc, self).__init__()
    
    def forward(self, t, y):
        # Lorenz Model parameters
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0
        N, _ = y.shape
        nx = 3
        
        # Initialize the output tensor
        y_dot = torch.zeros([N, nx + nx**2], device=y.device)
        
        # Extract state variables
        x1 = y[:, 0]
        x2 = y[:, 1]
        x3 = y[:, 2]
        
        # Compute state derivatives
        x1_dot = sigma * (x2 - x1)
        x2_dot = rho * x1 - x2 - x1 * x3
        x3_dot = x1 * x2 - beta * x3
        y_dot[:, 0:nx] = torch.stack([x1_dot, x2_dot, x3_dot], dim=1)
        
        # State transition matrix A
        A = torch.zeros([N, nx, nx], device=y.device)
        A[:, 0, 0] = A[:, 0, 0] - sigma
        A[:, 0, 1] = A[:, 0, 1] + sigma
        A[:, 1, 0] = rho - x3
        A[:, 1, 1] = A[:, 1, 1] - 1.0
        A[:, 1, 2] = A[:, 1, 2] - x1
        A[:, 2, 0] = A[:, 2, 0] + x2
        A[:, 2, 1] = A[:, 2, 1] + x1
        A[:, 2, 2] = A[:, 2, 2] - beta
        
        # Extract Phi
        Phi = y[:, nx:].reshape(N, nx, nx)
        
        # Compute Phi_dot
        Phi_dot = A @ Phi
        y_dot[:, nx:] = Phi_dot.reshape(N, -1)
            
        return y_dot



class f_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_k_1):
        # x_k_1: The states at time k-1, [B * L, D]
        
        tspan = torch.tensor([0.0, 0.03], device=x_k_1.device)
        Func = ODEfunc()
        N, nx = x_k_1.shape
        X = torch.cat((x_k_1, torch.eye(nx, device=x_k_1.device).unsqueeze(0).repeat(N, 1, 1).reshape(N, -1)), dim=1)
        y = odeint(Func, X, tspan, rtol=1e-13, atol=1e-13)[-1, :, :] # [B * L, D]
        x_k = y[:, 0 : nx] + 0.01 * torch.randn_like(y[:, 0 : nx])
        # x_k = y[:, 0 : nx] + 2 * torch.randn_like(y[:, 0 : nx])
        # x_k = y[:, 0 : nx]
        Phi = y[:, nx:].reshape(N, nx, nx)
        ctx.save_for_backward(Phi)
        return x_k
    
    @staticmethod
    def backward(ctx, grad_out_in):
        (Phi, ) = ctx.saved_tensors
        return (Phi.transpose(2, 1) @ grad_out_in.unsqueeze(-1)).squeeze(-1)
    
f = f_class.apply