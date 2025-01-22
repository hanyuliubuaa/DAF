import torch
import torch.nn as nn
from Lorenz.Dynamics import f
from Lorenz.Measurements import h

def LossF(x_k_1, z_k):
    # x_k_1: The states at time k-1, [B, L, Dx]
    # z_k: The measurements at time k, [B, L, Dz]
    B, L, D = x_k_1.shape
    _, _, Dz = z_k.shape
    x_k_1 = x_k_1.reshape(B * L, D)
    z_k = z_k.reshape(B * L, Dz)
    x_k = f(x_k_1)
    N = 1
    Total_loss = 0
    for _ in range(N):
        x_k = x_k + 0.01 * torch.randn_like(x_k) # Add process noise
        # x_k = x_k + 2 * torch.randn_like(x_k) # Add process noise
        z_predict = h(x_k)
        residual = z_predict - z_k # [B*L, Dz]
        residual = residual.unsqueeze(2)  # [B*L, Dz, 1]
        # R_inv = (1/100 * torch.eye(Dz, device=x_k_1.device)).unsqueeze(0).repeat(B*L, 1, 1) # [B*L, Dz, Dz]
        R_inv = torch.diag(torch.tensor([1/10000, 1/100], device=x_k_1.device)).unsqueeze(0).repeat(B*L, 1, 1) # [B*L, Dz, Dz]
        # R_inv = torch.diag(torch.tensor([1/16, 1/4], device=x_k_1.device)).unsqueeze(0).repeat(B*L, 1, 1) # [B*L, Dz, Dz] Q = 2
        loss = (residual.transpose(1, 2) @ R_inv @ residual).squeeze()
        loss = torch.mean(loss)
        Total_loss = Total_loss + loss / N
    
    return Total_loss
