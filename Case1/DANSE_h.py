import torch
def h(x):
    N, _ = x.shape
    z = x
    H = torch.eye(3, device=x.device).unsqueeze(0).repeat(N, 1, 1)
    return z, H