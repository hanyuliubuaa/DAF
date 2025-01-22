import torch
def h(x):
    N, _ = x.shape
    z = torch.zeros([N, 2], device=x.device)
    z[:, 0] = x[:, 0] ** 2
    z[:, 1] = x[:, 1]
    H = torch.zeros([N, 2, 3], device=x.device)
    H[:, 0, 0] = 2 * x[:, 0]
    H[:, 1, 1] = H[:, 1, 1] + 1.0
    return z, H