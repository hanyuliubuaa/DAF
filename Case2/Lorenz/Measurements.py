import torch

# class h_class(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         N, _ = x.shape
#         z = x
#         H = torch.eye(3, device=x.device).unsqueeze(0).repeat(N, 1, 1)
#         ctx.save_for_backward(H)
#         return z
    
#     @staticmethod
#     def backward(ctx, grad_out_in):
#         (H, ) = ctx.saved_tensors
#         return (H.transpose(2, 1) @ grad_out_in.unsqueeze(-1)).squeeze(-1)

# h = h_class.apply


class h_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        N, _ = x.shape
        z = torch.zeros([N, 2], device=x.device)
        z[:, 0] = x[:, 0] ** 2
        z[:, 1] = x[:, 1]
        H = torch.zeros([N, 2, 3], device=x.device)
        H[:, 0, 0] = 2 * x[:, 0]
        H[:, 1, 1] = H[:, 1, 1] + 1.0
        ctx.save_for_backward(H)
        return z
    
    @staticmethod
    def backward(ctx, grad_out_in):
        (H, ) = ctx.saved_tensors
        return (H.transpose(2, 1) @ grad_out_in.unsqueeze(-1)).squeeze(-1)
    
h = h_class.apply