import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Transformer import Model
import numpy as np
import matplotlib.pyplot as plt
from Loss_Function import LossF
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Lorenz.Dynamics import f
from Lorenz.Measurements import h
import time

# torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
B = 32 # Batch size
L = 210 # Length
D = 2 # Input dimension
L_warm = 10 # Total time before filtering
model_weights_path = 'model/PDAF2_128_32_8.pt'
class Config:
    def __init__(self):
        self.dec_in = 2 # Input dimension
        self.c_out = 3 # Output dimension
        self.d_model = 128 # Embedding dimension
        self.dropout = 0.0 # Dropout
        self.n_heads = 32 # Number of heads
        self.d_ff = 4 * self.d_model # MLP hidden layer dimension
        self.activation = 'gelu' # Activation function
        self.d_layers = 8 # Number of Transformer blocks
config = Config()


X_TRUE = torch.tensor(np.loadtxt("data2/Test_gt.txt", dtype='float32'), device=device).view(-1, 3, L-L_warm).transpose(2, 1)
Z_TRUE = torch.tensor(np.loadtxt("data2/Test_data.txt", dtype='float32'), device=device).view(-1, 2, L).transpose(2, 1)
label = torch.tensor(np.loadtxt("data2/Test_label.txt", dtype='float32'), device=device).view(-1, 2, L-L_warm).transpose(2, 1)

# Initialize
model = Model(config).to(device)
model.load_state_dict(torch.load(model_weights_path))
model.eval() # Close dropout
L = nn.MSELoss()

x_true = X_TRUE[0::2, :, :]
z = Z_TRUE[0::2, :, :]
with torch.no_grad():
    outputs = model(z)
residual_1 = outputs[:, L_warm:, :] - x_true
# residual_1 = h(f(outputs[:, L_warm:209, :].reshape(500*199, 3))).reshape(500, 199, 2) - h(x_true[:, 1:, :].reshape(500*199, 3)).reshape(500, 199, 2)

x_true = X_TRUE[1::2, :, :]
z = Z_TRUE[1::2, :, :]
with torch.no_grad():
    outputs = model(z)
residual_2 = outputs[:, L_warm:, :] - x_true
# residual_2 = h(f(outputs[:, L_warm:209, :].reshape(500*199, 3))).reshape(500, 199, 2) - h(x_true[:, 1:, :].reshape(500*199, 3)).reshape(500, 199, 2)

RMSE = torch.sqrt(torch.mean(torch.mean((residual_1*residual_1 + residual_2*residual_2)/2, dim=0), dim=1))
# RMSE = torch.sqrt(torch.mean((residual_1*residual_1 + residual_2*residual_2)/2, dim=0)[:, 1])

print(torch.sqrt(torch.mean(RMSE**2)))

np.savetxt('RMSE_PDAF_2.txt', RMSE.cpu().numpy())
# np.savetxt('z2.txt', RMSE.cpu().numpy())