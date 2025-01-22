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

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
B = 32 # Batch size
L = 210 # Length
D = 3 # Input dimension
L_warm = 10 # Total time before filtering
model_weights_path = 'model/PDAF1_128_8_8.pt'
class Config:
    def __init__(self):
        self.dec_in = 3 # Input dimension
        self.c_out = 3 # Output dimension
        self.d_model = 128 # Embedding dimension
        self.dropout = 0.0 # Dropout
        self.n_heads = 8 # Number of heads
        self.d_ff = 4 * self.d_model # MLP hidden layer dimension
        self.activation = 'gelu' # Activation function
        self.d_layers = 8 # Number of Transformer blocks
config = Config()


X_TRUE = torch.tensor(np.loadtxt("data1/Test_gt.txt", dtype='float64'), device=device).view(-1, 3, L-L_warm).transpose(2, 1)
Z_TRUE = torch.tensor(np.loadtxt("data1/Test_data.txt", dtype='float64'), device=device).view(-1, 3, L).transpose(2, 1)
label = torch.tensor(np.loadtxt("data1/Test_label.txt", dtype='float64'), device=device).view(-1, 3, L-L_warm).transpose(2, 1)

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

x_true = X_TRUE[1::2, :, :]
z = Z_TRUE[1::2, :, :]
with torch.no_grad():
    outputs = model(z)
residual_2 = outputs[:, L_warm:, :] - x_true

RMSE = torch.sqrt(torch.mean(torch.mean((residual_1*residual_1 + residual_2*residual_2)/2, dim=0), dim=1))

print(torch.sqrt(torch.mean(RMSE**2)))

np.savetxt('RMSE_PDAF_1.txt', RMSE.cpu().numpy())