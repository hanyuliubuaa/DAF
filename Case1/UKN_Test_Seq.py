import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from DANSE_h import h
from UKN_f import f
import time
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example usage:
B = 32  # Batch size
L = 210  # Sequence length
D_in = 6  # Input size
D_out = 9  # Output size
D_Hidden = 725  # Hidden state size
lr = 0.001 # Learning rate
num_epochs = 10 # Number of total epochs
L_warm = 10 # Total time before filtering
model_weights_path = 'model/UKN1_725.pt'

class GRUModel(nn.Module):
    def __init__(self, D_in, D_out, D_Hidden):
        super(GRUModel, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.D_Hidden = D_Hidden
        self.gru_cell = nn.GRUCell(input_size=D_in, hidden_size=D_Hidden)
        self.linear = nn.Linear(D_Hidden, D_out)
    def forward(self, x, h_0):
        h = self.gru_cell(x, h_0)
        output_t = self.linear(h)
        return output_t, h

model = GRUModel(D_in, D_out, D_Hidden).to(device=device)
total = sum([param.nelement() for param in model.parameters()])
print('Number of parameter: % .4f' % (total))


X_TRUE = torch.tensor(np.loadtxt("data1/Test_gt.txt", dtype='float64'), device=device).view(-1, 3, L-L_warm).transpose(2, 1)
Z_TRUE = torch.tensor(np.loadtxt("data1/Test_data.txt", dtype='float64'), device=device).view(-1, 3, L).transpose(2, 1)
label = torch.tensor(np.loadtxt("data1/Test_label.txt", dtype='float64'), device=device).view(-1, 3, L-L_warm).transpose(2, 1)

# Initialize
model.load_state_dict(torch.load(model_weights_path))
model.eval()

x_true = X_TRUE
z = Z_TRUE
data = z
B = 1
hh = torch.zeros([B, D_Hidden], device=device)
dx = torch.zeros([B, 3], device=device)
x_esti = torch.zeros([B, 3], device=device)
with torch.no_grad():
    start_time = time.time()
    for i in range(1000):
        print(i)
        for j in range(L):
            if j > 0:
                x_pred = f(x_esti)
                y_pred, _ = h(x_pred)
                dy = data[i, j, :].unsqueeze(0) - y_pred
                K, h_next = model(torch.concatenate((dx, dy), dim=1), hh)
                K = K / 100
                hh = h_next
                x_esti = x_pred + (K.reshape([B, 3, 3]) @ (data[i, j, :] - y_pred).unsqueeze(-1)).squeeze()
                dx = x_esti - x_pred

    end_time = time.time()
    print('Sequential time:', end_time - start_time)

    
