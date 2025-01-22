import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from DANSE_h import h
import time
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example usage:
B = 32  # Batch size
L = 210  # Sequence length
D_in = 3  # Input size
D_out = 6  # Output size
D_Hidden = 725  # Hidden state size
lr = 0.001 # Learning rate
num_epochs = 10 # Number of total epochs
L_warm = 10 # Total time before filtering
model_weights_path = 'model/DANSE1_725.pt'

class GRUModel(nn.Module):
    def __init__(self, D_in, D_out, D_Hidden):
        super(GRUModel, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.D_Hidden = D_Hidden
        self.gru = nn.GRU(input_size=D_in, hidden_size=D_Hidden, batch_first=True)
        self.linear = nn.Linear(D_Hidden, D_out)
    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.linear(gru_out)
        output[:, :, -3:] = torch.nn.functional.softplus(output[:, :, -3:])
        return output

model = GRUModel(D_in, D_out, D_Hidden).to(device)
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

# Sequential
with torch.no_grad():
    start_time = time.time()
    for i in range(1000):
        print(i)
        for j in range(L - L_warm):
            labels = label[i, j, :].unsqueeze(0)
            outputs = model(z[i, :(L_warm + j + 1), :].unsqueeze(0))[:, -1, :]
            x_pred = outputs[:, :3] # [1, 3]
            P_pred = torch.diag_embed(outputs[:, 3:]) # [1, 3, 3]
            z_pred, H = h(x_pred)
            P_zz = H @ P_pred @ H.transpose(1, 2) + torch.diag(torch.tensor([100, 100, 100], device=device)).unsqueeze(0) # [1, 3, 3]
            K = P_pred @ H.transpose(1, 2) @ torch.linalg.inv(P_zz)
            x_esti = x_pred + (K @ (labels.reshape(-1, D_in) - z_pred).unsqueeze(-1)).squeeze()

    end_time = time.time()
    print('Sequential time:', end_time - start_time)