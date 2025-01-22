import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from DANSE_h import h
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
L = nn.MSELoss()

x_true = X_TRUE[0::2, :, :]
z = Z_TRUE[0::2, :, :]
labels = label[0::2, :, :]
with torch.no_grad():
    outputs = model(z)[:, L_warm:, :]
    x_pred = outputs[:, :, :3].reshape(-1, 3) # [B*200, 3]
    P_pred = torch.diag_embed(outputs[:, :, 3:].reshape(-1, 3)) # [B*200, 3, 3]
    z_pred, H = h(x_pred)
    P_zz = H @ P_pred @ H.transpose(1, 2) + torch.diag(torch.tensor([100, 100, 100], device=device)).unsqueeze(0).repeat(500*200, 1, 1) # [B*200, 3, 3]
    K = P_pred @ H.transpose(1, 2) @ torch.linalg.inv(P_zz)
    x_esti = x_pred + (K @ (labels.reshape(-1, 3) - z_pred).unsqueeze(-1)).squeeze()
    x_esti = x_esti.reshape([500, 200, 3])
residual_1 = x_esti - x_true

x_true = X_TRUE[1::2, :, :]
z = Z_TRUE[1::2, :, :]
labels = label[1::2, :, :]
with torch.no_grad():
    outputs = model(z)[:, L_warm:, :]
    x_pred = outputs[:, :, :3].reshape(-1, 3) # [B*200, 3]
    P_pred = torch.diag_embed(outputs[:, :, 3:].reshape(-1, 3)) # [B*200, 3, 3]
    z_pred, H = h(x_pred)
    P_zz = H @ P_pred @ H.transpose(1, 2) + torch.diag(torch.tensor([100, 100, 100], device=device)).unsqueeze(0).repeat(500*200, 1, 1) # [B*200, 3, 3]
    K = P_pred @ H.transpose(1, 2) @ torch.linalg.inv(P_zz)
    x_esti = x_pred + (K @ (labels.reshape(-1, 3) - z_pred).unsqueeze(-1)).squeeze()
    x_esti = x_esti.reshape([500, 200, 3])
residual_2 = x_esti - x_true

RMSE = torch.sqrt(torch.mean(torch.mean((residual_1*residual_1 + residual_2*residual_2)/2, dim=0), dim=1))

print(torch.sqrt(torch.mean(RMSE**2)))

np.savetxt('RMSE_DANSE_1.txt', RMSE.cpu().numpy())