import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from DANSE_h import h
from UKN_f import f
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example usage:
B = 32  # Batch size
L = 210  # Sequence length
D_in = 5  # Input size
D_out = 6  # Output size
D_Hidden = 725  # Hidden state size
lr = 0.0001 # Learning rate
num_epochs = 1 # Number of total epochs
L_warm = 10 # Total time before filtering
model_weights_path = 'model/UKN.pt'

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
# model.load_state_dict(torch.load('model/UKN3.pt'))
total = sum([param.nelement() for param in model.parameters()])
print('Number of parameter: % .4f' % (total))


# Load dataset
def str2float(list):
    strlist = []
    for i in list:
        strlist.append(float(i))
    return strlist

class CustomDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data_list = data_list
        self.label_list = label_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        text = self.data_list[idx]
        label = self.label_list[idx]
        return text, label


# Train data and label
# When creating a dataset, each row represents a set of data and each column is listed as a dimension of the data
Train_data_list = np.loadtxt("data2/Train_data.txt", dtype='float64')
Train_label_list = np.loadtxt("data2/Train_label.txt", dtype='float64')
custom_dataset = CustomDataset(Train_data_list, Train_label_list)
Train_loader = DataLoader(custom_dataset, batch_size=B, shuffle=True)

# Test data and label
Test_data_list = np.loadtxt("data2/Test_data.txt", dtype='float64')
Test_label_list = np.loadtxt("data2/Test_label.txt", dtype='float64')
custom_dataset = CustomDataset(Test_data_list, Test_label_list)
Test_loader = DataLoader(custom_dataset, batch_size=len(Test_data_list), shuffle=False)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=False)
LossF = nn.MSELoss()

# Train
Loss_plot = np.zeros([num_epochs])
for epoch in range(num_epochs):
    model.train() # Open dropout
    for batch_index, (data, labels) in enumerate(Train_loader):
        data = data.view(-1, 2, L).transpose(2, 1).to(device)
        B, _, _ = data.shape
        labels = labels.view(-1, 2, L-L_warm).transpose(2, 1).to(device)
        hh = torch.zeros([B, D_Hidden], device=device)
        dx = torch.zeros([B, 3], device=device)
        x_esti = torch.zeros([B, 3], device=device)
        loss = 0
        for i in range(L):
            if i > 0:
                x_pred = f(x_esti)
                y_pred, _ = h(x_pred)
                dy = data[:, i, :] - y_pred
                K, h_next = model(torch.concatenate((dx, dy), dim=1), hh)
                K = K / 100
                hh = h_next
                x_esti = x_pred + (K.reshape([B, 3, 2]) @ (data[:, i, :] - y_pred).unsqueeze(-1)).squeeze()
                dx = x_esti - x_pred
                loss = loss + LossF(data[:, i, :], y_pred)
        l2_norm = sum(torch.norm(param, p=2) for param in model.parameters())
        loss = loss / (L-1) + 0.01 * l2_norm
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index % 1 == 0:
            print('[{}/{}],[{}/{}],loss={:.4f}'.format(epoch, num_epochs, batch_index, len(Train_loader), loss))

torch.save(model.state_dict(), model_weights_path)