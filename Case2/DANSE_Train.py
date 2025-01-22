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
D_in = 2  # Input size
D_out = 6  # Output size
D_Hidden = 725  # Hidden state size
lr = 0.000001 # Learning rate
num_epochs = 10 # Number of total epochs
L_warm = 10 # Total time before filtering
model_weights_path = 'model/DANSE3.pt'

class GRUModel(nn.Module):
    def __init__(self, D_in, D_out, D_Hidden):
        super(GRUModel, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.D_Hidden = D_Hidden
        self.gru = nn.GRU(input_size=D_in, hidden_size=D_Hidden, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(D_Hidden, D_out)
    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.linear(gru_out)
        output[:, :, -3:] = torch.nn.functional.softplus(output[:, :, -3:].clone())
        return output

model = GRUModel(D_in, D_out, D_Hidden).to(device)
model.load_state_dict(torch.load('model/DANSE2.pt'))

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

# Train
Loss_plot = np.zeros([num_epochs])
for epoch in range(num_epochs):
    model.train() # Open dropout
    for batch_index, (data, labels) in enumerate(Train_loader):
        data = data.view(-1, D_in, L).transpose(2, 1).to(device)
        B, _, _ = data.shape
        labels = labels.view(-1, D_in, L-L_warm).transpose(2, 1).to(device)
        outputs = model(data)[:, L_warm:, :] # [B, 200, 6]
        x_pred = outputs[:, :, :3].reshape(-1, 3) # [B*200, 3]
        P_pred = torch.diag_embed(outputs[:, :, 3:].reshape(-1, 3)) # [B*200, 3, 3]
        z_pred, H = h(x_pred)
        P_zz = H @ P_pred @ H.transpose(1, 2) + torch.diag(torch.tensor([10000, 100], device=device)).unsqueeze(0).repeat(B*200, 1, 1) # [B*200, 2, 2]
        labels = labels.reshape(-1, D_in) # [B*200, 2]
        residual = labels - z_pred # [B*200, 2]
        residual = residual.unsqueeze(2)  # [B*200, 2, 1]
        loss = (residual.transpose(1, 2) @ torch.linalg.inv(P_zz) @ residual).squeeze() + torch.log(torch.linalg.det(P_zz))
        loss = torch.mean(loss)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index % 10 == 0:
            print('[{}/{}],[{}/{}],loss={:.4f}'.format(epoch, num_epochs, batch_index, len(Train_loader), loss))

torch.save(model.state_dict(), model_weights_path)