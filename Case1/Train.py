import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Transformer import Model
import numpy as np
import matplotlib.pyplot as plt
from Loss_Function import LossF
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
B = 32 # Batch size
L = 210 # Length
D = 3 # Input dimension
lr = 0.000001 # Learning rate
num_epochs = 10 # Number of total epochs
L_warm = 10 # Total time before filtering
model_weights_path = 'model/PDAF.pt'
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
Train_data_list = np.loadtxt("data1/Train_data.txt", dtype='float64')
Train_label_list = np.loadtxt("data1/Train_label.txt", dtype='float64')
# Train_label_list = np.loadtxt("./data_nonlinear/Pretrain_UKF_new.txt", dtype='float64')
custom_dataset = CustomDataset(Train_data_list, Train_label_list)
Train_loader = DataLoader(custom_dataset, batch_size=B, shuffle=True)

# Test data and label
Test_data_list = np.loadtxt("data1/Test_data.txt", dtype='float64')
Test_label_list = np.loadtxt("data1/Test_label.txt", dtype='float64')
custom_dataset = CustomDataset(Test_data_list, Test_label_list)
Test_loader = DataLoader(custom_dataset, batch_size=len(Test_data_list), shuffle=False)


# Initialize
model = Model(config).to(device)
model.load_state_dict(torch.load('model/PDAF3.pt'))
total = sum([param.nelement() for param in model.parameters()])
print('Number of parameter: % .4f' % (total))

# Loss and Update
# LossF = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=False)


# Train
Loss_plot = np.zeros([num_epochs])
for epoch in range(num_epochs):
    model.train() # Open dropout
    for batch_index, (data, labels) in enumerate(Train_loader):
        data = data.view(-1, D, L).transpose(2, 1).to(device)
        labels = labels.view(-1, 3, L-L_warm).transpose(2, 1).to(device)
        outputs = model(data)[:, L_warm:, :]
        #compute loss
        loss = LossF(outputs, labels)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index % 10 == 0:
            print('[{}/{}],[{}/{}],loss={:.4f}'.format(epoch, num_epochs, batch_index, len(Train_loader), loss))

torch.save(model.state_dict(), model_weights_path)

# Test
model.eval() # Close dropout
with torch.no_grad():
    for data, labels in Test_loader:
        data = data.view(-1, D, L).transpose(2, 1).to(device)
        labels = labels.view(-1, 3, L-L_warm).transpose(2, 1).to(device)
        outputs = model(data)[:, L_warm:, :]
        MSE = LossF(outputs, labels)
        print("MSE: {}".format(MSE))

