import torch
from torch import optim
from torch import nn

import net


# data preprocessing
class DataSet():
    def __init__(self, filepath4train, filepath4val):
        with open(filepath4train, 'r') as f:
            self.train_data = f.readlines()
        with open(filepath4val, 'r') as f:
            self.val_data = f.readlines()
        self.train_index = 0
        self.val_index = 0
    def reset(self):
        self.train_index = 0
        self.val_index = 0
    def get_train_num(self):
        assert len(self.train_data) % 2 == 0
        return len(self.train_data) // 2
    def get_val_num(self):
        assert len(self.val_data) % 2 == 0
        return len(self.val_data) // 2
    def get_train_data(self, batch_size):
        x = []
        y = []
        for _ in range(batch_size):
            if self.train_index < len(self.train_data) - 1:
                x.append(float(self.train_data[self.train_index]))
                y.append(float(self.train_data[self.train_index + 1]))
                self.train_index += 2
        return torch.tensor(x), torch.tensor(y)
    def get_val_data(self, batch_size):
        x = []
        y = []
        for _ in range(batch_size):
            if self.val_index < len(self.val_data) - 1:
                x.append(float(self.val_data[self.val_index]))
                y.append(float(self.val_data[self.val_index + 1]))
                self.val_index += 2
        return torch.tensor(x), torch.tensor(y)


# some filepaths
train_ds_fp = '~/Python_test/first_pytorch/y=2x+3_train.txt'
val_ds_fp = '~/Python_test/first_pytorch/y=2x+3_val.txt'
state_dict_fp = '~/Python_test/first_pytorch/first_net.pth'

# instantiate dataset and network
ds = DataSet(train_ds_fp, val_ds_fp)
first_net = net.FirstNet()

# configuration
# 不建议将运行设备搞成可配置
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.SGD(first_net.parameters(), lr=0.01)
epochs_num = 1000
batch_size = 6

# training
for epoch in range(epochs_num):
    ds.reset()
    # one epoch
    iterations = ds.get_train_num() // batch_size \
        if ds.get_train_num() % batch_size == 0 \
        else ds.get_train_num() // batch_size + 1
    for iteration in range(iterations):
        x, y = ds.get_train_data(batch_size)
        y_hat = first_net(x)
        loss = criterion(y, y_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # validation
    if epoch % 10 == 9:
        loss = 0
        iterations = ds.get_val_num() // batch_size \
            if ds.get_val_num() % batch_size == 0 \
            else ds.get_val_num() // batch_size + 1
        for iteration in range(iterations):
            x, y = ds.get_val_data(batch_size)
            y_hat = first_net(x)
            loss += criterion(y, y_hat).item()
        print(f'epoch {epoch}: w = {first_net.w.item()}, b = {first_net.b.item()}, loss = {loss}')
        if loss < 1e-6:
            break

# store parameters
torch.save(first_net.state_dict(), state_dict_fp)
