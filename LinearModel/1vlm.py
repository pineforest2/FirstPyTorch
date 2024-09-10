import torch
import torch.nn as nn
import torch.optim as optim

class FirstNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        y_hat = self.w * x + self.b
        return y_hat

# y=2x+3
x = torch.tensor([
    -0.8, -5.1, 91.7, 0.6, -0.2,
    35.6, 0.1, 5.0, -1.5, 0.0,
    91.7, 0.6, 1000.0, 7.3, 33.3
])
y = torch.tensor([
    1.4, -7.2, 186.4, 4.2, 2.6,
    74.2, 3.2, 13.0, 0.0, 3.0,
    186.4, 4.2, 2003.0, 17.6, 69.6
])
ds_len = x.shape[0]
num_val = ds_len // 6
num_train = ds_len - num_val

# instantiate network
first_net = FirstNet()

# configuration
# 不建议将运行设备搞成可配置
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(first_net.parameters(), lr=0.01)
num_epochs = 1000
batch_size = 2

# training
for epoch in range(num_epochs):
    for iteration in range(num_train):
        output = first_net(x[iteration])
        loss = criterion(output, y[iteration])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # validation
    if epoch % 10 == 9:
        error = 0
        for iteration in range(num_val):
            output = first_net(x[num_train + iteration])
            error += criterion(output, y[num_train + iteration]).item()
        print(f'epoch {epoch}: w={first_net.w.item()}, b={first_net.b.item()}, error={error}')
        if error < 1e-5:
            break

# storing parameters
torch.save(first_net.state_dict(), './first_net.pth')

# testing
first_net4test = FirstNet()
first_net4test.load_state_dict(
    torch.load(
        './first_net.pth'
    )
)

with torch.no_grad():
    x = torch.tensor([66.0, -66.0, 7.5])
    y_hat = first_net4test(x)
    print(y_hat)
