import torch

import net


first_net = net.FirstNet()
first_net.load_state_dict(
    torch.load(
        '/home/b514/Python_test/first_pytorch/first_net.pth'
    )
)

with torch.no_grad():
    x = torch.tensor([66.0, -66.0])
    y_hat = first_net(x)
    print(y_hat)