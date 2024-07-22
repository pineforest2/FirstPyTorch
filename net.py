import torch
from torch import nn

class FirstNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        y_hat = self.w * x + self.b
        return y_hat


