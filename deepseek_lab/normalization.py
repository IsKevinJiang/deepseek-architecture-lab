import torch

import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.parameter.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        x_norm = x * torch.rsqrt(torch.mean(x ** 2, dim = -1, keepdim = True) + self.eps)
        return self.weight * x_norm