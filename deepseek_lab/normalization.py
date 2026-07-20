import torch

import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.parameter.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        input_type = x.dtype
        x_float32 = x.to(torch.float32)
        x_norm = x_float32 * torch.rsqrt(torch.mean(x_float32 ** 2, dim = -1, keepdim = True) + self.eps)
        return (self.weight * x_norm).to(input_type)