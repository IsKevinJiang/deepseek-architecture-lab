import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size) -> None:
        super().__init__()

        #W_gate (hidden -> intermediate)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        #W_up (hidden -> intermediate)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        #W_down (intermediate -> hidden)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        x = self.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(x)
    
    def silu(self, x):
        return x * torch.sigmoid(x)
