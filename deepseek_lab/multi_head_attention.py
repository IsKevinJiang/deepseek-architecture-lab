import torch
import torch.nn as nn
from deepseek_lab.position import RotaryEmbedding

class MHA(nn.Module):
    def __init__(self, hidden_dim = 512, num_heads = 8):
        super().__init__()
        if (hidden_dim % num_heads != 0):
            raise ValueError("Hidden dimension not divisble by number of attention heads")
        head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.rope = RotaryEmbedding(head_dim)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads


    def forward(self, x):

