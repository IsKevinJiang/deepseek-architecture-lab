import torch
import torch.nn as nn
from modelforge.position import RotaryEmbedding

class SDPAMHA(nn.Module):
    def __init__(self, hidden_dim = 512, num_heads = 8, max_seq_len=1024):
        super().__init__()

        if (hidden_dim % num_heads != 0):
            raise ValueError("Hidden dimension not divisble by number of attention heads")
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.RoPE = RotaryEmbedding(self.head_dim, max_seq_len)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len


    def forward(self, x, offset=0):
        # input is [B, S, D]
        batch = x.size(0)
        sequence = x.size(1)

        q = self.q_proj(x).reshape(batch, sequence, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, sequence, self.num_heads, self.head_dim)

        q, k = self.RoPE(torch.transpose(q, 1,2), torch.transpose(k,1,2), offset)
        v = torch.transpose(self.v_proj(x).reshape(batch, sequence, self.num_heads, self.head_dim), 1, 2)

        attention = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        h = attention.transpose(1,2).reshape(batch, sequence, self.hidden_dim)
        output = self.out_proj(h)
        return output
