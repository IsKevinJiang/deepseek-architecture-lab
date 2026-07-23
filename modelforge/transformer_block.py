import torch.nn as nn
from modelforge.feed_forward import SwiGLU
from modelforge.normalization import RMSNorm
from modelforge.multi_head_attention import MHA

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, max_seq_len, intermediate_dim):
        super().__init__()
        self.rmsnorm1 = RMSNorm(hidden_dim)
        self.rmsnorm2 = RMSNorm(hidden_dim)
        self.attention = MHA(hidden_dim, num_heads, max_seq_len)
        self.swiglu = SwiGLU(hidden_dim, intermediate_dim)
    
    def forward(self, x, offset=0):
        x1 = x + self.attention(self.rmsnorm1(x), offset)
        x2 = x1 + self.swiglu(self.rmsnorm2(x1))
        return x2
