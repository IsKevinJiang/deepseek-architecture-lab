import torch
import torch.nn as nn
def rotate_half(x):
    # Helper to create a term needed in the RoPE equation with the sine term

    d = x.shape[-1]
    if (d % 2 == 1):
        raise ValueError("RoPE requires dimensions to be even")

    a, b = x[..., :d//2], -x[..., d//2:]
    return torch.cat([b, a], dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len = 1024, base = 10_000):
        super().__init__()
        if (dim % 2 == 1):
            raise ValueError("RoPE requires dimensions to be even")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        #Caching cos and sin values for use later when we apply rotations
        dim_pairs = torch.arange(dim//2, dtype=torch.float32)
        base_freq = base ** -(2 * dim_pairs / dim)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.outer(positions, base_freq).repeat(1,2)

        cos_rot = torch.cos(theta)
        sin_rot = torch.sin(theta)

        self.register_buffer("cos_rot", cos_rot, persistent=False)
        self.register_buffer("sin_rot", sin_rot, persistent=False)
        self.register_buffer("base_freq", base_freq, persistent=False)


        
        