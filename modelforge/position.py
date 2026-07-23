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
    def __init__(self, dim, max_seq_len, base = 10_000):
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

        #Both are [max_seq_len, D]
        cos_rot = torch.cos(theta)
        sin_rot = torch.sin(theta)

        self.register_buffer("cos_rot", cos_rot, persistent=False)
        self.register_buffer("sin_rot", sin_rot, persistent=False)
        self.register_buffer("base_freq", base_freq, persistent=False)

    def forward(self, q, k, offset = 0):
        seq_len = q.size(2)
        q_dtype = q.dtype
        k_dtype = k.dtype

        if (q.size(-1) != self.dim or k.size(-1) != self.dim):
            raise ValueError("Query or Key tensor does not have the right feature dimension")
        if (offset < 0 or offset + seq_len > self.max_seq_len):
            raise ValueError("Offset is invalid")
        if (q.size(2) != k.size(2)):
            raise ValueError("Sequence length between Q and K do not match")

        #Q & K are [B,H,S,D]
        cos_rot = self.cos_rot[offset:seq_len + offset, :]
        sin_rot = self.sin_rot[offset:seq_len + offset, :]
        q_pos = q * cos_rot.to(q_dtype) + rotate_half(q) * sin_rot.to(q_dtype)
        k_pos = k * cos_rot.to(k_dtype) + rotate_half(k) * sin_rot.to(k_dtype)

        return q_pos, k_pos


        
        