import torch.nn as nn
from deepseek_lab.normalization import RMSNorm
from deepseek_lab.transformer_block import TransformerBlock

class Model(nn.Module):
    def __init__(self, hidden_dim, num_heads, max_seq_len, intermediate_dim, vocab_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rmsnorm = RMSNorm(hidden_dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(hidden_dim,
                                                                   num_heads,
                                                                   max_seq_len,
                                                                   intermediate_dim)
                                                                   for _ in range(num_layers)
                                                                   ])
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)

        #Storing config parameters
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.intermediate_dim = intermediate_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

    def forward(self, x, offset=0):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x, offset)
        x = self.linear(self.rmsnorm(x))
        return x