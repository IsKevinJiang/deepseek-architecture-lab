import torch.nn as nn
from deepseek_lab.feed_forward import SwiGLU
from deepseek_lab.normalization import RMSNorm
from deepseek_lab.multi_head_attention import MHA
from deepseek_lab.transformer_block import TransformerBlock