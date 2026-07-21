# Parameter Count

## Model Configuration

| Symbol | Configuration | Value |
|---|---|---:|
| $V$ | Vocabulary size | 50,257 |
| $D$ | Hidden dimension | 512 |
| $H$ | Attention heads | 8 |
| $L$ | Transformer layers | 16 |
| $I$ | SwiGLU intermediate dimension | 1,408 |
| $S$ | Maximum sequence length | 1,024 |

## Parameters by Component

### Embedding and Output Layers

- Token embedding: $VD$
- Untied output head: $VD$

### Each Transformer Block

- Query, key, value, and output projections: $4D^2$
- SwiGLU gate, up, and down projections: $3DI$
- Two RMSNorm layers: $2D$

### Final Normalization

- Final RMSNorm: $D$

## Total

$$
\begin{aligned}
N &= VD + VD + L\left(4D^2 + 3DI + 2D\right) + D \\ 
&= 102{,}860{,}288
\end{aligned}
$$

The model therefore contains approximately **102.9 million trainable parameters**.

RoPE has no learned parameters. The number of attention heads does not change the
parameter count because the total projection dimension remains $D$. The maximum
sequence length only changes the size of the non-trainable RoPE buffers.
