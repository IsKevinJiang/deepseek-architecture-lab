
# deepseek-architecture-lab

> A from-scratch PyTorch implementation and controlled study of modern decoder-only language-model architecture, training infrastructure, and distributed pretraining.

## Overview

This project builds a modern language model from first principles in raw PyTorch, beginning with a trusted Llama-style dense baseline and then extending it with:

- **Full Multi-head Latent Attention (MLA)**, including query compression, key/value compression, decoupled RoPE, and inference-time weight absorption.
- **Fine-grained Mixture of Experts (MoE)** with shared experts, top-2 routing, dropless dispatch, and DeepSeek-V3-style auxiliary-loss-free load balancing.
- **Industrial pretraining infrastructure**, including memory-mapped data shards, mixed precision, exact checkpoint-and-resume, experiment tracking, profiling, SLURM, and Distributed Data Parallel training.
- **Controlled architecture comparisons** using matched active parameters, matched training tokens, per-architecture learning-rate sweeps, multiple random seeds, validation loss, and HellaSwag.

The goal is not to reproduce a frontier-scale model. The goal is to understand and measure the systems and architectural ideas that make modern pretraining work.

## Architecture

### 1. Dense Baseline

The first model is a decoder-only, Llama-style transformer implemented without the Hugging Face `transformers` library.

Planned components:

- Causal multi-head self-attention
- Pre-norm **RMSNorm**
- **SwiGLU** feed-forward network
- **Rotary positional embeddings**
- Bias-free linear layers
- Untied token-embedding and output-projection weights
- Explicit, validated weight initialization
- Approximately 100M parameters

The dense baseline serves as the correctness reference for every later architectural change.

### 2. Full Multi-head Latent Attention

The MLA implementation follows the complete formulation rather than the simplified key/value-only approximation commonly used in tutorials.

Planned components:

- Query down-projection into a query latent
- Query up-projection into per-head non-rotary queries
- Key/value down-projection into a shared KV latent
- Separate key and value up-projections
- RMSNorm applied to both compressed latent representations
- Decoupled RoPE with per-head rotary queries
- One shared rotary key across attention heads
- Concatenated rotary and non-rotary query/key components
- KV cache containing only the compressed KV latent and shared rotary key
- Inference-time weight absorption
- Numerical equivalence testing between absorbed and non-absorbed inference paths

### 3. Fine-Grained Mixture of Experts

The MoE stage replaces selected dense feed-forward layers with routed experts.

Planned components:

- Multiple small routed experts
- One or more shared experts that always execute
- Top-2 token routing
- Dropless token dispatch
- Approximately 200M total parameters and 60M active parameters
- Learned routing-bias updates for auxiliary-loss-free balancing
- A small sequence-wise auxiliary loss as a stability safeguard
- Per-step expert utilization, routing entropy, token-count, and bias-value logging

Correctness takes priority over kernel efficiency. The initial implementation will use a simple expert loop before introducing grouped or batched dispatch.

## Training Infrastructure

The training engine is intended to model real pretraining workflows rather than a notebook-only demonstration.

| Capability | Planned behavior |
|---|---|
| Precision | `bfloat16` autocast on Ampere-or-newer GPUs; `float16` with gradient scaling where required |
| Optimizer | AdamW with explicit decay and no-decay parameter groups |
| Schedule | Cosine decay with warmup; warmup-stable-decay may be evaluated |
| Stability | Gradient clipping and finite-loss checks |
| Effective batch size | Gradient accumulation |
| Attention kernel | `torch.nn.functional.scaled_dot_product_attention` |
| Compilation | `torch.compile` where stable and beneficial |
| Logging | Training loss, validation loss, learning rate, gradient norm, throughput, MFU, and architecture-specific metrics |
| Checkpointing | Model, optimizer, scheduler, dataloader position, scaler state, and RNG states |
| Resume behavior | Automatically resume from the latest valid checkpoint |
| Cluster execution | SLURM jobs, job arrays, and preemption-aware restart behavior |
| Distributed execution | PyTorch DDP launched with `torchrun` |

## Data Pipeline

Pipeline design:

1. Tokenize the corpus offline with the GPT-2 BPE tokenizer through `tiktoken`.
2. Write tokens into compact binary shards.
3. Reserve a held-out validation split before architecture experiments begin.
4. Load shards through memory mapping rather than loading the corpus into RAM.
5. Shuffle deterministically while preserving resumable dataloader state.
6. Use the same tokenizer, training data, validation data, and token budget for every architecture.

## Roadmap

### Phase 1 — Dense baseline

- [x] Implement causal attention, RMSNorm, SwiGLU, RoPE, and residual blocks
- [ ] Validate initialization and parameter counts
- [ ] Pass single-batch overfitting
- [ ] Pass tiny-corpus sampling test

### Phase 2 — Data and training engine

- [ ] Build offline tokenization and binary sharding
- [ ] Create a held-out validation split
- [ ] Implement memory-mapped, resumable loading
- [ ] Add mixed precision, AdamW, scheduling, clipping, and accumulation
- [ ] Add SDPA, compilation, experiment logging, and profiling
- [ ] Implement complete checkpoint-and-resume
- [ ] Pass the interrupted-run resume test

### Phase 3 — Full MLA

- [ ] Implement query and KV latent compression
- [ ] Add latent normalization
- [ ] Add decoupled RoPE
- [ ] Pass staged numerical tests
- [ ] Implement and validate weight absorption
- [ ] Measure KV-cache savings

### Phase 4 — Fine-grained MoE

- [ ] Implement routed and shared experts
- [ ] Add top-2 dropless routing
- [ ] Add auxiliary-loss-free balancing
- [ ] Add sequence-wise auxiliary balancing loss
- [ ] Instrument expert utilization and routing behavior
- [ ] Evaluate expert-collapse behavior

### Phase 5 — Distributed training and evaluation

- [ ] Add SLURM job templates and automatic resume
- [ ] Add DDP and per-rank data sharding
- [ ] Measure 1/2/4/8-GPU scaling
- [ ] Run learning-rate sweeps for each architecture
- [ ] Add HellaSwag evaluation
- [ ] Run matched-token, matched-active-parameter comparisons
- [ ] Run multiple seeds and report uncertainty
