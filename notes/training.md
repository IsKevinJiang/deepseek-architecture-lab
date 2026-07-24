# Stage 3: Training Engine

## Purpose

The training engine repeatedly gives the model token sequences, measures its next-token predictions, calculates gradients, and updates its parameters. It also handles stable optimization, validation, logging, and exact checkpoint resumption.

## 1. Batch, target, and logit shapes

The shard loader returns input and target tensors with shape `(B, S)`:

- `B` is the batch size.
- `S` is the sequence length.
- Each target is the token immediately after its matching input token.

The model produces logits with shape `(B, S, V)`, where `V` is the vocabulary size. Each position therefore has one score for every possible next token.

For cross-entropy loss, the tensors are reshaped as follows:

- Logits: `(B, S, V) -> (B * S, V)`
- Targets: `(B, S) -> (B * S)`

Using `-1` in `reshape` tells PyTorch to infer that dimension. Using `logits.size(-1)` is safer than repeating a hard-coded vocabulary size because it reads `V` from the actual tensor.

## 2. Cross-entropy loss

Logits are raw, unnormalized scores rather than probabilities.

Cross-entropy compares the vocabulary scores at every position with the correct next-token ID. Internally, it performs the equivalent of log-softmax and negative log-likelihood in a numerically stable operation.

Raw logits should be passed directly to cross-entropy. We should not apply softmax first. A lower loss means the model is assigning more probability to the correct next tokens.

## 3. Token types and BF16 autocast

Token IDs use different data types at different stages:

- Shard files store token IDs as `uint16` to save disk space. All 50,257 GPT-2 token IDs fit within its maximum value of 65,535.
- The loader converts batch token IDs to `torch.long`.
- Embedding layers require integer index tensors such as `torch.long`.

Token IDs must not become BF16 because they are discrete indices.

BF16 autocast is used during the forward pass. It lets supported GPU operations, especially matrix multiplications, run in BF16 for better speed and memory efficiency. Model parameters may remain stored in FP32 while autocast selects BF16 for suitable computations.

## 4. One optimizer update

A standard optimizer update follows this order:

1. Clear old gradients with `optimizer.zero_grad()`.
2. Load inputs and targets and move them to the selected device.
3. Run the forward pass under BF16 autocast.
4. Calculate cross-entropy loss.
5. Run `loss.backward()` to calculate gradients.
6. Clip the combined gradient norm.
7. Set the learning rate for the current update.
8. Run `optimizer.step()` to update the parameters.

PyTorch accumulates new gradients into each parameter's existing `.grad` field. It does not automatically replace them. This enables gradient accumulation, but gradients must be deliberately cleared before starting a new optimizer update.

## 5. Gradient accumulation

Gradient accumulation simulates a larger batch when that full batch would not fit in GPU memory.

If a microbatch contains `B * S` tokens and there are `A` accumulation steps, the effective tokens per optimizer update are:

`B * S * A`

For one optimizer update:

1. Clear gradients once.
2. Run one microbatch through the model.
3. Divide its loss by `A`.
4. Call `backward()` to add its gradients.
5. Repeat for all `A` microbatches.
6. Clip the combined gradient norm once.
7. Call `optimizer.step()` once.

Dividing each microbatch loss by `A` makes the accumulated result approximately equal to the mean gradient of one larger batch. Without this division, increasing accumulation would also multiply the gradient magnitude.

Learning-rate scheduling, logging, validation, and checkpoint intervals should normally count optimizer updates, not individual microbatches.

## 6. Gradient clipping

An unusual batch can create a very large gradient norm. This could move the weights too far in one update, destabilize training, or produce non-finite values.

Global norm clipping measures the combined norm of all parameter gradients. If the norm exceeds the configured maximum, every gradient is multiplied by the same scale factor. This reduces the update magnitude while preserving its overall direction.

Clipping individual gradient values could distort the direction, so global norm clipping is preferred. It belongs after all accumulated backward passes and before `optimizer.step()`.

The returned pre-clipping norm is useful to log. Frequent large gradient norms can reveal unstable training.

## 7. AdamW parameter grouping

AdamW applies weight decay separately from its gradient-based update. Weight decay helps regularize weight matrices but is normally not applied to every parameter.

The groups are:

- **Decay group:** parameters with two or more dimensions, including linear and embedding weight matrices.
- **No-decay group:** one-dimensional parameters such as normalization scales, plus scalar parameters.

Optimizer groups contain the actual parameter objects. Parameter identities are only used to verify the grouping.

The checks confirm that:

1. Neither group contains duplicate parameter objects.
2. The groups do not overlap.
3. Their union contains every trainable parameter exactly once.

Both groups receive the same scheduled learning rate. The decay group uses the configured weight decay, while the no-decay group uses zero weight decay.

## 8. Learning-rate warmup and cosine decay

The learning rate controls the size of an optimizer update.

At the beginning, gradients can be noisy and AdamW has not built useful momentum estimates. Linear warmup gradually raises the learning rate to its configured maximum, reducing the chance of unstable early updates.

After warmup, cosine decay gradually lowers the learning rate toward a configured minimum. Smaller updates near the end reduce the chance of repeatedly stepping past a good solution.

Warmup is normally a relatively small part of training, not half of the run. The maximum learning rate is reached at the end of warmup, and cosine decay covers the remaining updates.

The scheduled learning rate is assigned to every optimizer group before `optimizer.step()`. When training resumes, the schedule must use the restored update count rather than starting warmup again.

## 9. Training and validation

Training batches are used for forward passes, backward passes, and weight updates. Validation batches measure performance on unseen data and must never update the model.

`model.train()` enables training-specific behavior such as dropout. `model.eval()` disables that behavior. However, `model.eval()` does not turn off gradient tracking or freeze parameters by itself, so validation should also use `torch.inference_mode()` or `torch.no_grad()`.

A validation pass should:

1. Switch the model to evaluation mode.
2. Disable gradient tracking.
3. Read a fixed number of validation batches.
4. Calculate their losses without `backward()` or `optimizer.step()`.
5. Average the losses.
6. Return the model to training mode.

Using deterministic validation batches makes measurements comparable. If training loss decreases while validation loss rises, the model may be overfitting, or the validation sample may be too small or unrepresentative.

## 10. Checkpointing and exact resumption

A checkpoint allows training to continue after an interruption. Saving only `model.state_dict()` is enough for inference, but it is not enough for exact training resumption.

An exact checkpoint stores:

- Model state
- Optimizer state, including AdamW momentum estimates
- Completed optimizer-update count
- Training loader state
- Validation loader state when its exact position matters
- CPU random-number-generator state
- CUDA random-number-generator state
- Training configuration needed to reconstruct the run

The model must be created before its state is loaded. The optimizer must then be created using that model's parameter objects before loading the optimizer state. This reconnects AdamW's saved state to the new model instance.

If a checkpoint was saved after update 600, the next update is 601. The learning-rate schedule must continue from the restored update instead of warming up again.

Restoring loader and random states preserves the future batches and random operations. Without them, training may continue successfully, but it will follow a different numerical path.

Checkpoints should be saved atomically: write to a temporary file and replace the final checkpoint only after the write succeeds. An interrupted save then cannot destroy the previous valid checkpoint.

An exact-resume test compares an uninterrupted run against a run that saves, reconstructs its objects, reloads, and finishes. Their losses, parameters, optimizer state, loader state, and random state should match.

## 11. Central training configuration

A training configuration keeps related settings together instead of scattering hard-coded values throughout the script. It can contain:

- Model dimensions
- Data location and batch shape
- Device and autocast data type
- AdamW settings
- Gradient accumulation and clipping
- Warmup and cosine-decay settings
- Total optimizer updates
- Logging, validation, and checkpoint intervals

The configuration should be created before the model, loaders, and optimizer so every object uses consistent values. Derived values should come from the configuration or actual tensors. This prevents mismatches such as using different sequence lengths in the loader and model.

Model architecture settings and training settings represent different concerns and may eventually be separated into different configuration objects.

## Complete training flow

1. Create and validate the configuration.
2. Seed the random-number generators.
3. Create the training and validation shard loaders.
4. Create the model and move it to the selected device.
5. Create AdamW with verified decay and no-decay groups.
6. Load a checkpoint if training is being resumed.
7. For each remaining optimizer update:
   - Set the scheduled learning rate.
   - Clear the previous gradients.
   - Accumulate gradients over the configured microbatches.
   - Clip the combined gradient norm.
   - Update the model parameters.
   - Log training measurements when required.
   - Run validation when required.
   - Save an atomic checkpoint when required.
8. Save a final checkpoint after the last update.
