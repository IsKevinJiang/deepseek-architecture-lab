# Stage 2: Data Preparation and Loading

## Purpose of this stage

The purpose of the data pipeline is to turn raw text documents into next-token training batches efficiently and reproducibly. The full dataset should not need to fit in RAM, and training should be able to stop and resume at the exact same place.

The two main components are:

- `ShardWriter`: converts tokenized documents into binary shard files.
- `ShardDataLoader`: reads those shards and produces input and target batches.

## End-to-end pipeline

The data moves through the following stages:

1. Stream documents from FineWeb-Edu rather than downloading the entire dataset into RAM.
2. Tokenize each document with the GPT-2 BPE tokenizer from `tiktoken`.
3. Add an end-of-text token at the start of each document to mark a document boundary.
4. Convert the token IDs to `uint16` values.
5. Add the tokens to the `ShardWriter` buffer.
6. Write the buffer to a binary shard whenever it becomes full.
7. Memory-map the shard token data with the `ShardDataLoader`.
8. Slice token windows and turn them into shifted input and target tensors.

The current preparation script is a smoke test. It streams 100 documents from the FineWeb-Edu `sample-10BT` configuration and uses small 10,000-token shards. It does not prepare or train on the complete 10-billion-token dataset yet.

## Tokenization

Language models operate on token IDs rather than raw text. The GPT-2 tokenizer uses byte-pair encoding to divide text into tokens and map each token to an integer ID.

An end-of-text token is placed at each document boundary. This helps the model learn that the next document is not a continuation of the previous document.

Tokens are stored as `uint16` because this type represents values from 0 through 65,535. The GPT-2 vocabulary contains 50,257 token IDs, so every possible ID fits. Using two bytes per token is more storage-efficient than using a 32-bit or 64-bit integer.

The preparation step checks that every token ID fits before converting it to `uint16`. This prevents silent overflow and corrupted training data.

## Binary shards

A shard is one portion of the tokenized dataset stored in a separate `.bin` file. Sharding makes a large dataset easier to write, load, shuffle, validate, resume, and distribute across machines.

The `ShardWriter` maintains a fixed-size in-memory token buffer. If a document does not fit in the remaining buffer space, the writer fills the current shard, writes it to disk, and continues the remaining tokens in the next shard. Tokens should not be lost or duplicated at a shard boundary.

Calling `finish` writes the final partially filled buffer. It does nothing when the buffer is empty, which prevents an empty shard from being created after an exactly full shard.

Each binary shard contains:

1. A 1,024-byte header made from 256 32-bit integers.
2. A payload containing the `uint16` token IDs.

The important header fields are:

- Magic number: identifies the file as one of this project's shard files.
- Format version: allows the format to change safely in the future.
- Token count: tells the loader how many valid tokens are stored in the payload.

The header does not contain the training or validation split. The split is encoded in the shard filename.

## Training and validation split

Training data is used to update model parameters. Validation data is never used for parameter updates; it measures how well the model predicts unseen data.

The validation set must stay fixed across the dense, MLA, and MoE experiments. Otherwise, their validation losses would not be directly comparable.

The current writer assigns the first completed shard to validation and all later shards to training. This is acceptable for an early smoke test, but it is not the preferred design for the full dataset.

For the real dataset, complete documents should be assigned randomly and reproducibly to one split before sharding. A document must never be divided between training and validation. If part of the same document appears in both splits, data leakage can make validation performance look artificially good. Selecting only the first streamed documents can also produce a validation set that is not representative of the full dataset.

## Memory mapping

Memory mapping makes a file on disk behave like an array without copying the complete file into RAM first.

The loader reads the 1,024-byte header normally and maps the token payload as a read-only `uint16` array. When a batch requests a small token range, the operating system loads only the necessary file pages into RAM. It may cache recently used pages and discard them when memory is needed.

The data flow during loading is:

`binary shard on disk -> memory-mapped token window -> PyTorch tensor in RAM -> GPU during training`

Memory mapping is useful because:

- The dataset can be much larger than system RAM.
- Opening a shard does not require loading every token.
- Token ranges can still be accessed with normal array-style slicing.
- The operating system manages file caching automatically.

Memory mapping does not mean that no RAM is used. The requested pages and the PyTorch batch still occupy memory; the entire shard simply does not need to be loaded at once.

## Next-token batches

For batch size `B` and sequence length `S`, the loader needs `B * S + 1` consecutive tokens.

The first `B * S` tokens become the inputs. The last `B * S` tokens, starting one position later, become the targets. Both are reshaped to `(B, S)`.

Each target is therefore the token that immediately follows its matching input token. This trains the language model to predict the next token from the tokens that came before it.

The loader advances by `B * S`, not `B * S + 1`. The final token in the current window must become the first input token in the next window. Advancing past it would skip one valid next-token relationship between batches.

## Moving between shards and epochs

If the current shard does not contain enough remaining tokens for a full batch window, the loader moves to the next shard. The current implementation does not join the small remainder of one shard to the start of another, so those leftover tokens are skipped.

After the final shard, the loader wraps around to the beginning and increments the epoch counter. Training shards are reshuffled at the epoch boundary. Validation shards remain sorted so every validation run reads data in the same order.

The loader limits how many shards it can try while searching for a sufficiently large batch. If every shard is too small, it raises an error instead of cycling forever.

## Determinism and shuffling

Training shards are shuffled so the model does not see them in the same order every epoch. A seeded NumPy random-number generator makes that shuffle reproducible.

Validation order remains deterministic because comparable validation measurements must use the same examples in the same order. Determinism is also important for debugging and exact checkpoint-resume tests.

## Exact loader resume

Saving only the current shard is not enough to resume exactly. The loader state records:

- Current epoch.
- Current shard index.
- Exact token position inside that shard.
- Current shuffled shard order.
- Random-number-generator state.
- Training or validation split.
- Batch size and sequence length.

When restoring, the loader checks that its configuration matches the saved state, reconstructs the saved shard order, verifies that every file still exists, loads the correct shard, restores the token position, and restores the random state.
