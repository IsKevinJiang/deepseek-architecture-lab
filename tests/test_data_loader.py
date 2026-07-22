import numpy as np
import pytest
import torch

from deepseek_lab.data import ShardDataLoader


def write_shard(path, tokens):
    tokens = np.asarray(tokens, dtype=np.uint16)
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)

    with path.open("wb") as file:
        file.write(header.tobytes())
        file.write(tokens.tobytes())


def test_loader_finds_and_sorts_only_requested_split(tmp_path):
    write_shard(tmp_path / "dataset_train_000002.bin", np.arange(20, 30))
    write_shard(tmp_path / "dataset_val_000000.bin", np.arange(90, 100))
    write_shard(tmp_path / "dataset_train_000001.bin", np.arange(10))

    loader = ShardDataLoader(tmp_path, batch_size=1, sequence_length=4, split="train")

    assert [path.name for path in loader.shards] == [
        "dataset_train_000001.bin",
        "dataset_train_000002.bin",
    ]
    assert loader.current_shard == 0
    assert loader.position == 0
    np.testing.assert_array_equal(loader.tokens, np.arange(10, dtype=np.uint16))


def test_loader_rejects_missing_split(tmp_path):
    write_shard(tmp_path / "dataset_val_000000.bin", np.arange(10))

    with pytest.raises(ValueError, match="No shards found"):
        ShardDataLoader(tmp_path, batch_size=1, sequence_length=4, split="train")


def test_next_batch_has_shifted_targets_shape_and_dtype(tmp_path):
    write_shard(tmp_path / "dataset_train_000001.bin", np.arange(20))
    loader = ShardDataLoader(tmp_path, batch_size=2, sequence_length=3, split="train")

    inputs, targets = loader.next_batch()

    torch.testing.assert_close(
        inputs,
        torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
    )
    torch.testing.assert_close(
        targets,
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
    )
    assert inputs.shape == (2, 3)
    assert targets.shape == (2, 3)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    assert loader.position == 6


def test_successive_batches_continue_from_previous_position(tmp_path):
    write_shard(tmp_path / "dataset_train_000001.bin", np.arange(20))
    loader = ShardDataLoader(tmp_path, batch_size=2, sequence_length=3, split="train")

    loader.next_batch()
    inputs, targets = loader.next_batch()

    torch.testing.assert_close(
        inputs,
        torch.tensor([[6, 7, 8], [9, 10, 11]], dtype=torch.long),
    )
    torch.testing.assert_close(
        targets,
        torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.long),
    )
    assert loader.position == 12


def test_loader_advances_shards_and_wraps_after_last_shard(tmp_path):
    write_shard(tmp_path / "dataset_train_000001.bin", np.arange(5))
    write_shard(tmp_path / "dataset_train_000002.bin", np.arange(100, 105))
    loader = ShardDataLoader(tmp_path, batch_size=1, sequence_length=4, split="train")

    first_inputs, _ = loader.next_batch()
    second_inputs, _ = loader.next_batch()
    wrapped_inputs, _ = loader.next_batch()

    torch.testing.assert_close(first_inputs, torch.tensor([[0, 1, 2, 3]]))
    torch.testing.assert_close(second_inputs, torch.tensor([[100, 101, 102, 103]]))
    torch.testing.assert_close(wrapped_inputs, torch.tensor([[0, 1, 2, 3]]))
    assert loader.current_shard == 0
    assert loader.position == 4
