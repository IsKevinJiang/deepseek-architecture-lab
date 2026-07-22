import copy

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


def test_loader_finds_only_requested_split(tmp_path):
    write_shard(tmp_path / "dataset_train_000002.bin", np.arange(20, 30))
    write_shard(tmp_path / "dataset_val_000000.bin", np.arange(90, 100))
    write_shard(tmp_path / "dataset_train_000001.bin", np.arange(10))

    loader = ShardDataLoader(
        tmp_path,
        batch_size=1,
        sequence_length=4,
        split="train",
        seed=32,
    )

    assert {path.name for path in loader.shards} == {
        "dataset_train_000001.bin",
        "dataset_train_000002.bin",
    }
    assert loader.current_shard == 0
    assert loader.position == 0

    expected_start = 0 if loader.shards[0].name.endswith("000001.bin") else 20
    np.testing.assert_array_equal(
        loader.tokens,
        np.arange(expected_start, expected_start + 10, dtype=np.uint16),
    )


def test_training_shuffle_is_reproducible_for_same_seed(tmp_path):
    for index in range(1, 6):
        write_shard(
            tmp_path / f"dataset_train_{index:06d}.bin",
            np.arange(index * 10, index * 10 + 10),
        )

    first = ShardDataLoader(tmp_path, 1, 4, "train", seed=123)
    second = ShardDataLoader(tmp_path, 1, 4, "train", seed=123)

    assert [path.name for path in first.shards] == [
        path.name for path in second.shards
    ]


def test_validation_shards_remain_sorted(tmp_path):
    write_shard(tmp_path / "dataset_val_000002.bin", np.arange(20, 30))
    write_shard(tmp_path / "dataset_val_000000.bin", np.arange(10))
    write_shard(tmp_path / "dataset_val_000001.bin", np.arange(10, 20))

    loader = ShardDataLoader(tmp_path, 1, 4, "val", seed=123)

    assert [path.name for path in loader.shards] == [
        "dataset_val_000000.bin",
        "dataset_val_000001.bin",
        "dataset_val_000002.bin",
    ]


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
    write_shard(tmp_path / "dataset_val_000000.bin", np.arange(5))
    write_shard(tmp_path / "dataset_val_000001.bin", np.arange(100, 105))
    loader = ShardDataLoader(tmp_path, batch_size=1, sequence_length=4, split="val")

    first_inputs, _ = loader.next_batch()
    second_inputs, _ = loader.next_batch()
    wrapped_inputs, _ = loader.next_batch()

    torch.testing.assert_close(first_inputs, torch.tensor([[0, 1, 2, 3]]))
    torch.testing.assert_close(second_inputs, torch.tensor([[100, 101, 102, 103]]))
    torch.testing.assert_close(wrapped_inputs, torch.tensor([[0, 1, 2, 3]]))
    assert loader.current_shard == 0
    assert loader.position == 4
    assert loader.epoch == 1


def test_training_shards_reshuffle_at_epoch_boundary(tmp_path):
    shard_names = [f"dataset_train_{index:06d}.bin" for index in range(1, 5)]
    for index, name in enumerate(shard_names):
        write_shard(tmp_path / name, np.arange(index * 10, index * 10 + 5))

    seed = 17
    expected_rng = np.random.default_rng(seed)
    expected_first_order = shard_names.copy()
    expected_rng.shuffle(expected_first_order)
    expected_second_order = expected_first_order.copy()
    expected_rng.shuffle(expected_second_order)

    loader = ShardDataLoader(tmp_path, 1, 4, "train", seed=seed)
    assert [path.name for path in loader.shards] == expected_first_order

    for _ in range(len(shard_names) + 1):
        loader.next_batch()

    assert loader.epoch == 1
    assert loader.current_shard == 0
    assert [path.name for path in loader.shards] == expected_second_order


def test_restored_loader_produces_identical_future_batches(tmp_path):
    for index in range(1, 5):
        start = index * 100
        write_shard(
            tmp_path / f"dataset_train_{index:06d}.bin",
            np.arange(start, start + 9),
        )

    original = ShardDataLoader(tmp_path, 1, 4, "train", seed=11)
    for _ in range(5):
        original.next_batch()

    saved_state = copy.deepcopy(original.state_dict())
    expected_batches = [original.next_batch() for _ in range(12)]

    restored = ShardDataLoader(tmp_path, 1, 4, "train", seed=999)
    restored.load_state_dict(saved_state)

    assert restored.epoch == saved_state["epoch"]
    assert restored.current_shard == saved_state["current_shard"]
    assert restored.position == saved_state["token_position"]
    assert [path.name for path in restored.shards] == saved_state["shard_names"]

    for expected_inputs, expected_targets in expected_batches:
        actual_inputs, actual_targets = restored.next_batch()
        torch.testing.assert_close(actual_inputs, expected_inputs)
        torch.testing.assert_close(actual_targets, expected_targets)


def test_load_state_rejects_incompatible_configuration(tmp_path):
    write_shard(tmp_path / "dataset_train_000001.bin", np.arange(20))
    original = ShardDataLoader(tmp_path, 1, 4, "train", seed=1)
    incompatible = ShardDataLoader(tmp_path, 2, 4, "train", seed=1)

    with pytest.raises(ValueError, match="mismatch"):
        incompatible.load_state_dict(original.state_dict())
