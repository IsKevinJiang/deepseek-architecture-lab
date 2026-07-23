import numpy as np

from modelforge.data import ShardWriter


def read_shard(path):
    with path.open("rb") as file:
        header = np.fromfile(file, dtype=np.int32, count=256)
        tokens = np.fromfile(file, dtype=np.uint16)
    return header, tokens


def assert_valid_shard(path, expected_tokens):
    header, tokens = read_shard(path)

    assert len(header) == 256
    assert header[0] == 20240520
    assert header[1] == 1
    assert header[2] == len(expected_tokens)
    assert np.all(header[3:] == 0)
    np.testing.assert_array_equal(tokens, expected_tokens)
    assert path.stat().st_size == 1024 + 2 * len(expected_tokens)


def test_shard_writer_creates_output_directory_and_initial_state(tmp_path):
    output_dir = tmp_path / "nested" / "shards"

    writer = ShardWriter(shard_size=10, output_dir=output_dir)

    assert output_dir.is_dir()
    assert writer.buffer.shape == (10,)
    assert writer.buffer.dtype == np.uint16
    assert writer.position == 0
    assert writer.shard_index == 0


def test_finish_writes_partial_validation_shard(tmp_path):
    writer = ShardWriter(shard_size=10, output_dir=tmp_path)
    tokens = np.array([7, 8, 9, 10], dtype=np.uint16)

    writer.add(tokens)
    assert list(tmp_path.glob("*.bin")) == []

    writer.finish()

    path = tmp_path / "dataset_val_000000.bin"
    assert_valid_shard(path, tokens)
    assert writer.position == 0
    assert writer.shard_index == 1


def test_tokens_crossing_boundary_are_preserved_without_duplication(tmp_path):
    writer = ShardWriter(shard_size=10, output_dir=tmp_path)
    first = np.array([0, 1, 2, 3], dtype=np.uint16)
    second = np.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=np.uint16)

    writer.add(first)
    writer.add(second)
    writer.finish()

    validation_path = tmp_path / "dataset_val_000000.bin"
    training_path = tmp_path / "dataset_train_000001.bin"
    assert_valid_shard(validation_path, np.arange(10, dtype=np.uint16))
    assert_valid_shard(training_path, np.array([10, 11], dtype=np.uint16))

    _, validation_tokens = read_shard(validation_path)
    _, training_tokens = read_shard(training_path)
    reconstructed = np.concatenate((validation_tokens, training_tokens))
    np.testing.assert_array_equal(reconstructed, np.arange(12, dtype=np.uint16))


def test_exact_full_shard_does_not_create_empty_final_shard(tmp_path):
    writer = ShardWriter(shard_size=10, output_dir=tmp_path)
    tokens = np.arange(10, dtype=np.uint16)

    writer.add(tokens)
    writer.finish()

    paths = sorted(tmp_path.glob("*.bin"))
    assert paths == [tmp_path / "dataset_val_000000.bin"]
    assert_valid_shard(paths[0], tokens)


def test_single_add_can_span_multiple_shards(tmp_path):
    writer = ShardWriter(shard_size=10, output_dir=tmp_path)
    tokens = np.arange(25, dtype=np.uint16)

    writer.add(tokens)
    writer.finish()

    paths = sorted(tmp_path.glob("*.bin"))
    assert [path.name for path in paths] == [
        "dataset_train_000001.bin",
        "dataset_train_000002.bin",
        "dataset_val_000000.bin",
    ]

    shards_by_index = sorted(paths, key=lambda path: path.name[-10:-4])
    payloads = [read_shard(path)[1] for path in shards_by_index]
    np.testing.assert_array_equal(np.concatenate(payloads), tokens)
    assert [len(payload) for payload in payloads] == [10, 10, 5]
