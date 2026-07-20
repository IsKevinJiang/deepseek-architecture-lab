import pytest
import torch

from deepseek_lab.position import RotaryEmbedding, rotate_half


def test_rotate_half_matches_known_vector():
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])

    actual = rotate_half(input_tensor)

    torch.testing.assert_close(actual, expected)


def test_rotate_half_preserves_shape_and_leading_dimensions():
    input_tensor = torch.randn(2, 3, 5, 8)

    output = rotate_half(input_tensor)

    assert output.shape == input_tensor.shape


def test_two_half_rotations_equal_negated_input():
    input_tensor = torch.randn(2, 3, 5, 8)

    output = rotate_half(rotate_half(input_tensor))

    torch.testing.assert_close(output, -input_tensor)


def test_rotate_half_rejects_odd_final_dimension():
    input_tensor = torch.randn(2, 3, 5)

    with pytest.raises(ValueError, match="even"):
        rotate_half(input_tensor)


def test_rotary_embedding_builds_expected_cache_shapes():
    rotary = RotaryEmbedding(dim=8, max_seq_len=4)

    assert rotary.base_freq.shape == (4,)
    assert rotary.cos_rot.shape == (4, 8)
    assert rotary.sin_rot.shape == (4, 8)


def test_rotary_embedding_builds_expected_frequencies():
    rotary = RotaryEmbedding(dim=8, max_seq_len=4, base=10_000)
    expected = torch.tensor([1.0, 0.1, 0.01, 0.001])

    torch.testing.assert_close(rotary.base_freq, expected)


def test_rotary_embedding_position_zero_is_identity():
    rotary = RotaryEmbedding(dim=8, max_seq_len=4)

    torch.testing.assert_close(rotary.cos_rot[0], torch.ones(8))
    torch.testing.assert_close(rotary.sin_rot[0], torch.zeros(8))


def test_rotary_embedding_uses_nonpersistent_buffers_not_parameters():
    rotary = RotaryEmbedding(dim=8, max_seq_len=4)
    buffer_names = set(dict(rotary.named_buffers()))

    assert buffer_names == {"base_freq", "cos_rot", "sin_rot"}
    assert list(rotary.parameters()) == []
    assert rotary.state_dict() == {}


def test_rotary_embedding_rejects_odd_dimension():
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(dim=7, max_seq_len=4)


def test_rotary_forward_position_zero_is_identity_with_different_head_counts():
    rotary = RotaryEmbedding(dim=8, max_seq_len=4)
    query = torch.randn(2, 3, 1, 8)
    key = torch.randn(2, 1, 1, 8)

    rotated_query, rotated_key = rotary(query, key)

    torch.testing.assert_close(rotated_query, query)
    torch.testing.assert_close(rotated_key, key)


def test_rotary_forward_matches_cached_formula_with_position_offset():
    rotary = RotaryEmbedding(dim=8, max_seq_len=6)
    query = torch.randn(1, 2, 3, 8)
    key = torch.randn(1, 2, 3, 8)
    offset = 2
    cosine = rotary.cos_rot[2:5].unsqueeze(0).unsqueeze(0)
    sine = rotary.sin_rot[2:5].unsqueeze(0).unsqueeze(0)
    expected_query = query * cosine + rotate_half(query) * sine
    expected_key = key * cosine + rotate_half(key) * sine

    actual_query, actual_key = rotary(
        query,
        key,
        offset=offset,
    )

    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)


def test_rotary_forward_preserves_shapes_dtypes_and_vector_magnitudes():
    rotary = RotaryEmbedding(dim=8, max_seq_len=4)
    query = torch.randn(2, 3, 4, 8).to(torch.bfloat16)
    key = torch.randn(2, 1, 4, 8).to(torch.bfloat16)

    rotated_query, rotated_key = rotary(query, key)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert rotated_query.dtype == query.dtype
    assert rotated_key.dtype == key.dtype
    torch.testing.assert_close(
        rotated_query.float().norm(dim=-1),
        query.float().norm(dim=-1),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        rotated_key.float().norm(dim=-1),
        key.float().norm(dim=-1),
        rtol=1e-2,
        atol=1e-2,
    )


def test_rotary_forward_rejects_wrong_feature_dimension():
    rotary = RotaryEmbedding(dim=8, max_seq_len=4)
    query = torch.randn(1, 2, 4, 6)
    key = torch.randn(1, 2, 4, 8)

    with pytest.raises(ValueError):
        rotary(query, key)


def test_rotary_forward_rejects_mismatched_sequence_lengths():
    rotary = RotaryEmbedding(dim=8, max_seq_len=4)
    query = torch.randn(1, 2, 3, 8)
    key = torch.randn(1, 2, 4, 8)

    with pytest.raises(ValueError):
        rotary(query, key)


@pytest.mark.parametrize("offset", [-1, 3])
def test_rotary_forward_rejects_positions_outside_cache(offset):
    rotary = RotaryEmbedding(dim=8, max_seq_len=4)
    query = torch.randn(1, 2, 2, 8)
    key = torch.randn(1, 2, 2, 8)

    with pytest.raises(ValueError):
        rotary(query, key, offset=offset)
