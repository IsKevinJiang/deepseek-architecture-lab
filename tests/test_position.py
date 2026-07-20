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
