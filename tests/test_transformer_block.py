import pytest
import torch

from modelforge.feed_forward import SwiGLU
from modelforge.multi_head_attention import MHA
from modelforge.normalization import RMSNorm
from modelforge.transformer_block import TransformerBlock


def test_transformer_block_preserves_shape_with_default_offset():
    block = TransformerBlock(
        hidden_dim=8,
        num_heads=2,
        max_seq_len=6,
        intermediate_dim=16,
    )
    input_tensor = torch.randn(2, 5, 8)

    output = block(input_tensor)

    assert output.shape == input_tensor.shape
    assert torch.isfinite(output).all()


def test_transformer_block_owns_expected_independent_components():
    block = TransformerBlock(
        hidden_dim=8,
        num_heads=2,
        max_seq_len=6,
        intermediate_dim=16,
    )

    assert isinstance(block.rmsnorm1, RMSNorm)
    assert isinstance(block.rmsnorm2, RMSNorm)
    assert isinstance(block.attention, MHA)
    assert isinstance(block.swiglu, SwiGLU)
    assert block.rmsnorm1 is not block.rmsnorm2
    assert block.rmsnorm1.weight.data_ptr() != block.rmsnorm2.weight.data_ptr()


def test_transformer_block_matches_pre_norm_residual_equations():
    torch.manual_seed(0)
    block = TransformerBlock(
        hidden_dim=8,
        num_heads=2,
        max_seq_len=6,
        intermediate_dim=16,
    )
    input_tensor = torch.randn(2, 5, 8)

    after_attention = input_tensor + block.attention(block.rmsnorm1(input_tensor), 0)
    expected = after_attention + block.swiglu(block.rmsnorm2(after_attention))
    actual = block(input_tensor, 0)

    torch.testing.assert_close(actual, expected)


def test_transformer_block_has_expected_parameter_count():
    hidden_dim = 8
    intermediate_dim = 16
    block = TransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=2,
        max_seq_len=6,
        intermediate_dim=intermediate_dim,
    )
    expected = 2 * hidden_dim + 4 * hidden_dim**2 + 3 * hidden_dim * intermediate_dim

    assert sum(parameter.numel() for parameter in block.parameters()) == expected


def test_transformer_block_passes_offset_to_attention():
    block = TransformerBlock(
        hidden_dim=8,
        num_heads=2,
        max_seq_len=4,
        intermediate_dim=16,
    )
    input_tensor = torch.randn(1, 2, 8)

    with pytest.raises(ValueError):
        block(input_tensor, offset=3)


def test_transformer_block_produces_finite_gradients_for_input_and_parameters():
    block = TransformerBlock(
        hidden_dim=8,
        num_heads=2,
        max_seq_len=6,
        intermediate_dim=16,
    )
    input_tensor = torch.randn(2, 5, 8, requires_grad=True)

    loss = block(input_tensor, 0).square().mean()
    loss.backward()

    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
