import pytest
import torch
import torch.nn.functional as F

from modelforge.multi_head_attention import MHA


def reference_attention(module, input_tensor, offset=0):
    batch, sequence, _ = input_tensor.shape

    query = module.q_proj(input_tensor).reshape(
        batch, sequence, module.num_heads, module.head_dim
    )
    key = module.k_proj(input_tensor).reshape(
        batch, sequence, module.num_heads, module.head_dim
    )
    value = module.v_proj(input_tensor).reshape(
        batch, sequence, module.num_heads, module.head_dim
    )

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query, key = module.RoPE(query, key, offset)

    context = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
    )
    merged = context.transpose(1, 2).reshape(batch, sequence, module.hidden_dim)
    return module.out_proj(merged)


def test_attention_preserves_batch_sequence_and_hidden_shape():
    module = MHA(hidden_dim=8, num_heads=2, max_seq_len=6)
    input_tensor = torch.randn(3, 5, 8)

    output = module(input_tensor)

    assert output.shape == input_tensor.shape
    assert torch.isfinite(output).all()


def test_attention_has_four_bias_free_model_width_projections():
    module = MHA(hidden_dim=8, num_heads=2, max_seq_len=6)

    assert module.q_proj.weight.shape == (8, 8)
    assert module.k_proj.weight.shape == (8, 8)
    assert module.v_proj.weight.shape == (8, 8)
    assert module.out_proj.weight.shape == (8, 8)
    assert module.q_proj.bias is None
    assert module.k_proj.bias is None
    assert module.v_proj.bias is None
    assert module.out_proj.bias is None
    assert sum(parameter.numel() for parameter in module.parameters()) == 4 * 8 * 8


def test_attention_rejects_hidden_size_not_divisible_by_heads():
    with pytest.raises(ValueError, match="divis"):
        MHA(hidden_dim=10, num_heads=3)


def test_attention_matches_fused_reference_calculation():
    torch.manual_seed(0)
    module = MHA(hidden_dim=8, num_heads=2, max_seq_len=6)
    input_tensor = torch.randn(2, 5, 8)

    expected = reference_attention(module, input_tensor)
    actual = module(input_tensor)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_attention_prevents_future_tokens_from_affecting_past_outputs():
    torch.manual_seed(0)
    module = MHA(hidden_dim=8, num_heads=2, max_seq_len=6)
    original = torch.randn(1, 5, 8)
    changed_future = original.clone()
    changed_future[:, 3:] = torch.randn_like(changed_future[:, 3:]) * 100

    original_output = module(original)
    changed_output = module(changed_future)

    torch.testing.assert_close(
        original_output[:, :3],
        changed_output[:, :3],
        rtol=1e-5,
        atol=1e-6,
    )


def test_attention_passes_offset_bounds_through_to_rope():
    module = MHA(hidden_dim=8, num_heads=2, max_seq_len=4)
    input_tensor = torch.randn(1, 2, 8)

    with pytest.raises(ValueError):
        module(input_tensor, offset=3)


def test_attention_produces_finite_gradients_for_input_and_all_weights():
    module = MHA(hidden_dim=8, num_heads=2, max_seq_len=6)
    input_tensor = torch.randn(2, 5, 8, requires_grad=True)

    loss = module(input_tensor).square().mean()
    loss.backward()

    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
