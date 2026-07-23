import pytest
import torch
import torch.nn as nn

from modelforge.model import Model
from modelforge.normalization import RMSNorm
from modelforge.transformer_block import TransformerBlock


def make_tiny_model():
    return Model(
        hidden_dim=8,
        num_heads=2,
        max_seq_len=6,
        intermediate_dim=16,
        vocab_size=32,
        num_layers=2,
    )


def test_model_returns_finite_logits_with_expected_shape():
    model = make_tiny_model()
    token_ids = torch.randint(0, model.vocab_size, (2, 5))

    logits = model(token_ids)

    assert logits.shape == (2, 5, model.vocab_size)
    assert torch.isfinite(logits).all()


def test_model_stores_configuration_values():
    model = make_tiny_model()

    assert model.hidden_dim == 8
    assert model.num_heads == 2
    assert model.max_seq_len == 6
    assert model.intermediate_dim == 16
    assert model.vocab_size == 32
    assert model.num_layers == 2


def test_model_contains_independent_transformer_blocks():
    model = make_tiny_model()

    assert isinstance(model.transformer_blocks, nn.ModuleList)
    assert len(model.transformer_blocks) == model.num_layers
    assert all(isinstance(block, TransformerBlock) for block in model.transformer_blocks)
    assert model.transformer_blocks[0] is not model.transformer_blocks[1]
    first_parameter = next(model.transformer_blocks[0].parameters())
    second_parameter = next(model.transformer_blocks[1].parameters())
    assert first_parameter.data_ptr() != second_parameter.data_ptr()


def test_model_uses_separate_embedding_and_output_weights():
    model = make_tiny_model()

    assert model.embedding.weight.shape == (model.vocab_size, model.hidden_dim)
    assert model.linear.weight.shape == (model.vocab_size, model.hidden_dim)
    assert model.embedding.weight.data_ptr() != model.linear.weight.data_ptr()
    assert model.linear.bias is None


def test_model_is_causal():
    torch.manual_seed(0)
    model = make_tiny_model()
    original = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 20, 21]])

    original_logits = model(original)
    changed_logits = model(changed_future)

    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3])


def test_model_rejects_positions_beyond_maximum_sequence_length():
    model = make_tiny_model()
    token_ids = torch.randint(0, model.vocab_size, (1, 2))

    with pytest.raises(ValueError):
        model(token_ids, offset=5)


def test_model_has_expected_parameter_count():
    model = make_tiny_model()
    expected = (
        2 * model.vocab_size * model.hidden_dim
        + model.num_layers
        * (
            4 * model.hidden_dim**2
            + 3 * model.hidden_dim * model.intermediate_dim
            + 2 * model.hidden_dim
        )
        + model.hidden_dim
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == expected


def test_model_produces_finite_gradients_for_all_parameters():
    model = make_tiny_model()
    token_ids = torch.randint(0, model.vocab_size, (2, 5))

    loss = model(token_ids).square().mean()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_model_initializes_linear_and_embedding_weights_with_expected_normal_distribution():
    torch.manual_seed(0)
    model = make_tiny_model()
    initialized_weights = []

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            initialized_weights.append(module.weight.detach().flatten())

    all_weights = torch.cat(initialized_weights)
    assert abs(all_weights.mean().item()) < 0.003
    assert abs(all_weights.std().item() - 0.02) < 0.003


def test_model_initializes_every_rmsnorm_weight_to_one():
    model = make_tiny_model()

    for module in model.modules():
        if isinstance(module, RMSNorm):
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
