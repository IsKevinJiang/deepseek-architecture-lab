import torch

from deepseek_lab.feed_forward import SwiGLU


def test_swiglu_preserves_leading_dimensions_and_hidden_size():
    module = SwiGLU(hidden_size=8, intermediate_size=16)
    input_tensor = torch.randn(2, 3, 8)

    output = module(input_tensor)

    assert output.shape == input_tensor.shape


def test_swiglu_uses_three_bias_free_projections():
    module = SwiGLU(hidden_size=4, intermediate_size=6)

    assert module.gate_proj.bias is None
    assert module.up_proj.bias is None
    assert module.down_proj.bias is None

    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    assert parameter_count == 3 * 4 * 6


def test_swiglu_matches_known_values():
    module = SwiGLU(hidden_size=2, intermediate_size=2)
    identity = torch.eye(2)

    with torch.no_grad():
        module.gate_proj.weight.copy_(identity)
        module.up_proj.weight.copy_(identity)
        module.down_proj.weight.copy_(identity)

    input_tensor = torch.tensor([[[1.0, -1.0]]])
    expected = torch.tensor([[[0.7310586, 0.2689414]]])

    actual = module(input_tensor)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_swiglu_produces_finite_gradients_for_input_and_all_weights():
    module = SwiGLU(hidden_size=8, intermediate_size=16)
    input_tensor = torch.randn(2, 3, 8, requires_grad=True)

    loss = module(input_tensor).sum()
    loss.backward()

    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()

    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
