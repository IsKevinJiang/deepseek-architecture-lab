import torch

from modelforge.normalization import RMSNorm


def test_rmsnorm_preserves_shape_and_registers_weight():
    norm = RMSNorm(8)
    input_tensor = torch.randn(2, 3, 8)

    output = norm(input_tensor)

    assert output.shape == input_tensor.shape
    assert norm.weight.shape == (8,)
    assert norm.weight.requires_grad
    assert "weight" in dict(norm.named_parameters())


def test_rmsnorm_matches_known_vector():
    norm = RMSNorm(2, eps=1e-6)
    input_tensor = torch.tensor([[[3.0, 4.0]]])
    expected = torch.tensor([[[0.848528, 1.131371]]])

    actual = norm(input_tensor)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_rmsnorm_normalizes_each_token_independently():
    norm = RMSNorm(4, eps=1e-6)
    input_tensor = torch.tensor(
        [
            [[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]],
            [[-2.0, 0.0, 2.0, 0.0], [0.5, -1.0, 1.5, -2.0]],
        ]
    )

    output = norm(input_tensor)
    output_mean_squares = output.square().mean(dim=-1)

    torch.testing.assert_close(
        output_mean_squares,
        torch.ones_like(output_mean_squares),
        rtol=1e-5,
        atol=1e-5,
    )


def test_rmsnorm_produces_finite_gradients():
    norm = RMSNorm(8)
    input_tensor = torch.randn(2, 3, 8, requires_grad=True)

    loss = norm(input_tensor).sum()
    loss.backward()

    assert input_tensor.grad is not None
    assert norm.weight.grad is not None
    assert torch.isfinite(input_tensor.grad).all()
    assert torch.isfinite(norm.weight.grad).all()


def test_rmsnorm_preserves_bfloat16_dtype_and_matches_float32_reference():
    torch.manual_seed(0)
    norm = RMSNorm(8)
    input_float32 = torch.randn(2, 3, 8)
    input_bfloat16 = input_float32.to(torch.bfloat16)

    expected = norm(input_float32)
    actual = norm(input_bfloat16)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=1e-2,
        atol=1e-2,
    )
