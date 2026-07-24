import torch

from modelforge.multi_head_attention import MHA
from modelforge.sdpa_attention import SDPAMHA


def test_sdpa_attention_matches_manual_attention():
    torch.manual_seed(0)

    manual = MHA(hidden_dim=8, num_heads=2, max_seq_len=6)
    optimized = SDPAMHA(hidden_dim=8, num_heads=2, max_seq_len=6)

    optimized.load_state_dict(manual.state_dict())

    input_tensor = torch.randn(2, 5, 8)

    expected = manual(input_tensor)
    actual = optimized(input_tensor)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)