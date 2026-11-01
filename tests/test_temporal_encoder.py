"""Unit tests for the Dilated Residual Temporal Convolutional Network (TCN)."""

import torch

from vad.models.temporal_encoder import TemporalConvNet


def test_tcn_output_shapes_and_receptive_field() -> None:
    tcn = TemporalConvNet(
        in_dim=64,
        hidden_dim=128,
        num_layers=4,
        kernel_size=3,
        dilation_base=2,
        dropout=0.1,
        causal=False,
    )
    # R = 1 + 2 * (3 - 1) * (1 + 2 + 4 + 8) = 61
    assert tcn.receptive_field == 61

    x = torch.randn(2, 15, 64)
    out = tcn(x)
    assert out.shape == (2, 15, 128)
    assert torch.isfinite(out).all()


def test_tcn_variable_length_masking() -> None:
    tcn = TemporalConvNet(in_dim=32, hidden_dim=64, num_layers=3, kernel_size=3)
    x = torch.randn(2, 12, 32)
    mask = torch.ones((2, 12), dtype=torch.bool)
    mask[0, 8:] = False
    mask[1, 10:] = False

    out = tcn(x, mask=mask)
    assert out.shape == (2, 12, 64)
    # Padded positions must be strictly zeroed out
    assert torch.all(out[0, 8:] == 0.0)
    assert torch.all(out[1, 10:] == 0.0)
    assert torch.any(out[0, :8] != 0.0)


def test_tcn_causal_mode_prevents_future_leakage() -> None:
    """Verify that in causal mode, changing future tokens at t > t_0 never alters outputs at t <= t_0."""
    torch.manual_seed(42)
    tcn_causal = TemporalConvNet(
        in_dim=16,
        hidden_dim=32,
        num_layers=3,
        kernel_size=3,
        dilation_base=2,
        dropout=0.0,
        causal=True,
    )
    tcn_causal.eval()

    x1 = torch.randn(1, 10, 16)
    x2 = x1.clone()
    # Perturb future frames at indices 5..9
    x2[:, 5:, :] = torch.randn(1, 5, 16) * 50.0

    with torch.no_grad():
        out1 = tcn_causal(x1)
        out2 = tcn_causal(x2)

    # Outputs at timesteps 0..4 must be identical up to floating point precision
    assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-6)
    # Outputs at timesteps 5..9 should differ
    assert not torch.allclose(out1[:, 5:, :], out2[:, 5:, :], atol=1e-3)
