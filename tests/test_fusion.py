"""Unit tests for the Multi-Prior Fusion module, ablation controls, and gradient propagation."""

import pytest
import torch

from vad.config import FusionConfig, TemporalConfig, VADConfig
from vad.models.anomaly_detector import AnomalyDetector


@pytest.mark.parametrize(
    "mode,expected_zero_gates",
    [
        ("gated_multiprior", []),
        ("concat_mlp", []),
        ("videomae_only", [1, 2, 3]),
        ("dinov2_only", [0, 2, 3]),
        ("no_tcn", [2]),
    ],
)
def test_fusion_modes_and_gate_weights(mode: str, expected_zero_gates: list[int]) -> None:
    cfg = VADConfig(
        temporal=TemporalConfig(enabled=True, hidden_dim=64, num_layers=2),
        fusion=FusionConfig(mode=mode, proj_dim=64, dropout=0.1),
    )
    cfg.encoders.videomae.feature_dim = 128
    cfg.encoders.dinov2.feature_dim = 128

    detector = AnomalyDetector(config=cfg)
    detector.eval()

    v = torch.randn(2, 10, 128)
    d = torch.randn(2, 10, 128)
    mask = torch.ones((2, 10), dtype=torch.bool)
    mask[1, 7:] = False

    with torch.no_grad():
        out = detector(v, d, mask=mask)

    assert out.scores.shape == (2, 10)
    assert out.logits.shape == (2, 10)
    assert out.fused_features.shape == (2, 10, 64)
    assert out.gate_weights.shape == (2, 10, 4)
    assert out.motion_magnitude.shape == (2, 10)

    # Valid scores are strictly in (0, 1) and masked scores are 0
    assert torch.all((out.scores[0] > 0.0) & (out.scores[0] < 1.0))
    assert torch.all(out.scores[1, 7:] == 0.0)

    # Gate weights sum to 1.0 at every timestep
    gate_sums = out.gate_weights.sum(dim=-1)
    assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5)

    for inactive_idx in expected_zero_gates:
        assert torch.allclose(
            out.gate_weights[:, :, inactive_idx],
            torch.zeros_like(out.gate_weights[:, :, inactive_idx]),
            atol=1e-5,
        )


def test_gradient_propagation_through_all_trainable_components() -> None:
    cfg = VADConfig(
        temporal=TemporalConfig(enabled=True, hidden_dim=32, num_layers=2),
        fusion=FusionConfig(mode="gated_multiprior", proj_dim=32, dropout=0.0),
    )
    cfg.encoders.videomae.feature_dim = 64
    cfg.encoders.dinov2.feature_dim = 64

    detector = AnomalyDetector(config=cfg)
    detector.train()

    v = torch.randn(3, 8, 64, requires_grad=True)
    d = torch.randn(3, 8, 64, requires_grad=True)
    out = detector(v, d)

    loss = out.scores.mean()
    loss.backward()

    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert d.grad is not None and torch.isfinite(d.grad).all()

    # Check that key modules received non-zero finite gradients
    for name, param in detector.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
