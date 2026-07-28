import torch
import torch.nn.functional as F

from ams.config import UFOConfig
from ams.objective.targets import make_targets


def test_targets_without_norm_are_an_average() -> None:
    layers = (torch.randn(2, 5, 4), torch.randn(2, 5, 4), torch.randn(2, 5, 4))
    cfg = UFOConfig(
        average_top_k_layers=2, instance_norm_target_layer=False, layer_norm_targets=False
    )
    assert torch.equal(make_targets(layers, cfg), torch.stack(layers[-2:]).mean(0))


def test_instance_norm_target_layer_normalizes_each_channel() -> None:
    layers = tuple(torch.randn(2, 7, 4) * 5 + 3 for _ in range(2))
    cfg = UFOConfig(
        average_top_k_layers=1, instance_norm_target_layer=True, layer_norm_targets=False
    )
    target = make_targets(layers, cfg)
    assert torch.allclose(target.mean(dim=1), torch.zeros_like(target.mean(dim=1)), atol=2e-5)
    assert torch.allclose(
        target.var(dim=1, unbiased=False),
        torch.ones_like(target.var(dim=1, unbiased=False)),
        atol=2e-4,
    )


def test_published_eat_target_normalization_order() -> None:
    layers = tuple(torch.randn(2, 7, 4) * 3 + index for index in range(3))
    config = UFOConfig(
        average_top_k_layers=2,
        instance_norm_target_layer=True,
        layer_norm_targets=True,
    )
    expected_layers = [
        F.instance_norm(layer.transpose(1, 2)).transpose(1, 2) for layer in layers[-2:]
    ]
    expected = F.layer_norm(torch.stack(expected_layers).mean(0), (layers[0].shape[-1],))
    torch.testing.assert_close(make_targets(layers, config), expected, rtol=0, atol=0)
