import torch

from ams.config import ModelConfig, UFOConfig
from ams.model import EATModel
from ams.objective import UFOObjective


def test_eat_and_ufo_small_forward() -> None:
    model = EATModel(
        ModelConfig(d_model=24, depth=2, num_heads=3, target_length=32, max_time_patches=2)
    )
    objective = UFOObjective(
        model,
        UFOConfig(num_masks=2, average_top_k_layers=2),
        generator=torch.Generator().manual_seed(0),
    )
    output = objective(torch.randn(2, 1000))
    assert output.loss.ndim == 0
    assert output.prediction.shape == (4, 16, 24)
    output.loss.backward()


def test_ema_schedule_endpoints_and_counter_roundtrip() -> None:
    config = ModelConfig(
        d_model=24,
        depth=1,
        num_heads=3,
        target_length=32,
        max_time_patches=2,
    )
    objective = UFOObjective(
        EATModel(config),
        UFOConfig(
            average_top_k_layers=1,
            ema_decay=0.9,
            ema_end_decay=0.99,
            ema_anneal_steps=10,
        ),
    )
    assert objective._current_ema_decay() == 0.9
    objective._ema_num_updates.fill_(10)
    assert objective._current_ema_decay() == 0.99
    state = objective.state_dict()
    restored = UFOObjective(
        EATModel(config),
        UFOConfig(
            average_top_k_layers=1,
            ema_decay=0.9,
            ema_end_decay=0.99,
            ema_anneal_steps=10,
        ),
    )
    restored.load_state_dict(state)
    assert int(restored._ema_num_updates) == 10
