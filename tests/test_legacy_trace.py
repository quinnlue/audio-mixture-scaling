from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ams.config import ModelConfig, OptimConfig, UFOConfig
from ams.model import EATModel
from ams.objective import UFOObjective
from ams.seeding import seed_everything
from ams.train.optim import build_optimizer, build_scheduler


def test_normalization_disabled_matches_recorded_legacy_trace() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "legacy_proxy_loss_trace.json").read_text()
    )
    seed_everything(0)
    model = EATModel(
        ModelConfig(
            d_model=24,
            depth=2,
            num_heads=3,
            target_length=32,
            max_time_patches=2,
            decoder_depth=1,
        )
    )
    seed_everything(0)
    objective = UFOObjective(
        model,
        UFOConfig(
            num_masks=1,
            average_top_k_layers=2,
            instance_norm_target_layer=False,
            batch_norm_target_layer=False,
            layer_norm_target_layer=False,
            layer_norm_targets=False,
            instance_norm_targets=False,
            ema_decay=0.999,
            ema_end_decay=0.999,
            ema_anneal_steps=0,
            loss_beta=1.0,
        ),
        generator=torch.Generator().manual_seed(7),
    )
    optim_config = OptimConfig(
        lr=5e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.05,
        warmup_steps=2,
        total_steps=20,
        min_lr_scale=0.002,
    )
    optimizer = build_optimizer(objective, optim_config)
    scheduler = build_scheduler(optimizer, optim_config, 20)
    waveforms = torch.randn(2, 4_000, generator=torch.Generator().manual_seed(123))
    mask = torch.ones_like(waveforms, dtype=torch.bool)

    observed = []
    for _ in range(20):
        output = objective(waveforms, mask)
        observed.append(float(output.loss.detach()))
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        objective.update_teacher()

    assert observed == pytest.approx(fixture["losses"], abs=1e-5, rel=0)
