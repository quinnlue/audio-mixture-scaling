from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from ams.config import (
    CheckpointConfig,
    DataConfig,
    LoopConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    UFOConfig,
)
from ams.data.collate import WaveformBatch
from ams.distributed import DistributedContext
from ams.seeding import seed_everything
from ams.train import Trainer


class _Dataset(Dataset[torch.Tensor]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([float(index), 1.0])


def _collate(items: list[torch.Tensor]) -> WaveformBatch:
    values = torch.stack(items)
    return WaveformBatch(values, torch.ones_like(values, dtype=torch.bool))


class _Output:
    def __init__(self, loss: torch.Tensor) -> None:
        self.loss = loss


class _Objective(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.generator = torch.Generator().manual_seed(99)
        self.register_buffer("_ema_num_updates", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> _Output:
        del mask
        noise = torch.rand((), device=x.device, generator=self.generator) * 0.01
        return _Output(self.model(x).square().mean() + self.bias.square() + noise)

    def update_teacher(self) -> None:
        self._ema_num_updates += 1


def _config(max_epochs: int, *, save_every_n_steps: int | None = None) -> RunConfig:
    return RunConfig(
        model=ModelConfig(d_model=12, depth=1, num_heads=3),
        ufo=UFOConfig(average_top_k_layers=1),
        data=DataConfig(
            dataset_dir="unused",
            batch_size=2,
            num_workers=0,
            persistent_workers=False,
            prefetch_factor=None,
        ),
        optim=OptimConfig(lr=1e-2, warmup_steps=0, total_steps=4),
        loop=LoopConfig(
            max_epochs=max_epochs,
            precision="float32",
            log_every_n_steps=100,
        ),
        checkpoint=CheckpointConfig(
            save_every_n_epochs=1,
            save_every_n_steps=save_every_n_steps,
        ),
    )


def _run(
    output_dir,
    max_epochs: int,
    resume=None,
    *,
    save_every_n_steps: int | None = None,
    callbacks=None,
) -> _Objective:
    loader = DataLoader(
        _Dataset(),
        batch_size=2,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(1234),
    )
    model = torch.nn.Linear(2, 1)
    objective = _Objective(model)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    Trainer(
        model=model,
        objective=objective,
        train_loader=loader,
        config=_config(max_epochs, save_every_n_steps=save_every_n_steps),
        context=context,
        output_dir=output_dir,
        resume=resume,
        callbacks=callbacks,
    ).fit()
    return objective


def test_epoch_resume_matches_uninterrupted(tmp_path) -> None:
    seed_everything(17)
    uninterrupted = _run(tmp_path / "full", 2)
    seed_everything(17)
    _run(tmp_path / "split", 1)
    checkpoint = tmp_path / "split" / "checkpoints" / "epoch_0001.pt"
    resumed = _run(tmp_path / "split", 2, checkpoint)
    for left, right in zip(
        uninterrupted.state_dict().values(),
        resumed.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(left, right)


class _StopAtStep:
    def __init__(self, optimizer_step: int) -> None:
        self.optimizer_step = optimizer_step

    def on_start(self, state) -> None:
        pass

    def on_log(self, state, metrics) -> None:
        pass

    def on_checkpoint(self, state, path) -> None:
        if state.optimizer_step == self.optimizer_step and path.name.startswith("step_"):
            raise RuntimeError("simulated preemption")

    def on_end(self, state) -> None:
        pass

    def on_failure(self, state, error) -> None:
        pass


def test_mid_epoch_resume_matches_uninterrupted(tmp_path) -> None:
    seed_everything(23)
    uninterrupted = _run(tmp_path / "full-mid", 2, save_every_n_steps=1)
    seed_everything(23)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        _run(
            tmp_path / "split-mid",
            2,
            save_every_n_steps=1,
            callbacks=[_StopAtStep(1)],
        )
    checkpoint = tmp_path / "split-mid" / "checkpoints" / "step_00000001.pt"
    resumed = _run(
        tmp_path / "split-mid",
        2,
        checkpoint,
        save_every_n_steps=1,
    )
    for left, right in zip(
        uninterrupted.state_dict().values(),
        resumed.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(left, right)
