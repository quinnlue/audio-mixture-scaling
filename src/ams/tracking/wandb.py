"""Rank-zero W&B callback with a stable optimizer-step axis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from ams.config import TrackingConfig, to_dict


class WandBTracking:
    def __init__(self, config: TrackingConfig, run_config: Any, *, enabled: bool) -> None:
        self.config = config
        self.run_config = run_config
        self.enabled = enabled and config.wandb_mode != "disabled"
        self.run: Any = None

    def on_start(self, state: Any) -> None:
        if not self.enabled:
            return
        import wandb

        self.run = wandb.init(
            project=self.config.wandb_project,
            id=self.config.wandb_run_id,
            resume="allow",
            mode=cast(
                Literal["online", "offline", "disabled", "shared"],
                self.config.wandb_mode,
            ),
            config=to_dict(self.run_config),
            name=self.config.trial_id,
            tags=["eat", "audio-ssl", "mixture-scaling"],
        )
        wandb.define_metric("optimizer_step")
        wandb.define_metric("*", step_metric="optimizer_step")

    def on_log(self, state: Any, metrics: dict[str, float]) -> None:
        if self.run is not None:
            self.run.log({"optimizer_step": state.optimizer_step, **metrics})

    def on_checkpoint(self, state: Any, path: Path) -> None:
        pass

    def on_validation(self, state: Any, metrics: dict[str, float]) -> None:
        self.on_log(state, metrics)

    def on_end(self, state: Any) -> None:
        if self.run is not None:
            self.run.summary.update(
                {
                    "optimizer_step": state.optimizer_step,
                    "last_loss": state.last_loss,
                    **state.last_val_metrics,
                }
            )
            self.run.finish()
            self.run = None

    def on_failure(self, state: Any, error: BaseException) -> None:
        if self.run is not None:
            self.run.finish(exit_code=1)
            self.run = None


__all__ = ["WandBTracking"]
