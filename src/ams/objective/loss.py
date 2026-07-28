"""Loss helpers for UFO."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ufo_frame_loss(
    prediction: torch.Tensor, target: torch.Tensor, removal_mask: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    return F.mse_loss(prediction, target, reduction="none").mean(-1)[removal_mask].mean()


def ufo_utterance_loss(student_cls: torch.Tensor, teacher_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_cls, teacher_target)


__all__ = ["ufo_frame_loss", "ufo_utterance_loss"]
