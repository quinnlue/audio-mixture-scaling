"""EAT utterance-frame objective with an EMA teacher."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import torch

from ams.config import UFOConfig
from ams.model.eat import EATModel

from .decoder import UFODecoder
from .loss import ufo_frame_loss, ufo_utterance_loss
from .masking import inverse_block_masking, restore_patch_tokens
from .targets import make_targets


@dataclass
class UFOOutput:
    loss: torch.Tensor
    frame_loss: torch.Tensor
    utterance_loss: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    frame_mask: torch.Tensor
    student_cls: torch.Tensor
    teacher_utterance_target: torch.Tensor


class UFOObjective(torch.nn.Module):
    """Train an EAT model against normalized EMA teacher patch targets."""

    def __init__(
        self, model: EATModel, cfg: UFOConfig, *, generator: torch.Generator | None = None
    ) -> None:
        super().__init__()
        self.model, self.cfg, self.generator = model, cfg, generator
        self.mask_token = torch.nn.Parameter(torch.empty(model.cfg.d_model))
        torch.nn.init.normal_(
            self.mask_token, std=float(getattr(cfg, "mask_token_init_scale", 0.02))
        )
        self.decoder = UFODecoder(model.cfg, cfg)
        self.teacher_model = deepcopy(model)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("_ema_num_updates", torch.zeros((), dtype=torch.long), persistent=True)

    def _teacher_targets(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tokens, layout = self.teacher_model.tokenize(features)
            embedded = self.teacher_model.embed(tokens, layout)
            encoded = self.teacher_model.encode_embedded(embedded, return_hidden_states=True)
            assert encoded.hidden_states is not None
            return make_targets(encoded.hidden_states, self.cfg)

    def forward(
        self, waveforms: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> UFOOutput:
        features = self.model.process(waveforms, padding_mask)
        tokens, layout = self.model.tokenize(features)
        embedded = self.model.embed(tokens, layout)
        targets = self._teacher_targets(features)
        patch_targets = targets[:, 1:]
        utterance_targets = patch_targets.mean(1)
        num_masks = int(getattr(self.cfg, "num_masks", 1))
        if num_masks > 1:
            embedded = embedded.repeat_interleave(num_masks, 0)
            patch_targets = patch_targets.repeat_interleave(num_masks, 0)
            utterance_targets = utterance_targets.repeat_interleave(num_masks, 0)
        cls, patches = embedded[:, :1], embedded[:, 1:]
        _, removal, ids_restore, ids_keep = inverse_block_masking(
            batch_size=patches.shape[0],
            time_patches=layout.time_patches,
            frequency_patches=layout.frequency_patches,
            mask_ratio=float(getattr(self.cfg, "mask_ratio", 0.8)),
            block_size=tuple(getattr(self.cfg, "block_size", (2, 2))),
            device=patches.device,
            generator=self.generator,
        )
        kept = torch.gather(patches, 1, ids_keep.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
        student = self.model.encode_embedded(torch.cat([cls, kept], 1))
        prediction = self.decoder(
            restore_patch_tokens(
                student.z[:, 1:], mask_token=self.mask_token, ids_restore=ids_restore
            ),
            layout,
        )
        frame_loss = ufo_frame_loss(prediction, patch_targets.detach(), removal)
        utterance_loss = ufo_utterance_loss(student.cls, utterance_targets.detach())
        loss = frame_loss + float(getattr(self.cfg, "loss_beta", 1.0)) * utterance_loss
        return UFOOutput(
            loss,
            frame_loss,
            utterance_loss,
            prediction,
            patch_targets.detach(),
            removal,
            student.cls,
            utterance_targets.detach(),
        )

    def _current_ema_decay(self) -> float:
        start = float(getattr(self.cfg, "ema_decay", 0.9998))
        end = getattr(self.cfg, "ema_end_decay", None)
        steps = int(getattr(self.cfg, "ema_anneal_steps", 0))
        if end is None or steps <= 0:
            return start
        updates = self._ema_num_updates
        assert isinstance(updates, torch.Tensor)
        step = int(updates.item())
        return float(end) if step >= steps else start + (float(end) - start) * (step / steps)

    @torch.no_grad()
    def update_teacher(self) -> None:
        decay = self._current_ema_decay()
        teacher = cast(list[torch.Tensor], list(self.teacher_model.parameters()))
        student = cast(list[torch.Tensor], list(self.model.parameters()))
        torch._foreach_mul_(teacher, decay)
        torch._foreach_add_(teacher, student, alpha=1.0 - decay)
        teacher_buffers, student_buffers = (
            list(self.teacher_model.buffers()),
            list(self.model.buffers()),
        )
        if teacher_buffers:
            torch._foreach_copy_(teacher_buffers, student_buffers)
        self.teacher_model.eval()
        updates = self._ema_num_updates
        assert isinstance(updates, torch.Tensor)
        updates.add_(1)


__all__ = ["UFOObjective", "UFOOutput"]
