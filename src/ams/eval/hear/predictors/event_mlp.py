"""Frame-level multi-label probe used by HEAR event tasks."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, kw_only=True)
class EventMLPConfig:
    """Training settings for a small frame-wise event classifier."""

    input_dim: int
    num_classes: int
    hidden_dim: int = 256
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 65_536
    seed: int = 0
    device: str | torch.device = "cpu"


class EventMLP(nn.Module):
    """Two-layer MLP predicting independent class activity per frame."""

    def __init__(self, config: EventMLPConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_event_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test_by_clip: list[np.ndarray],
    *,
    config: EventMLPConfig,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Fit a frame-wise BCE probe and return per-clip probabilities."""
    if x_train.ndim != 2 or y_train.ndim != 2:
        raise ValueError("Event probe features and targets must both be rank-2")
    if x_train.shape[0] == 0:
        raise ValueError("Cannot train an event probe with zero frames")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Event probe features and targets must have the same number of frames")

    _seed(config.seed)
    device = torch.device(config.device)
    model = EventMLP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)
    loss_fn = nn.BCEWithLogitsLoss()
    features = torch.from_numpy(x_train.astype(np.float32, copy=False)).to(device)
    targets = torch.from_numpy(y_train.astype(np.float32, copy=False)).to(device)
    batch_size = max(int(config.batch_size), 1)
    last_loss = 0.0

    for _ in range(config.epochs):
        model.train()
        permutation = torch.randperm(features.shape[0], device=device)
        total_loss = 0.0
        total_frames = 0
        for indices in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features[indices]), targets[indices])
            loss.backward()
            optimizer.step()
            n_frames = int(indices.shape[0])
            total_loss += float(loss.detach()) * n_frames
            total_frames += n_frames
        last_loss = total_loss / max(total_frames, 1)

    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for test_features in x_test_by_clip:
            if test_features.shape[0] == 0:
                probabilities.append(np.zeros((0, config.num_classes), dtype=np.float32))
                continue
            rows = []
            for start in range(0, test_features.shape[0], batch_size):
                batch_features = test_features[start : start + batch_size].astype(np.float32)
                batch = torch.from_numpy(batch_features).to(device)
                rows.append(torch.sigmoid(model(batch)).cpu().numpy())
            probabilities.append(np.concatenate(rows, axis=0).astype(np.float32, copy=False))
    return probabilities, {"loss": float(last_loss)}


__all__ = ["EventMLP", "EventMLPConfig", "train_event_mlp"]
