"""Torch linear scene probe."""

import random

import numpy as np
import torch


def train_linear_probe_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    num_classes: int,
    epochs: int,
    batch_size: int,
    device: str | torch.device,
    seed: int,
    standardize: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit a deterministic linear softmax classifier and return probabilities."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    if standardize:
        mean, std = x_train.mean(0), x_train.std(0)
        x_train, x_test = (
            (x_train - mean) / np.maximum(std, 1e-6),
            (x_test - mean) / np.maximum(std, 1e-6),
        )
    model = torch.nn.Linear(x_train.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    features, labels = (
        torch.as_tensor(x_train, device=device),
        torch.as_tensor(y_train, device=device),
    )
    for _ in range(epochs):
        for index in torch.randperm(len(features), device=device).split(batch_size):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(features[index]), labels[index])
            loss.backward()
            opt.step()
    with torch.no_grad():
        logits = model(torch.as_tensor(x_test, device=device))
        scores = logits.softmax(1).cpu().numpy()
        test_loss = torch.nn.functional.cross_entropy(
            logits, torch.as_tensor(y_test, device=device)
        ).item()
    return scores.astype(np.float32), {
        "accuracy": float((scores.argmax(1) == y_test).mean()),
        "loss": float(test_loss),
    }
