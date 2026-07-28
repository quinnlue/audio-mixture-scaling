"""Cosine weighted k-nearest-neighbour scene probe."""

import numpy as np


def knn_predict_scores(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, num_classes: int, k: int
) -> np.ndarray:
    """Return class scores from cosine-weighted neighbours."""
    train = x_train / np.maximum(np.linalg.norm(x_train, axis=1, keepdims=True), 1e-12)
    test = x_test / np.maximum(np.linalg.norm(x_test, axis=1, keepdims=True), 1e-12)
    order = np.argpartition(-(test @ train.T), min(k, len(train)) - 1, axis=1)[
        :, : min(k, len(train))
    ]
    scores = np.zeros((len(test), num_classes), dtype=np.float32)
    for row, neighbours in enumerate(order):
        scores[row] = np.bincount(
            y_train[neighbours],
            weights=np.maximum((test[row] @ train[neighbours].T), 0),
            minlength=num_classes,
        )
    return scores
