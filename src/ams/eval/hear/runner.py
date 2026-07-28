"""HEAR protocol runner."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import torch

from .config import HearEvalConfig, HearEvalResult
from .cv import iter_splits
from .data import build_manifest
from .extract import extract_frames, extract_scene, resolve_device
from .metrics import (
    classification_metrics,
    event_metrics,
    events_to_frame_targets,
    finite_float_dict,
    postprocess_event_probs,
)
from .predictors import (
    EventMLPConfig,
    knn_predict_scores,
    train_event_mlp,
    train_linear_probe_scores,
)
from .tasks import TaskSpec, discover_tasks


def _summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    keys = {key for row in rows for key in row}
    return {
        key: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        if (
            values := [
                float(row[key])
                for row in rows
                if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
            ]
        )
        else {"mean": None, "std": None}
        for key in sorted(keys)
    }


def _scene(
    task: TaskSpec, model: torch.nn.Module, config: HearEvalConfig, device: torch.device
) -> dict[str, Any]:
    samples, labels = build_manifest(task, limit=config.limit)
    extracted = extract_scene(model, task, samples, labels, config, device=device)
    splits = iter_splits(task, samples, max_folds=config.max_folds)
    predictors: dict[str, Any] = {}
    if config.run_knn:
        knn: dict[str, Any] = {}
        for k in config.ks:
            knn_folds: list[dict[str, Any]] = []
            for split in splits:
                scores = knn_predict_scores(
                    extracted.embeddings[split.train],
                    extracted.labels[split.train],
                    extracted.embeddings[split.test],
                    num_classes=len(labels.index_to_raw),
                    k=k,
                )
                knn_folds.append(
                    {
                        "split": split.name,
                        "metrics": finite_float_dict(
                            classification_metrics(
                                extracted.labels[split.test], scores, len(labels.index_to_raw)
                            )
                        ),
                    }
                )
            knn[f"k{k}"] = {
                "folds": knn_folds,
                "summary": _summary([fold["metrics"] for fold in knn_folds]),
            }
        predictors["knn"] = knn
    if config.run_linear:
        linear_folds: list[dict[str, Any]] = []
        for split in splits:
            scores, probe = train_linear_probe_scores(
                extracted.embeddings[split.train],
                extracted.labels[split.train],
                extracted.embeddings[split.test],
                extracted.labels[split.test],
                num_classes=len(labels.index_to_raw),
                epochs=config.probe_epochs,
                batch_size=config.batch_size,
                device=device,
                seed=config.seed,
            )
            linear_folds.append(
                {
                    "split": split.name,
                    "metrics": finite_float_dict(
                        classification_metrics(
                            extracted.labels[split.test], scores, len(labels.index_to_raw)
                        )
                        | {f"probe_{key}": value for key, value in probe.items()}
                    ),
                }
            )
        predictors["linear"] = {
            "folds": linear_folds,
            "summary": _summary([fold["metrics"] for fold in linear_folds]),
        }
    return {
        "task": task.task_name,
        "embedding_type": task.embedding_type,
        "num_samples": len(samples),
        "num_classes": len(labels.index_to_raw),
        "cache_hit": extracted.cache_hit,
        "predictors": predictors,
    }


def _event_postprocessing(task: TaskSpec) -> tuple[float, float]:
    params = task.metadata.get("evaluation_params", {}).get("event_postprocessing_grid", {})
    median_values = params.get("median_filter_ms", ())
    duration_values = params.get("min_duration", ())
    return (
        float(median_values[0]) if median_values else 200.0,
        float(duration_values[0]) if duration_values else 100.0,
    )


def _event_training_rows(
    frame_targets: list[np.ndarray],
    frames: tuple[Any, ...],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = indices.astype(int).tolist()
    return (
        np.concatenate([frames[index].frames for index in selected], axis=0),
        np.concatenate([frame_targets[index] for index in selected], axis=0),
    )


def _event(
    task: TaskSpec, model: torch.nn.Module, config: HearEvalConfig, device: torch.device
) -> dict[str, Any]:
    samples, labels = build_manifest(task, limit=config.limit)
    extracted = extract_frames(model, task, samples, labels, config, device=device)
    splits = iter_splits(task, samples, max_folds=config.max_folds)
    num_classes = len(labels.index_to_raw)
    frame_targets = [
        events_to_frame_targets(sample.events, clip.timestamps_ms, num_classes=num_classes)
        for sample, clip in zip(samples, extracted.clips, strict=True)
    ]
    median_filter_ms, min_duration_ms = _event_postprocessing(task)
    rows: list[dict[str, Any]] = []
    for split in splits:
        if not len(split.train) or not len(split.test):
            raise ValueError(f"Event task {task.task_name!r} produced an empty train or test split")
        x_train, y_train = _event_training_rows(frame_targets, extracted.clips, split.train)
        test_indices = split.test.astype(int).tolist()
        probabilities, probe = train_event_mlp(
            x_train,
            y_train,
            [extracted.clips[index].frames for index in test_indices],
            config=EventMLPConfig(
                input_dim=x_train.shape[1],
                num_classes=num_classes,
                hidden_dim=config.event_hidden_dim,
                dropout=config.event_dropout,
                lr=config.event_lr,
                epochs=config.event_epochs,
                batch_size=config.event_frame_batch_size,
                seed=config.seed,
                device=device,
            ),
        )
        predictions = [
            postprocess_event_probs(
                probabilities[local_index],
                extracted.clips[sample_index].timestamps_ms,
                median_filter_ms=median_filter_ms,
                min_duration_ms=min_duration_ms,
            )
            for local_index, sample_index in enumerate(test_indices)
        ]
        references = [samples[index].events for index in test_indices]
        durations_ms = [extracted.clips[index].duration_ms for index in test_indices]
        rows.append(
            {
                "split": split.name,
                "train_size": int(len(split.train)),
                "test_size": int(len(split.test)),
                "frame_train_size": int(len(x_train)),
                "metrics": finite_float_dict(
                    event_metrics(
                        references,
                        predictions,
                        metrics=task.metrics,
                        durations_ms=durations_ms,
                        num_classes=num_classes,
                    )
                    | {"event_mlp_loss": probe["loss"]}
                ),
            }
        )
    return {
        "task": task.task_name,
        "embedding_type": task.embedding_type,
        "num_samples": len(samples),
        "num_classes": num_classes,
        "cache_hit": extracted.cache_hit,
        "event_mlp": {"folds": rows, "summary": _summary([row["metrics"] for row in rows])},
    }


def run_hear_eval(model: torch.nn.Module, config: HearEvalConfig) -> HearEvalResult:
    """Run selected HEAR archives and write JSON task artifacts."""
    device = resolve_device(config.device)
    tasks = discover_tasks(config.data_root)
    if config.tasks is not None:
        tasks = [task for task in tasks if task.task_name in set(config.tasks)]
    output = config.results_dir / config.model_id
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for task in tasks:
        payload = (
            _event(task, model, config, device)
            if task.embedding_type == "event"
            else _scene(task, model, config, device)
        )
        results[task.task_name] = payload
        (output / f"{task.task_name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )
    result = HearEvalResult(
        model_id=config.model_id, task_results=results, summary={"tasks": len(results)}
    )
    (output / "summary.json").write_text(
        json.dumps(
            {"config": config.to_dict(), "summary": result.summary, "tasks": results},
            indent=2,
            sort_keys=True,
        )
    )
    return result
