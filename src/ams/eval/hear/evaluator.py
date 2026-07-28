"""Trainer-facing HEAR evaluator adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

from .config import HearEvalConfig, HearEvalResult
from .runner import run_hear_eval

DEFAULT_SCORE_METRICS = (
    "top1_acc",
    "mAP",
    "aucroc",
    "event_onset_200ms_fms",
    "event_onset_50ms_fms",
    "event_onset_offset_50ms_20perc_fms",
)


@dataclass(frozen=True, kw_only=True)
class HearEvaluator:
    """Flatten HEAR task summaries into callback-compatible metrics."""

    config: HearEvalConfig
    metric_prefix: str = "hear"
    score_metrics: tuple[str, ...] = DEFAULT_SCORE_METRICS
    emit_loss: bool = True
    log_metrics: bool = False

    def __call__(self, model: Any, state: Any = None) -> dict[str, float]:
        return self.evaluate(model, state)

    def evaluate(self, model: Any, state: Any = None) -> dict[str, float]:
        """Run HEAR evaluation; ``state`` is accepted for trainer compatibility."""
        del state
        return self.metrics_from_result(run_hear_eval(model, self.config))

    def metrics_from_result(self, result: HearEvalResult) -> dict[str, float]:
        """Extract finite mean summary values and the aggregate score."""
        out: dict[str, float] = {}
        scores: list[float] = []
        for task, payload in result.task_results.items():
            event_probe = payload.get("event_mlp")
            if isinstance(event_probe, dict):
                for metric, stats in event_probe.get("summary", {}).items():
                    mean = stats.get("mean")
                    if isinstance(mean, Real) and math.isfinite(mean):
                        key = f"{self.metric_prefix}/{task}/event_mlp/{metric}"
                        if self.log_metrics or metric in self.score_metrics:
                            out[key] = float(mean)
                        if metric in self.score_metrics:
                            scores.append(float(mean))
            for name, details in payload.get("predictors", {}).items():
                variants = details.items() if name == "knn" else [(name, details)]
                for variant, value in variants:
                    for metric, stats in value.get("summary", {}).items():
                        mean = stats.get("mean")
                        if isinstance(mean, Real) and math.isfinite(mean):
                            key = f"{self.metric_prefix}/{task}/{name}/{variant}/{metric}"
                            if self.log_metrics or metric in {
                                "top1_acc",
                                "mAP",
                                "probe_accuracy",
                                "event_onset_200ms_fms",
                                "event_onset_50ms_fms",
                                "event_onset_offset_50ms_20perc_fms",
                            }:
                                out[key] = float(mean)
                            if metric in self.score_metrics:
                                scores.append(float(mean))
        if scores:
            out[f"{self.metric_prefix}/score"] = sum(scores) / len(scores)
            if self.emit_loss:
                out["loss"] = -out[f"{self.metric_prefix}/score"]
        return out
