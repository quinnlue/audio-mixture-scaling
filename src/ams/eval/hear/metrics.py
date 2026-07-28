"""Metrics used by HEAR scene and event task runners."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import EventLabel


def top1_accuracy(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float((scores.argmax(1) == y_true).mean())


def _hot(y: np.ndarray, classes: int) -> np.ndarray:
    return np.eye(classes, dtype=np.int8)[y]


def classification_metrics(
    y_true: np.ndarray, scores: np.ndarray, num_classes: int
) -> dict[str, float]:
    """Compute HEAR-compatible scene classification metrics."""
    output = {"top1_acc": top1_accuracy(y_true, scores)}
    targets = _hot(y_true, num_classes)
    try:
        output["mAP"] = float(average_precision_score(targets, scores, average="macro"))
    except ValueError:
        output["mAP"] = float("nan")
    try:
        auc = float(roc_auc_score(targets, scores, average="macro", multi_class="ovr"))
        output["aucroc"] = auc
        output["d_prime"] = math.sqrt(2) * __import__("statistics").NormalDist().inv_cdf(auc)
    except ValueError:
        output["aucroc"] = output["d_prime"] = float("nan")
    return output


def finite_float_dict(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: (
            float(value)
            if isinstance(value, (float, np.floating)) and math.isfinite(value)
            else None
            if isinstance(value, (float, np.floating))
            else value
        )
        for key, value in payload.items()
    }


def events_to_frame_targets(
    events: tuple[EventLabel, ...], timestamps_ms: np.ndarray, *, num_classes: int
) -> np.ndarray:
    """Rasterize event intervals into frame-wise multi-label targets."""
    targets = np.zeros((len(timestamps_ms), num_classes), dtype=np.float32)
    if timestamps_ms.size == 0:
        return targets
    frame_ms = float(np.median(np.diff(timestamps_ms))) if timestamps_ms.size > 1 else 1.0
    starts = timestamps_ms
    ends = timestamps_ms + max(frame_ms, 1e-6)
    for event in events:
        targets[(starts < event.end_ms) & (ends > event.start_ms), event.label] = 1.0
    return targets


def _median_filter(active: np.ndarray, window_frames: int) -> np.ndarray:
    if window_frames <= 1:
        return active
    if window_frames % 2 == 0:
        window_frames += 1
    pad = window_frames // 2
    padded = np.pad(active.astype(np.int32), ((pad, pad), (0, 0)), mode="edge")
    cumulative = np.vstack((np.zeros((1, active.shape[1]), dtype=np.int32), padded.cumsum(axis=0)))
    window_sum = cumulative[window_frames:] - cumulative[:-window_frames]
    return window_sum >= (window_frames // 2 + 1)


def postprocess_event_probs(
    probabilities: np.ndarray,
    timestamps_ms: np.ndarray,
    *,
    threshold: float = 0.5,
    median_filter_ms: float = 200.0,
    min_duration_ms: float = 100.0,
) -> list[EventLabel]:
    """Threshold frame probabilities, smooth them, and emit contiguous events."""
    if probabilities.ndim != 2:
        raise ValueError("Event probabilities must have shape (frames, classes)")
    if probabilities.shape[0] != timestamps_ms.shape[0]:
        raise ValueError("Timestamp count must equal event-probability frame count")
    if not len(timestamps_ms):
        return []
    frame_ms = float(np.median(np.diff(timestamps_ms))) if len(timestamps_ms) > 1 else 1.0
    active = _median_filter(
        probabilities >= threshold,
        max(int(round(median_filter_ms / max(frame_ms, 1e-6))), 1),
    )
    events: list[EventLabel] = []
    for label in range(active.shape[1]):
        start: int | None = None
        for frame, is_active in enumerate(active[:, label].tolist() + [False]):
            if is_active and start is None:
                start = frame
            elif not is_active and start is not None:
                start_ms = float(timestamps_ms[start])
                end_ms = float(timestamps_ms[frame - 1] + frame_ms)
                if end_ms - start_ms >= min_duration_ms:
                    events.append(
                        EventLabel(
                            label=label,
                            label_raw=label,
                            label_text=str(label),
                            start_ms=start_ms,
                            end_ms=end_ms,
                        )
                    )
                start = None
    return events


def _event_counts(
    reference: tuple[EventLabel, ...],
    prediction: list[EventLabel],
    *,
    onset_tolerance_ms: float,
    offset_tolerance_ms: float | None = None,
    offset_tolerance_ratio: float | None = None,
) -> tuple[int, int, int]:
    references: dict[int, list[EventLabel]] = defaultdict(list)
    predictions: dict[int, list[EventLabel]] = defaultdict(list)
    for event in reference:
        references[event.label].append(event)
    for event in prediction:
        predictions[event.label].append(event)
    true_positive = false_positive = false_negative = 0
    for label in references.keys() | predictions.keys():
        used: set[int] = set()
        refs = references[label]
        for predicted in predictions[label]:
            candidates = [
                (abs(predicted.start_ms - ref.start_ms), index, ref)
                for index, ref in enumerate(refs)
                if index not in used
                and abs(predicted.start_ms - ref.start_ms) <= onset_tolerance_ms
            ]
            if offset_tolerance_ms is not None:
                candidates = [
                    (distance, index, ref)
                    for distance, index, ref in candidates
                    if abs(predicted.end_ms - ref.end_ms)
                    <= max(
                        offset_tolerance_ms,
                        (offset_tolerance_ratio or 0.0) * (ref.end_ms - ref.start_ms),
                    )
                ]
            if candidates:
                _, index, _ = min(candidates)
                used.add(index)
                true_positive += 1
            else:
                false_positive += 1
        false_negative += len(refs) - len(used)
    return true_positive, false_positive, false_negative


def _event_fms(
    references: list[tuple[EventLabel, ...]],
    predictions: list[list[EventLabel]],
    *,
    onset_tolerance_ms: float,
    offset_tolerance_ms: float | None = None,
    offset_tolerance_ratio: float | None = None,
) -> float:
    true_positive = false_positive = false_negative = 0
    for reference, prediction in zip(references, predictions, strict=True):
        tp, fp, fn = _event_counts(
            reference,
            prediction,
            onset_tolerance_ms=onset_tolerance_ms,
            offset_tolerance_ms=offset_tolerance_ms,
            offset_tolerance_ratio=offset_tolerance_ratio,
        )
        true_positive += tp
        false_positive += fp
        false_negative += fn
    denominator = (2 * true_positive) + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def _segment_1s_error_rate(
    references: list[tuple[EventLabel, ...]],
    predictions: list[list[EventLabel]],
    durations_ms: list[float],
    *,
    num_classes: int,
) -> float:
    deletions = insertions = total_references = 0
    for reference, prediction, duration_ms in zip(
        references, predictions, durations_ms, strict=True
    ):
        segments = max(int(math.ceil(duration_ms / 1000.0)), 1)
        reference_counts = np.zeros((segments, num_classes), dtype=np.int64)
        prediction_counts = np.zeros((segments, num_classes), dtype=np.int64)
        for counts, events in ((reference_counts, reference), (prediction_counts, prediction)):
            for event in events:
                start = max(int(math.floor(event.start_ms / 1000.0)), 0)
                end = min(int(math.floor(max(event.end_ms - 1e-6, 0.0) / 1000.0)), segments - 1)
                counts[start : end + 1, event.label] += 1
        common = np.minimum(reference_counts, prediction_counts).sum()
        deletions += max(int(reference_counts.sum() - common), 0)
        insertions += max(int(prediction_counts.sum() - common), 0)
        total_references += int(reference_counts.sum())
    return (deletions + insertions) / max(total_references, 1)


def event_metrics(
    references: list[tuple[EventLabel, ...]],
    predictions: list[list[EventLabel]],
    *,
    metrics: tuple[str, ...],
    durations_ms: list[float] | None = None,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Score the HEAR event-detection metrics requested by task metadata."""
    output: dict[str, float] = {}
    if "event_onset_200ms_fms" in metrics:
        output["event_onset_200ms_fms"] = _event_fms(
            references, predictions, onset_tolerance_ms=200.0
        )
    if "event_onset_50ms_fms" in metrics:
        output["event_onset_50ms_fms"] = _event_fms(
            references, predictions, onset_tolerance_ms=50.0
        )
    if "event_onset_offset_50ms_20perc_fms" in metrics:
        output["event_onset_offset_50ms_20perc_fms"] = _event_fms(
            references,
            predictions,
            onset_tolerance_ms=50.0,
            offset_tolerance_ms=50.0,
            offset_tolerance_ratio=0.2,
        )
    if "segment_1s_er" in metrics:
        if durations_ms is None or num_classes is None:
            raise ValueError("segment_1s_er requires clip durations and a class count")
        output["segment_1s_er"] = _segment_1s_error_rate(
            references, predictions, durations_ms, num_classes=num_classes
        )
    return output
