"""HEAR manifests and audio decoding."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf

from .tasks import TaskSpec


@dataclass(frozen=True, kw_only=True)
class EventLabel:
    """One labelled event interval in milliseconds."""

    label: int
    label_raw: str | int
    label_text: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class ArchiveSource:
    archive: Any
    row: int

    def read(self, sample_rate: int) -> np.ndarray:
        return decode_audio(self.archive.audio_bytes(self.row), sample_rate=sample_rate)

    def frame_count(self) -> int:
        return self.archive.num_frames(self.row)

    def sample_rate_hz(self) -> int:
        return self.archive.sample_rate(self.row)

    @property
    def uid(self) -> str:
        return f"{self.archive.split(self.row)}/{self.archive.clip_id(self.row)}"


@dataclass(frozen=True, kw_only=True)
class HearSample:
    index: int
    source: ArchiveSource
    clip_id: str
    split: str
    fold: int | None
    label: int | None
    label_raw: str | int | None
    label_text: str | None
    events: tuple[EventLabel, ...]


@dataclass(frozen=True, kw_only=True)
class LabelMap:
    raw_to_index: dict[str | int, int]
    index_to_raw: tuple[str | int, ...]
    index_to_text: tuple[str, ...]


def _raw(value: Any) -> str | int:
    if isinstance(value, list) and len(value) == 1:
        return _raw(value[0])
    if isinstance(value, bool):
        raise TypeError("Boolean labels are not supported")
    return (
        int(value) if isinstance(value, (int, float)) and float(value).is_integer() else str(value)
    )


def build_manifest(
    task: TaskSpec, *, limit: int | None = None
) -> tuple[list[HearSample], LabelMap]:
    """Load task rows and map their labels to contiguous integer IDs."""
    from .archive import TaskArchive

    archive = TaskArchive.open(task.archive_path)
    rows = [
        row
        for split in task.splits
        for row in range(archive.num_rows)
        if archive.split(row) == split
    ]
    values = [archive.value(row) for row in rows]
    raw_values = (
        [_raw(event["label"]) for value in values for event in value]
        if task.embedding_type == "event"
        else [_raw(value) for value in values]
    )
    ordered = sorted(set(raw_values), key=lambda value: (not isinstance(value, int), str(value)))
    mapping = {value: index for index, value in enumerate(ordered)}
    label_map = LabelMap(
        raw_to_index=mapping,
        index_to_raw=tuple(ordered),
        index_to_text=tuple(archive.vocab.get(value, str(value)) for value in ordered),
    )
    samples: list[HearSample] = []
    count: dict[str, int] = {}
    for row, value in zip(rows, values, strict=True):
        split = archive.split(row)
        if limit is not None and count.get(split, 0) >= limit:
            continue
        count[split] = count.get(split, 0) + 1
        source = ArchiveSource(archive, row)
        fold = int(split[4:]) if split.startswith("fold") else None
        if task.embedding_type == "event":
            events = tuple(
                EventLabel(
                    label=mapping[_raw(e["label"])],
                    label_raw=_raw(e["label"]),
                    label_text=label_map.index_to_text[mapping[_raw(e["label"])]],
                    start_ms=float(e["start"]),
                    end_ms=float(e["end"]),
                )
                for e in value
            )
            label = label_raw = label_text = None
        else:
            label_raw = _raw(value)
            label = mapping[label_raw]
            label_text = label_map.index_to_text[label]
            events = ()
        samples.append(
            HearSample(
                index=len(samples),
                source=source,
                clip_id=archive.clip_id(row),
                split=split,
                fold=fold,
                label=label,
                label_raw=label_raw,
                label_text=label_text,
                events=events,
            )
        )
    return samples, label_map


def decode_audio(data: bytes, *, sample_rate: int) -> np.ndarray:
    """Decode encoded mono audio at the HEAR reference sample rate."""
    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if rate != sample_rate:
        raise ValueError(f"Archive audio is {rate} Hz, expected {sample_rate} Hz")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim != 1 or not len(audio):
        raise ValueError("Archive audio must be non-empty mono")
    return audio


def manifest_fingerprint(samples: list[HearSample], label_map: LabelMap) -> str:
    """Return a stable cache key for task content and label semantics."""
    return hashlib.sha256(
        repr(
            (
                label_map.index_to_raw,
                [(s.source.uid, s.label_raw, s.events, s.source.frame_count()) for s in samples],
            )
        ).encode()
    ).hexdigest()
