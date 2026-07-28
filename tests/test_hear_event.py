"""Regression coverage for the HEAR frame-level event path."""

from __future__ import annotations

import io
import json

import numpy as np
import soundfile as sf
import torch

from ams.eval.hear import HearEvalConfig, run_hear_eval
from ams.eval.hear.archive import write_task_archive


class _Frames(torch.nn.Module):
    def forward(self, waveforms: torch.Tensor, padding: torch.Tensor):
        return type(
            "Output",
            (),
            {"z": waveforms[:, ::200].unsqueeze(-1), "mask": padding[:, ::200]},
        )()


def _wav(value: float) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.full(800, value, dtype=np.float32), 16_000, format="WAV")
    return buffer.getvalue()


def test_event_archive_runs_frame_probe(tmp_path):
    metadata = {
        "task_name": "tiny_events",
        "embedding_type": "event",
        "prediction_type": "multilabel",
        "split_mode": "new_split",
        "splits": ["train", "valid", "test"],
        "evaluation": ["event_onset_200ms_fms", "event_onset_50ms_fms"],
        "evaluation_params": {
            "event_postprocessing_grid": {"median_filter_ms": [1], "min_duration": [1]}
        },
    }
    labels = json.dumps([{"label": 0, "start": 0, "end": 50}])
    write_task_archive(
        tmp_path / "tiny_events.parquet",
        clip_ids=["train", "valid", "test"],
        splits=["train", "valid", "test"],
        value_jsons=[labels, labels, labels],
        num_frames=[800, 800, 800],
        sample_rates=[16_000, 16_000, 16_000],
        audio=[_wav(0.1), _wav(0.2), _wav(0.3)],
        task_metadata=metadata,
        vocab_pairs=[[0, "event"]],
    )

    config = HearEvalConfig(
        data_root=tmp_path,
        results_dir=tmp_path / "results",
        device="cpu",
        event_epochs=1,
        event_hidden_dim=4,
        event_frame_batch_size=8,
        run_knn=False,
        run_linear=False,
    )
    result = run_hear_eval(_Frames(), config)

    task = result.task_results["tiny_events"]
    fold = task["event_mlp"]["folds"][0]
    assert task["embedding_type"] == "event"
    assert fold["frame_train_size"] == 4
    assert set(fold["metrics"]) >= {"event_mlp_loss", "event_onset_200ms_fms"}
    assert run_hear_eval(_Frames(), config).task_results["tiny_events"]["cache_hit"]
