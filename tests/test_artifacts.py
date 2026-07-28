from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from ams.tracking.artifacts import LocalArtifacts, sha256


def test_runner_artifact_contract(tmp_path) -> None:
    model = torch.nn.Linear(2, 2)
    state = SimpleNamespace(
        model=model,
        epoch=0,
        completed_epoch=0,
        batch_in_epoch=0,
        optimizer_step=3,
        world_size=1,
        config_fingerprint="full-fingerprint",
        resume_fingerprint="resume-fingerprint",
        last_loss=0.5,
        last_val_metrics={},
    )
    metadata = {"campaign_id": "campaign", "trial_id": "trial", "dist_id": 2}
    artifacts = LocalArtifacts(tmp_path, metadata)
    artifacts.on_start(state)
    checkpoint = tmp_path / "checkpoints" / "epoch_0001.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"value": 1}, checkpoint)
    artifacts.on_checkpoint(state, checkpoint)
    artifacts.on_end(state)

    resume = tmp_path / "checkpoints" / "resume.pt"
    resume_metadata = json.loads(
        (tmp_path / "checkpoints" / "resume.json").read_text(encoding="utf-8")
    )
    assert resume_metadata["sha256"] == sha256(resume)
    assert resume_metadata["world_size"] == 1
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "complete"
    rows = (tmp_path / "results" / "hear_metrics.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["inventory_only"] is True
    assert (tmp_path / "exports" / "step_00000003" / "model.safetensors").is_file()
