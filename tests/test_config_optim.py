from __future__ import annotations

from dataclasses import replace

import pytest

from ams.config import dump_config, fingerprint, load_config, resume_fingerprint
from ams.train.optim import lr_scale


def test_yaml_roundtrip_override_typing_and_fingerprint(tmp_path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text(
        """
model:
  d_model: 144
  depth: 4
  num_heads: 3
ufo:
  average_top_k_layers: 4
data:
  format: parquet
  dataset_dir: /data/test
  num_workers: 0
  persistent_workers: false
  prefetch_factor: null
""",
        encoding="utf-8",
    )
    config = load_config(
        [source],
        ["optim.lr=3e-4", "loop.max_epochs=2", "runtime.deterministic=true"],
    )
    assert config.optim.lr == pytest.approx(3e-4)
    assert config.loop.max_epochs == 2
    assert config.runtime.deterministic is True
    assert isinstance(config.model.patch_size, tuple)
    resolved = tmp_path / "resolved.yaml"
    dump_config(config, resolved)
    roundtrip = load_config([resolved])
    assert roundtrip == config
    assert fingerprint(roundtrip) == fingerprint(config)
    assert resume_fingerprint(replace(config, loop=replace(config.loop, max_epochs=9))) == (
        resume_fingerprint(config)
    )


def test_unknown_override_is_rejected(tmp_path) -> None:
    source = tmp_path / "config.yaml"
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config field"):
        load_config([source], ["not_a_section.value=1"])


def test_warmup_cosine_anchor_values() -> None:
    assert lr_scale(0, warmup_steps=3, total_steps=10, minimum=0.002) == 0
    assert lr_scale(2, warmup_steps=3, total_steps=10, minimum=0.002) == pytest.approx(2 / 3)
    assert lr_scale(3, warmup_steps=3, total_steps=10, minimum=0.002) == 1
    assert 0.002 < lr_scale(9, warmup_steps=3, total_steps=10, minimum=0.002) < 0.1
    assert lr_scale(10, warmup_steps=3, total_steps=10, minimum=0.002) == pytest.approx(0.002)
