"""Pack legacy HEAR task trees into self-contained Parquet archives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _metadata(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be an object: {path}")
    return value


def discover_tasks(root: Path) -> list[tuple[Path, dict]]:
    if not root.exists():
        raise FileNotFoundError(root)
    found: dict[str, tuple[Path, dict]] = {}
    for path in sorted(root.glob("*/task_metadata.json")):
        meta = _metadata(path)
        name = str(meta["task_name"])
        previous = found.get(name)
        # Prefer non-HEAR2021 duplicate archives, matching the established harness.
        if previous is None or (
            "hear2021" in previous[0].name.lower() and "hear2021" not in path.parent.name.lower()
        ):
            found[name] = (path.parent, meta)
    return [found[name] for name in sorted(found)]


def _vocab(root: Path) -> list[list[str | int]]:
    path = root / "labelvocabulary.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fp:
        result = []
        for row in csv.DictReader(fp):
            raw = row["label"]
            key: str | int
            try:
                key = int(raw)
            except ValueError:
                key = raw
            result.append([key, raw])
        return result


def convert_task(root: Path, metadata: dict, out_dir: Path) -> Path:
    # Kept lazy so prep remains usable without HEAR's optional dependencies until invoked.
    from ams.eval.hear.archive import write_task_archive

    split_names = (
        tuple(map(str, metadata.get("splits", ())))
        or tuple(f"fold{i:02d}" for i in range(int(metadata["nfolds"])))
        if metadata.get("nfolds") is not None
        else ("train", "valid", "test")
    )
    import soundfile as sf

    ids = []
    splits = []
    values = []
    frames = []
    rates = []
    audio = []
    for split in split_names:
        for wav_name, value in json.loads(
            (root / f"{split}.json").read_text(encoding="utf-8")
        ).items():
            flac = root / "16000" / split / f"{Path(wav_name).stem}.flac"
            if not flac.exists():
                raise FileNotFoundError(flac)
            info = sf.info(flac)
            ids.append(wav_name)
            splits.append(split)
            values.append(json.dumps(value))
            frames.append(int(info.frames))
            rates.append(int(info.samplerate))
            audio.append(flac.read_bytes())
    destination = out_dir / f"{metadata['task_name']}.parquet"
    write_task_archive(
        destination,
        clip_ids=ids,
        splits=splits,
        value_jsons=values,
        num_frames=frames,
        sample_rates=rates,
        audio=audio,
        task_metadata=metadata,
        vocab_pairs=_vocab(root),
    )
    return destination


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/HEAR", type=Path)
    p.add_argument("--out-dir", default="data/hear_parquet", type=Path)
    p.add_argument("--tasks")
    a = p.parse_args()
    tasks = discover_tasks(a.data_root)
    if a.tasks:
        requested = {s.strip() for s in a.tasks.split(",") if s.strip()}
        tasks = [item for item in tasks if str(item[1]["task_name"]) in requested]
        missing = requested - {str(m["task_name"]) for _, m in tasks}
        if missing:
            raise SystemExit(f"Unknown task(s): {sorted(missing)}")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for root, meta in tasks:
        print(convert_task(root, meta, a.out_dir))


if __name__ == "__main__":
    main()
