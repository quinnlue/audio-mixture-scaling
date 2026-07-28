"""Named HEAR evaluation protocols."""

from typing import Any

from .config import HearEvalConfig

FAST_PRESET: dict[str, Any] = {
    "max_folds": 1,
    "probe_epochs": 10,
    "ks": (10,),
    "run_knn": True,
    "run_linear": True,
}
FULL_PRESET: dict[str, Any] = {
    "max_folds": None,
    "probe_epochs": 20,
    "ks": (1, 5, 10),
    "run_knn": True,
    "run_linear": True,
}
PRESETS = {"fast": FAST_PRESET, "full": FULL_PRESET}


def build_config(name: str, **overrides: Any) -> HearEvalConfig:
    """Combine a named protocol with explicit non-null overrides."""
    if name not in PRESETS:
        raise KeyError(f"Unknown HEAR preset {name!r}")
    return HearEvalConfig(
        **(
            dict(PRESETS[name])
            | {key: value for key, value in overrides.items() if value is not None}
        )
    )
