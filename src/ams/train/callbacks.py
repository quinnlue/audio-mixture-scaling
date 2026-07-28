"""Minimal callback protocol used by the single training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Callback(Protocol):
    def on_start(self, state: Any) -> None: ...

    def on_log(self, state: Any, metrics: dict[str, float]) -> None: ...

    def on_checkpoint(self, state: Any, path: Path) -> None: ...

    def on_validation(self, state: Any, metrics: dict[str, float]) -> None: ...

    def on_end(self, state: Any) -> None: ...

    def on_failure(self, state: Any, error: BaseException) -> None: ...


class NoOpCallback:
    def on_start(self, state: Any) -> None:
        pass

    def on_log(self, state: Any, metrics: dict[str, float]) -> None:
        pass

    def on_checkpoint(self, state: Any, path: Path) -> None:
        pass

    def on_validation(self, state: Any, metrics: dict[str, float]) -> None:
        pass

    def on_end(self, state: Any) -> None:
        pass

    def on_failure(self, state: Any, error: BaseException) -> None:
        pass


__all__ = ["Callback", "NoOpCallback"]
