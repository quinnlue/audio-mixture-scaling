"""Background export uploader for standalone runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class HubUpload:
    def __init__(
        self,
        output_dir: str | Path,
        repo_id: str | None,
        *,
        private: bool = True,
        enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.repo_id = repo_id
        self.private = private
        self.enabled = enabled and repo_id is not None
        self.scheduler: Any = None

    def on_start(self, state: Any) -> None:
        if not self.enabled:
            return
        from huggingface_hub import CommitScheduler, create_repo

        repo_id = self.repo_id
        if repo_id is None:
            raise RuntimeError("Hub upload enabled without a repository id")
        create_repo(repo_id, private=self.private, exist_ok=True)
        self.scheduler = CommitScheduler(
            repo_id=repo_id,
            folder_path=self.output_dir / "exports",
            path_in_repo="exports",
            every=10,
        )

    def on_log(self, state: Any, metrics: dict[str, float]) -> None:
        pass

    def on_checkpoint(self, state: Any, path: Path) -> None:
        pass

    def on_validation(self, state: Any, metrics: dict[str, float]) -> None:
        pass

    def on_end(self, state: Any) -> None:
        if self.scheduler is not None:
            self.scheduler.trigger().result()

    def on_failure(self, state: Any, error: BaseException) -> None:
        pass


__all__ = ["HubUpload"]
