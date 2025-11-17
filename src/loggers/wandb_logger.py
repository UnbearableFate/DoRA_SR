from __future__ import annotations

import os
from contextlib import AbstractContextManager
from typing import Any, Dict, Optional

from src.config import WandbConfig

try:
    import wandb
except Exception:  # pragma: no cover - wandb may not be installed in CI
    wandb = None


class WandbSession(AbstractContextManager):
    """Context manager that safely initializes and tears down a Weights & Biases run."""

    def __init__(self, cfg: WandbConfig, config_payload: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg
        self.config_payload = config_payload or {}
        self.run = None

    def __enter__(self):  # type: ignore[override]
        if not self.cfg.enabled or wandb is None:
            os.environ["WANDB_DISABLED"] = "true"
            return None

        os.environ["WANDB_MODE"] = self.cfg.mode
        self.run = wandb.init(
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.cfg.run_name,
            tags=self.cfg.tags,
            config=self.config_payload,
        )
        return self.run

    def __exit__(self, exc_type, exc, exc_tb):  # type: ignore[override]
        if self.run is not None:
            self.run.finish()
        self.run = None
        return False


__all__ = ["WandbSession"]
