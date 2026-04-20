from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class RuntimeDebugWriter:
    def __init__(self, enabled: bool = True, root: str | None = None) -> None:
        self.enabled = enabled
        base = root or os.getenv("AGENT_DEBUG_DIR", "debug_runs")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(base) / f"run-{stamp}"
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def step_file(self, step: int, suffix: str) -> Path:
        return self.run_dir / f"step-{step:03d}-{suffix}"

    def write_json(self, step: int, suffix: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self.step_file(step, suffix)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def write_text(self, step: int, suffix: str, content: str) -> None:
        if not self.enabled:
            return
        path = self.step_file(step, suffix)
        path.write_text(content, encoding="utf-8")
