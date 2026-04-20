"""Load `.env` from repository root once (no manual export needed)."""

from __future__ import annotations

from pathlib import Path


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # browser_agent/agent/env_bootstrap.py -> repo root
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)
