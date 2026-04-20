from __future__ import annotations

from .llm import LLMClient
from .tools import ToolAction


class ErrorReflector:
    """Small sub-agent: suggests recovery after a failed browser action (error adaptation)."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def reflect(
        self,
        *,
        task: str,
        observation_excerpt: str,
        action: ToolAction,
        error_message: str,
    ) -> str:
        system = (
            "You are a debugging assistant for a browser automation agent. "
            "Given a failed action, propose ONE concise recovery hint (max 3 sentences). "
            "Focus on alternative roles/names, waiting for load, scrolling, or navigating back. "
            "Do not suggest CSS/XPath selectors."
        )
        user = (
            f"Task: {task}\n"
            f"Failed tool: {action.tool} args={action.args} reason={action.reason}\n"
            f"Error: {error_message}\n"
            f"Observation excerpt:\n{observation_excerpt[:6000]}\n"
        )
        try:
            return self._llm.chat_text(system, user, temperature=0.15)
        except Exception as exc:  # noqa: BLE001
            return f"(reflector unavailable: {exc})"
