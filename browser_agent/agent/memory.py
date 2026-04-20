from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MemoryEvent:
    step: int
    kind: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    def __init__(self, max_events: int = 100) -> None:
        self.max_events = max_events
        self.events: List[MemoryEvent] = []
        self.current_task: str = ""
        self.completed_steps: List[str] = []
        self.visited_pages: List[str] = []
        self.recent_errors: List[str] = []
        self.action_history: List[Dict[str, Any]] = []

    def add(self, event: MemoryEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

    def add_text(self, step: int, kind: str, content: str, **metadata: Any) -> None:
        self.add(MemoryEvent(step=step, kind=kind, content=content, metadata=metadata))

    def start_task(self, task: str) -> None:
        self.current_task = task

    def record_page_visit(self, url: str) -> None:
        if not url:
            return
        if not self.visited_pages or self.visited_pages[-1] != url:
            self.visited_pages.append(url)
        self.visited_pages = self.visited_pages[-30:]

    def record_action(self, step: int, action: str, reason: str, success: bool, message: str) -> None:
        self.record_action_with_progress(
            step=step,
            action=action,
            reason=reason,
            success=success,
            message=message,
            useful_progress=success and action != "finish_task",
        )

    def record_action_with_progress(
        self,
        step: int,
        action: str,
        reason: str,
        success: bool,
        message: str,
        useful_progress: bool,
    ) -> None:
        self.action_history.append(
            {
                "step": step,
                "action": action,
                "reason": reason,
                "success": success,
                "message": message,
                "useful_progress": useful_progress,
            }
        )
        self.action_history = self.action_history[-40:]
        if useful_progress:
            self.completed_steps.append(f"step {step}: {action}")
            self.completed_steps = self.completed_steps[-30:]
        if not success:
            self.recent_errors.append(f"step {step}: {message}")
            self.recent_errors = self.recent_errors[-10:]

    def progress_summary(self) -> str:
        useful = sum(1 for item in self.action_history if item.get("useful_progress"))
        noops = sum(1 for item in self.action_history if item.get("success") and not item.get("useful_progress"))
        return (
            f"task={self.current_task or 'n/a'}; "
            f"useful_completed={useful}; "
            f"noop_success={noops}; "
            f"visited={len(self.visited_pages)}; "
            f"errors={len(self.recent_errors)}"
        )

    def recent(self, n: int = 12) -> List[MemoryEvent]:
        return self.events[-n:]

    def to_prompt(self, n: int = 12) -> str:
        base_lines = [
            f"current_task: {self.current_task or 'n/a'}",
            f"progress: {self.progress_summary()}",
            f"completed_steps: {self.completed_steps[-8:] or []}",
            f"visited_pages: {self.visited_pages[-8:] or []}",
            f"recent_errors: {self.recent_errors[-5:] or []}",
            f"action_history: {self.action_history[-8:] or []}",
        ]
        if not self.events:
            return "\n".join(base_lines + ["events: none"])
        lines: List[str] = []
        for ev in self.recent(n):
            meta = f" | meta={ev.metadata}" if ev.metadata else ""
            lines.append(f"[step={ev.step}] {ev.kind}: {ev.content}{meta}")
        return "\n".join(base_lines + ["events:"] + lines)

    def to_prompt_compact(self, n_events: int = 6, n_actions: int = 6) -> str:
        action_lines: List[str] = []
        for item in self.action_history[-n_actions:]:
            action_lines.append(
                f"step={item.get('step')} action={item.get('action')} success={item.get('success')} "
                f"useful={item.get('useful_progress')} reason={_clip(str(item.get('reason', '')), 80)}"
            )
        event_lines: List[str] = []
        for ev in self.recent(n_events):
            event_lines.append(
                f"[{ev.step}] {ev.kind}: {_clip(ev.content, 100)}"
            )
        lines = [
            f"task={_clip(self.current_task or 'n/a', 120)}",
            f"progress={self.progress_summary()}",
            f"visited_recent={self.visited_pages[-5:] or []}",
            f"errors_recent={self.recent_errors[-3:] or []}",
            f"actions_recent={json.dumps(action_lines, ensure_ascii=True)}",
        ]
        if event_lines:
            lines.append(f"events_recent={json.dumps(event_lines, ensure_ascii=True)}")
        return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
