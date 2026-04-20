from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha1
from typing import Deque, Optional

from .tools import Observation, ToolAction


@dataclass
class StagnationSignal:
    is_stuck: bool
    reason: str
    repeated_action_count: int
    same_url_count: int
    same_state_count: int


@dataclass
class _StepState:
    action_signature: str
    url: str
    state_signature: str
    affordance_signature: str


class LoopMonitor:
    def __init__(self, window: int = 6) -> None:
        self.window = max(window, 4)
        self._history: Deque[_StepState] = deque(maxlen=self.window)

    def record(self, action: ToolAction, observation: Observation) -> StagnationSignal:
        state = _StepState(
            action_signature=_action_sig(action),
            url=observation.url,
            state_signature=_state_sig(observation),
            affordance_signature=_affordance_sig(observation),
        )
        self._history.append(state)
        return self._detect()

    def _detect(self) -> StagnationSignal:
        if len(self._history) < 4:
            return StagnationSignal(False, "insufficient_history", 0, 0, 0)

        last = self._history[-1]
        repeated_action = _count_tail(self._history, lambda s: s.action_signature == last.action_signature)
        same_url = _count_tail(self._history, lambda s: s.url == last.url)
        same_state = _count_tail(self._history, lambda s: s.state_signature == last.state_signature)
        same_affordance = _count_tail(self._history, lambda s: s.affordance_signature == last.affordance_signature)

        action_is_scroll = last.action_signature.startswith("scroll:")
        repeated_same_target = repeated_action >= 3 and same_url >= 3 and (same_state >= 2 or same_affordance >= 3)
        stuck = repeated_same_target and (
            action_is_scroll
            or last.action_signature.startswith("type:")
            or last.action_signature.startswith("click:")
        )
        if stuck:
            return StagnationSignal(
                is_stuck=True,
                reason=(
                    f"action_loop: sig={last.action_signature}; repeated_action={repeated_action}, same_url={same_url}, "
                    f"same_state={same_state}, same_affordance={same_affordance}"
                ),
                repeated_action_count=repeated_action,
                same_url_count=same_url,
                same_state_count=same_state,
            )
        return StagnationSignal(
            is_stuck=False,
            reason="moving_or_not_enough_stagnation_signals",
            repeated_action_count=repeated_action,
            same_url_count=same_url,
            same_state_count=same_state,
        )


def _action_sig(action: ToolAction) -> str:
    if action.tool == "scroll":
        return f"scroll:{action.args.get('direction', 'down')}:{action.args.get('amount', 1200)}"
    if action.tool in {"click", "type"}:
        return f"{action.tool}:{action.args.get('role', '')}:{action.args.get('name', '')}"
    return action.tool


def _state_sig(obs: Observation) -> str:
    structure = obs.page_structure or {}
    head = tuple((structure.get("headings") or [])[:8])
    buttons = tuple((structure.get("buttons") or [])[:10])
    links = tuple((structure.get("links") or [])[:10])
    inputs = tuple((structure.get("inputs") or [])[:8])
    title = obs.title or ""
    blob = f"{title}|{head}|{buttons}|{links}|{inputs}"
    return sha1(blob.encode("utf-8")).hexdigest()[:14]


def _affordance_sig(obs: Observation) -> str:
    hints = tuple((obs.available_actions or [])[:24])
    return sha1(str(hints).encode("utf-8")).hexdigest()[:14]


def _count_tail(history: Deque[_StepState], predicate) -> int:  # type: ignore[no-untyped-def]
    count = 0
    for item in reversed(history):
        if predicate(item):
            count += 1
        else:
            break
    return count

