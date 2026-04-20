from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote_plus

from .tools import Observation, ToolAction


@dataclass
class StrategyOverride:
    action: ToolAction
    reason: str


_SEARCH_TASK_TOKENS = (
    "find",
    "search",
    "look for",
    "job",
    "vacanc",
    "ваканс",
    "работ",
    "найд",
    "документац",
    "docs",
    "товар",
    "product",
    "mail",
    "почт",
)
_SEARCH_UI_TOKENS = ("search", "find", "поиск", "искать", "найти", "query")


def is_search_like_task(task: str) -> bool:
    lowered = (task or "").lower()
    return any(token in lowered for token in _SEARCH_TASK_TOKENS)


def detect_search_ui(observation: Observation) -> bool:
    for hint in observation.available_actions or []:
        role, _, name = hint.partition(":")
        role = role.strip().lower()
        name = name.strip().lower()
        if role in {"searchbox", "textbox", "combobox"}:
            return True
        if role in {"button", "link"} and any(token in name for token in _SEARCH_UI_TOKENS):
            return True
    return False


def choose_stuck_override(task: str, observation: Observation, normalized_query: str | None = None) -> StrategyOverride:
    query = (normalized_query or "").strip() or _query_from_task(task)

    # 1) Prefer typing into search-capable input controls.
    for role in ("searchbox", "textbox", "combobox"):
        name = _best_input_name(observation, role)
        if name:
            return StrategyOverride(
                action=ToolAction(
                    tool="type",
                    args={"role": role, "name": name, "text": query},
                    reason="stagnation recovery: use visible search-like input",
                ),
                reason=f"type into {role}:{name}",
            )

    # 2) Click search trigger affordance when visible.
    for role in ("button", "link"):
        name = _best_search_trigger_name(observation, role)
        if name:
            return StrategyOverride(
                action=ToolAction(
                    tool="click",
                    args={"role": role, "name": name},
                    reason="stagnation recovery: open search flow",
                ),
                reason=f"click {role}:{name}",
            )

    # 3) Generic fallback: external search route refinement.
    google = f"https://www.google.com/search?q={quote_plus(query)}"
    return StrategyOverride(
        action=ToolAction(
            tool="navigate",
            args={"url": google},
            reason="stagnation recovery: route refinement via search engine",
        ),
        reason="navigate to search fallback",
    )


def _query_from_task(task: str) -> str:
    compact = re.sub(r"\s+", " ", (task or "").strip())
    if len(compact) <= 110:
        return compact
    return compact[:107] + "..."


def _best_input_name(observation: Observation, target_role: str) -> str | None:
    candidates: list[str] = []
    for hint in observation.available_actions or []:
        role, _, name = hint.partition(":")
        if role.strip().lower() != target_role:
            continue
        clean = name.strip()
        if clean:
            candidates.append(clean)
    if not candidates:
        return None
    for name in candidates:
        lowered = name.lower()
        if any(bad in lowered for bad in ("исключить", "exclude", "от", "до", "salary", "зарплат", "регион", "город")):
            continue
        if any(token in lowered for token in _SEARCH_UI_TOKENS):
            return name
    return candidates[0] if candidates else None


def _best_search_trigger_name(observation: Observation, target_role: str) -> str | None:
    for hint in observation.available_actions or []:
        role, _, name = hint.partition(":")
        if role.strip().lower() != target_role:
            continue
        clean = name.strip()
        lowered = clean.lower()
        if clean and any(token in lowered for token in _SEARCH_UI_TOKENS):
            return clean
    return None

