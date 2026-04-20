from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


ToolName = Literal[
    "navigate",
    "click",
    "type",
    "scroll",
    "wait",
    "back",
    "finish_task",
]


@dataclass
class ToolAction:
    tool: ToolName
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class Observation:
    url: str
    title: str
    dom_summary: str
    available_actions: List[str]
    step: int
    page_structure: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanStep:
    thought: str
    action: ToolAction


@dataclass
class PlanResult:
    steps: List[PlanStep]
    done: bool = False
    final_response: Optional[str] = None


@dataclass
class ExecutionResult:
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
