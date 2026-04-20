from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from .llm import LLMClient
from .memory import MemoryStore
from .tools import Observation, ToolAction


SafetyVerdict = Literal["allow", "confirm", "block"]


@dataclass
class SafetyDecision:
    verdict: SafetyVerdict
    rationale: str
    source: str = "model"
    route: str = "llm_review"


class SafetyEvaluator:
    """Two-stage safety guard: cheap local classifier + optional LLM review."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def evaluate(
        self,
        *,
        task: str,
        observation: Observation,
        action: ToolAction,
        memory: MemoryStore,
    ) -> SafetyDecision:
        _ = memory
        local = self._classify_local_risk(action)
        if local == "low_risk":
            return SafetyDecision(
                verdict="allow",
                rationale="Low-risk UI interaction by local structural classifier; LLM review skipped.",
                source="local_classifier",
                route="local_low_risk",
            )

        if local == "high_risk":
            return SafetyDecision(
                verdict="confirm",
                rationale="Potentially state-changing or irreversible action by local structural classifier.",
                source="local_classifier",
                route="local_high_risk",
            )

        compact_payload = self._build_compact_payload(task=task, observation=observation, action=action)
        system = (
            "You are a safety evaluator for browser automation actions.\n"
            "You are called only for ambiguous actions after a local risk gate.\n"
            "Use general action semantics and UI context, not site-specific assumptions.\n"
            "Classify into exactly one verdict:\n"
            "- allow: low-risk, reversible, or read-only interaction.\n"
            "- confirm: possibly state-changing, final, external, or hard to undo.\n"
            "- block: clearly unsafe, policy-violating, unrelated to user task, or likely harmful.\n"
            "Return ONLY JSON with this schema:\n"
            '{"verdict":"allow|confirm|block","rationale":"short reason"}'
        )

        try:
            raw = self._llm.chat_text(
                system,
                json.dumps(compact_payload, ensure_ascii=True),
                temperature=0.0,
                max_tokens=int(os.getenv("LM_SAFETY_MAX_TOKENS", "120")),
            )
            parsed = self._safe_parse_json(raw)
            verdict = str(parsed.get("verdict", "")).strip().lower()
            rationale = str(parsed.get("rationale", "")).strip()
            if verdict in {"allow", "confirm", "block"} and rationale:
                return SafetyDecision(
                    verdict=verdict,
                    rationale=rationale,
                    source="llm_compact",
                    route="llm_review",
                )
        except Exception as exc:  # noqa: BLE001
            return SafetyDecision(
                verdict="confirm",
                rationale=f"Ambiguous action and safety model unavailable: {exc}",
                source="fallback_conservative",
                route="llm_review",
            )

        return SafetyDecision(
            verdict="confirm",
            rationale="Safety model returned invalid output; using conservative generic fallback.",
            source="fallback_conservative",
            route="llm_review",
        )

    @staticmethod
    def _safe_parse_json(raw: str) -> dict:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        try:
            loaded = json.loads(text)
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _classify_local_risk(self, action: ToolAction) -> Literal["low_risk", "needs_llm_review", "high_risk"]:
        tool = str(action.tool or "").lower()
        args = action.args if isinstance(action.args, dict) else {}
        role = str(args.get("role", "")).strip().lower()
        reason = str(action.reason or "").strip().lower()
        name = str(args.get("name", "")).strip().lower()
        text = str(args.get("text", "")).strip()

        if tool in {"scroll", "wait", "back", "navigate"}:
            return "low_risk"
        if tool == "finish_task":
            return "high_risk"

        # Lightweight semantic hints as secondary signals (generic, not domain-specific).
        low_intent = self._intent_score(name + " " + reason, kind="low")
        high_intent = self._intent_score(name + " " + reason, kind="high")

        # Structural low-risk roles: navigation and disclosure UI controls.
        if tool == "click" and role in {"combobox", "tab", "menuitem", "textbox", "searchbox", "option"}:
            return "low_risk"
        if tool == "click" and role in {"button", "link"} and high_intent == 0:
            return "low_risk"
        if tool == "click" and role in {"button", "link"} and low_intent >= 1:
            return "low_risk"
        if tool == "type" and role in {"searchbox", "combobox"}:
            return "low_risk"
        if tool == "type" and role == "textbox" and len(text) <= 80 and low_intent >= 1:
            return "low_risk"

        if tool == "type":
            # Writing long free text into generic textboxes is ambiguous and may be state-changing.
            if role == "textbox" and len(text) > 120:
                return "needs_llm_review"
            if low_intent >= 2:
                return "low_risk"
            if high_intent >= 2:
                return "needs_llm_review"
            return "needs_llm_review"

        if tool == "click":
            if high_intent >= 2:
                return "high_risk"
            if low_intent >= 2:
                return "low_risk"
            if role in {"link"}:
                return "low_risk"
            return "needs_llm_review"

        return "needs_llm_review"

    @staticmethod
    def _intent_score(text: str, *, kind: Literal["low", "high"]) -> int:
        tokens = re.findall(r"[a-zA-Zа-яА-Я0-9_]+", text.lower())
        if kind == "low":
            cues = {
                "search",
                "filter",
                "sort",
                "open",
                "expand",
                "show",
                "next",
                "previous",
                "menu",
                "tab",
                "select",
                "поиск",
                "найти",
                "открыть",
                "показать",
                "фильтр",
            }
        else:
            cues = {
                "submit",
                "confirm",
                "approve",
                "send",
                "publish",
                "delete",
                "remove",
                "checkout",
                "purchase",
                "pay",
                "order",
                "save",
                "apply",
                "create",
                "final",
                "удалить",
                "оплат",
                "подтверд",
                "отправ",
                "примен",
                "создат",
            }
        return sum(1 for token in tokens if token in cues)

    def _build_compact_payload(self, *, task: str, observation: Observation, action: ToolAction) -> dict[str, Any]:
        args = action.args if isinstance(action.args, dict) else {}
        role = str(args.get("role", "")).strip()
        name = str(args.get("name", "")).strip()
        page_structure = observation.page_structure if isinstance(observation.page_structure, dict) else {}

        page_summary = str(page_structure.get("summary", "")).strip()
        nearby_hints = self._collect_nearby_hints(observation, role=role, name=name)
        return {
            "task": task[:500],
            "action": {
                "tool": action.tool,
                "reason": str(action.reason or "")[:220],
                "args": {
                    "role": role[:60],
                    "name": name[:180],
                    "text": str(args.get("text", ""))[:220],
                    "url": str(args.get("url", ""))[:240],
                },
            },
            "page": {
                "url": observation.url[:220],
                "title": observation.title[:200],
                "summary": page_summary[:260],
                "hints": nearby_hints[:3],
            },
        }

    @staticmethod
    def _collect_nearby_hints(observation: Observation, *, role: str, name: str) -> list[str]:
        hints: list[str] = []
        available = observation.available_actions if isinstance(observation.available_actions, list) else []
        if role and name:
            target = f"{role}:{name}".lower()
            for item in available[:25]:
                text = str(item).strip()
                if text and target in text.lower():
                    hints.append(text[:180])
                    break
        for item in available[:8]:
            text = str(item).strip()
            if text and text[:180] not in hints:
                hints.append(text[:180])
            if len(hints) >= 3:
                break
        return hints
