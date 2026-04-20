from __future__ import annotations

import os
from typing import Any, Dict, List

from .llm import LLMClient
from .memory import MemoryStore
from .tools import PlanResult, PlanStep, ToolAction


class Planner:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.system_prompt = (
            "You are a browser task planner.\n"
            "Return ONLY valid JSON (no markdown fences, no extra keys).\n"
            "\n"
            "CRITICAL PLANNING MODE: SINGLE-STEP ONLY\n"
            "- Return EXACTLY ONE next step in steps (steps must contain exactly 1 item).\n"
            "- If you considered multiple steps, keep only the first immediate executable one.\n"
            "- Do NOT plan the whole workflow.\n"
            "- Do NOT output multiple future actions.\n"
            "- If unsure, choose the safest useful next step (usually scroll/wait/back/navigate).\n"
            "- AVOID repeated blind scroll loops. If prior scrolls did not change page state, switch strategy.\n"
            "- Do NOT assume future page states.\n"
            "- Do NOT reference buttons/links/fields that are not present in the current observation.\n"
            "- For search-like tasks, prefer search UI interactions (type/click on search controls) before more scrolling.\n"
            "\n"
            "DATA ENTRY RULES\n"
            "- Never type fabricated, placeholder, or guessed values (no fake phone numbers, emails, codes).\n"
            "- Only type text explicitly derived from the user task or directly visible page context.\n"
            "- If required information is missing, ask the user via a safe step (e.g., finish_task with a request), "
            "or navigate/scroll to locate the needed info first.\n"
            "\n"
            "Allowed tools: navigate, click, type, scroll, wait, back, finish_task.\n"
            "Element selection rule: select ONLY by ARIA role + accessible name.\n"
            "Never use CSS, XPath, ids, classes.\n"
            "\n"
            "Action schema (must match validator):\n"
            '{\n'
            '  "done": boolean,\n'
            '  "final_response": string | null,\n'
            '  "steps": [\n'
            "    {\n"
            '      "thought": string,\n'
            '      "action": {\n'
            '        "tool": "navigate|click|type|scroll|wait|back|finish_task",\n'
            '        "args": object,\n'
            '        "reason": string\n'
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "\n"
            "Tool args requirements:\n"
            "- navigate: args={\"url\": string}\n"
            "- click: args={\"role\": string, \"name\": string, \"exact\"?: boolean}\n"
            "- type: args={\"role\": string, \"name\": string, \"text\": string, \"exact\"?: boolean}\n"
            "- scroll: args={\"direction\"?: \"up\"|\"down\", \"amount\"?: number}\n"
            "- wait: args={\"ms\"?: number}\n"
            "- back: args={}\n"
            "- finish_task: args={\"message\"?: string}\n"
        )

    def plan(self, task: str, observation_text: str, memory: MemoryStore, planning_hints: str = "") -> PlanResult:
        user_prompt = (
            f"User task:\n{task}\n\n"
            f"Current observation:\n{observation_text}\n\n"
            f"Recent memory:\n{memory.to_prompt_compact()}\n\n"
            f"Planner hints:\n{planning_hints or 'none'}\n\n"
            "Return exactly ONE next action step."
        )
        payload = self.llm.chat_json(
            self.system_prompt,
            user_prompt,
            max_tokens=int(os.getenv("LM_PLANNER_MAX_TOKENS", "220")),
        )
        return self._to_plan_result(payload)

    def _to_plan_result(self, payload: Dict[str, Any]) -> PlanResult:
        steps_raw: List[Dict[str, Any]] = payload.get("steps", [])
        steps: List[PlanStep] = []
        for step in steps_raw:
            action_raw = step.get("action", {})
            tool = action_raw.get("tool", "wait")
            args = action_raw.get("args", {})
            reason = action_raw.get("reason", "")
            action = ToolAction(tool=tool, args=args, reason=reason)
            steps.append(PlanStep(thought=step.get("thought", ""), action=action))

        return PlanResult(
            steps=steps,
            done=bool(payload.get("done", False)),
            final_response=payload.get("final_response"),
        )
