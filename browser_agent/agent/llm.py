from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from .env_bootstrap import load_dotenv_if_available

load_dotenv_if_available()


class LLMClient:
    def __init__(
        self,
        model: str = "qwen2.5-7b-instruct",
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: str = "lm-studio",
    ) -> None:
        self.model = os.getenv("LM_MODEL", model)
        self.client = OpenAI(
            base_url=os.getenv("LM_BASE_URL", base_url),
            api_key=os.getenv("LM_API_KEY", api_key),
        )

    def chat_json(self, system_prompt: str, user_prompt: str, *, max_tokens: int | None = None) -> Dict[str, Any]:
        base_messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        retries = int(os.getenv("LM_JSON_RETRIES", "1"))
        last_error = "unknown_error"
        raw = "{}"
        mt = os.getenv("LM_JSON_MAX_TOKENS")
        if max_tokens is None and mt and mt.isdigit():
            max_tokens = int(mt)

        for attempt in range(retries + 1):
            messages = list(base_messages)
            if attempt > 0:
                # Keep repair prompt compact; do not echo the full invalid payload back.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was invalid. "
                            f"Validation error: {last_error}. "
                            "Return ONLY strict JSON by schema, no markdown fences, no extra keys."
                        ),
                    }
                )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                **({"max_tokens": max_tokens} if isinstance(max_tokens, int) else {}),
            )
            raw = response.choices[0].message.content or "{}"
            parsed = self._safe_parse_json(raw)
            validated = self._validate_payload(parsed)
            if validated.get("ok"):
                return validated["payload"]
            if validated.get("payload"):
                # Soft-normalized payload accepted to avoid expensive repair loops.
                return validated["payload"]
            last_error = validated.get("error", "validation_failed")
            if attempt >= retries:
                break

        return {
            "done": False,
            "final_response": None,
            "steps": [
                {
                    "thought": f"Fallback due to invalid model JSON: {last_error}",
                    "action": {
                        "tool": "wait",
                        "args": {"ms": 700},
                        "reason": "Model returned invalid JSON payload",
                    },
                }
            ],
            "error": "invalid_model_json",
            "raw": raw,
        }

    @staticmethod
    def _safe_parse_json(raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"error": "invalid_json", "raw": raw}

    @staticmethod
    def _validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("error") == "invalid_json":
            return {"ok": False, "error": "invalid_json"}
        if not isinstance(payload, dict):
            return {"ok": False, "error": "payload_not_object"}

        if "steps" not in payload or not isinstance(payload.get("steps"), list):
            return {"ok": False, "error": "missing_steps"}

        done = bool(payload.get("done", False))
        steps_any = payload.get("steps", [])
        if not done and len(steps_any) != 1:
            if len(steps_any) > 1 and isinstance(steps_any[0], dict):
                payload["steps"] = [steps_any[0]]
            else:
                return {"ok": False, "error": f"expected_single_step_got_{len(steps_any)}"}

        allowed_tools = {"navigate", "click", "type", "scroll", "wait", "back", "finish_task"}
        arg_requirements = {
            "navigate": {"url"},
            "click": {"role", "name"},
            "type": {"role", "name", "text"},
            "scroll": set(),
            "wait": set(),
            "back": set(),
            "finish_task": set(),
        }
        for idx, step in enumerate(payload["steps"]):
            if not isinstance(step, dict):
                return {"ok": False, "error": f"step_{idx}_not_object"}
            action = step.get("action")
            if not isinstance(action, dict):
                return {"ok": False, "error": f"step_{idx}_missing_action"}
            tool = action.get("tool")
            args = action.get("args", {})
            if tool not in allowed_tools:
                return {"ok": False, "error": f"step_{idx}_invalid_tool"}
            if not isinstance(args, dict):
                return {"ok": False, "error": f"step_{idx}_args_not_object"}

            # Backward/alias compatibility: normalize common model arg shapes to role/name.
            # This is a guardrail; the planner prompt should still prefer role+name.
            if tool == "click":
                if "role" not in args or "name" not in args:
                    if "button" in args and "name" not in args:
                        args = {**args, "role": args.get("role") or "button", "name": args.get("button")}
                    elif "link" in args and "name" not in args:
                        args = {**args, "role": args.get("role") or "link", "name": args.get("link")}
            elif tool == "type":
                if ("role" not in args or "name" not in args) and "field" in args:
                    args = {**args, "role": args.get("role") or "textbox", "name": args.get("field")}
            action["args"] = args

            required = arg_requirements[tool]
            missing = [name for name in required if name not in args]
            if missing:
                return {"ok": False, "error": f"step_{idx}_missing_args:{','.join(missing)}"}

        if "done" in payload and not isinstance(payload.get("done"), bool):
            return {"ok": False, "error": "done_not_bool"}
        if payload.get("final_response") is not None and not isinstance(payload.get("final_response"), str):
            return {"ok": False, "error": "final_response_not_string"}

        return {"ok": True, "payload": payload}

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Short free-form completion for sub-agents (no strict JSON schema)."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if isinstance(max_tokens, int):
            kwargs["max_tokens"] = max_tokens
        else:
            mt = os.getenv("LM_CHAT_MAX_TOKENS")
            if mt and mt.isdigit():
                kwargs["max_tokens"] = int(mt)
        response = self.client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()
