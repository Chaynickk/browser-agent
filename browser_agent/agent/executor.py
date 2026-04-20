from __future__ import annotations

from typing import Callable, Optional

from .browser import BrowserController
from .tools import ExecutionResult, ToolAction


DestructiveConfirm = Callable[[ToolAction], bool]


class Executor:
    def __init__(
        self,
        browser: BrowserController,
        confirm_callback: Optional[DestructiveConfirm] = None,
    ) -> None:
        self.browser = browser
        self.confirm_callback = confirm_callback

    def execute(self, action: ToolAction, step: Optional[int] = None) -> ExecutionResult:
        self._log(step, action, "start")
        try:
            if action.tool == "navigate":
                msg = self.browser.navigate(url=action.args["url"])
            elif action.tool == "click":
                msg = self.browser.click(
                    role=action.args["role"],
                    name=action.args["name"],
                    exact=bool(action.args.get("exact", False)),
                )
            elif action.tool == "type":
                msg = self.browser.type_text(
                    role=action.args["role"],
                    name=action.args["name"],
                    text=action.args["text"],
                    exact=bool(action.args.get("exact", False)),
                )
            elif action.tool == "scroll":
                msg = self.browser.scroll(
                    direction=str(action.args.get("direction", "down")),
                    amount=int(action.args.get("amount", 1200)),
                )
            elif action.tool == "wait":
                msg = self.browser.wait(ms=int(action.args.get("ms", 700)))
            elif action.tool == "back":
                msg = self.browser.back()
            elif action.tool == "finish_task":
                msg = action.args.get("message", "Task finished")
            else:
                result = ExecutionResult(False, f"Unknown tool: {action.tool}")
                self._log(step, action, "error", result.message, stop_reason="unknown_tool")
                return result

            result = ExecutionResult(
                True,
                msg,
                details={"tool": action.tool, "args": action.args, "reason": action.reason},
            )
            self._log(step, action, "done", result.message)
            return result
        except Exception as exc:  # noqa: BLE001
            result = ExecutionResult(False, f"Execution error: {exc}", details={"tool": action.tool})
            self._log(step, action, "error", result.message, stop_reason="execution_exception")
            return result

    def confirm_action(self, action: ToolAction, rationale: str = "") -> bool:
        if self.confirm_callback:
            return self.confirm_callback(action)
        print(f"[SECURITY] Action requires confirmation: {action}")
        if rationale:
            print(f"[SECURITY] Rationale: {rationale}")
        answer = input("Allow this action? [y/N]: ").strip().lower()
        return answer in {"y", "yes"}

    @staticmethod
    def _log(
        step: Optional[int],
        action: ToolAction,
        phase: str,
        result: str = "",
        stop_reason: str = "",
    ) -> None:
        step_text = "?" if step is None else str(step)
        safe_result = _sanitize_for_console(result or "-")
        safe_reason = _sanitize_for_console(action.reason or "-")
        print(
            f"[executor] step={step_text} phase={phase} action={action.tool} "
            f"reason={safe_reason} result={safe_result} stop_reason={stop_reason or '-'}"
        )


def _sanitize_for_console(value: str) -> str:
    return str(value).encode("cp1251", errors="replace").decode("cp1251", errors="replace")
