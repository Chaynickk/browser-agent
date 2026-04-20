"""
Smoke test: Playwright + observe + executor tools (no LLM / no API key required).
Run: python smoke_test.py
"""

from __future__ import annotations

from browser_agent.agent.browser import BrowserController
from browser_agent.agent.executor import Executor
from browser_agent.agent.tools import ToolAction

HTML = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Smoke</title></head>
<body>
  <h1>Smoke test page</h1>
  <button type="button" name="go">Go</button>
  <label for="q">Search</label>
  <input id="q" type="text" name="q" placeholder="Type here" />
  <a href="#x">Example link</a>
</body>
</html>
"""


def main() -> None:
    browser = BrowserController(headless=True)
    browser.start()
    try:
        assert browser.page
        browser.page.set_content(HTML)
        obs = browser.observe(step=1)
        assert obs.url.startswith("http") or obs.url == "about:blank"
        assert "Smoke" in obs.title or "smoke" in obs.dom_summary.lower()
        assert obs.available_actions or "button" in obs.dom_summary.lower()

        ex = Executor(browser=browser)
        r1 = ex.execute(ToolAction(tool="click", args={"role": "button", "name": "Go"}, reason="smoke"), step=1)
        assert r1.success, r1.message
        r2 = ex.execute(
            ToolAction(tool="type", args={"role": "textbox", "name": "Search", "text": "hello"}, reason="smoke"),
            step=2,
        )
        assert r2.success, r2.message
        r3 = ex.execute(ToolAction(tool="scroll", args={"direction": "down", "amount": 400}, reason="smoke"), step=3)
        assert r3.success, r3.message
        print("smoke_test OK:", r1.message, "|", r2.message, "|", r3.message)
    finally:
        browser.close()


if __name__ == "__main__":
    main()
