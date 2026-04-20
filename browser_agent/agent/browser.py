from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from .env_bootstrap import load_dotenv_if_available
from .tools import Observation

load_dotenv_if_available()

# Max chars sent toward the LLM from a single observation (context budget).
_DEFAULT_OBS_TEXT_LIMIT = int(os.getenv("OBS_TEXT_CHAR_LIMIT", "9000"))


class BrowserController:
    def __init__(self, headless: bool = False, profile_dir: Optional[str] = None) -> None:
        self.headless = headless
        self.profile_dir = profile_dir
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def _launch_options(self) -> Dict[str, Any]:
        opts: Dict[str, Any] = {"headless": self.headless}
        exe = os.getenv("BROWSER_EXECUTABLE_PATH") or os.getenv("CHROMIUM_EXECUTABLE_PATH")
        if exe:
            opts["executable_path"] = exe
        return opts

    def start(self) -> None:
        self._playwright = sync_playwright().start()
        launch_opts = self._launch_options()
        if self.profile_dir:
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=self.profile_dir,
                **launch_opts,
            )
            self.page = self._context.pages[0] if self._context.pages else self._context.new_page()
        else:
            self._browser = self._playwright.chromium.launch(**launch_opts)
            self._context = self._browser.new_context()
            self.page = self._context.new_page()

    def close(self) -> None:
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def observe(self, step: int) -> Observation:
        if not self.page:
            raise RuntimeError("Browser is not started")
        aria = self._capture_aria_snapshot()
        dom_scan = self._generic_dom_scan()
        dom_summary = self._merge_dom_summary(aria, dom_scan)
        dom_summary = self._truncate(dom_summary, _DEFAULT_OBS_TEXT_LIMIT)
        available_actions = self._hints_from_aria_and_scan(aria, dom_scan)
        page_structure = self._build_page_structure(aria, dom_scan)
        return Observation(
            url=self.page.url,
            title=self.page.title(),
            dom_summary=dom_summary,
            available_actions=available_actions,
            step=step,
            page_structure=page_structure,
        )

    def navigate(self, url: str) -> str:
        if not self.page:
            raise RuntimeError("Browser is not started")
        self.page.goto(url, wait_until="domcontentloaded")
        return f"Navigated to {url}"

    def save_screenshot(self, path: str) -> str:
        if not self.page:
            raise RuntimeError("Browser is not started")
        self.page.screenshot(path=path, full_page=True)
        return f"Saved screenshot: {path}"

    def click(self, role: str, name: str, exact: bool = False) -> str:
        if not self.page:
            raise RuntimeError("Browser is not started")
        resolution_path: List[str] = []
        try:
            locator = self.page.get_by_role(role, name=name, exact=exact)
            locator.first.scroll_into_view_if_needed()
            locator.first.click(force=True)
            resolution_path.append("role_name_exact" if exact else "role_name")
            return f"Clicked role={role} name={name} via={'>'.join(resolution_path)}"
        except Exception as first_exc:  # noqa: BLE001
            resolution_path.append("role_name_failed")
            try:
                relaxed = self.page.get_by_role(role, name=re.compile(re.escape(name), re.IGNORECASE))
                relaxed.first.scroll_into_view_if_needed()
                relaxed.first.click(force=True)
                resolution_path.append("role_name_regex")
                return f"Clicked role={role} name={name} via={'>'.join(resolution_path)}"
            except Exception:  # noqa: BLE001
                raise first_exc

    def type_text(self, role: str, name: str, text: str, exact: bool = False) -> str:
        if not self.page:
            raise RuntimeError("Browser is not started")
        last_error: Optional[Exception] = None

        def try_fill(desc: str, get_locator: Callable[[], Any]) -> str:
            get_locator().first.fill(text)
            return desc

        strategies: List[tuple[str, Callable[[], Any]]] = [
            (
                f"role={role} name={name}",
                lambda: self.page.get_by_role(role, name=name, exact=exact),
            ),
            (
                f"role_regex={role} name~={name}",
                lambda: self.page.get_by_role(role, name=re.compile(re.escape(name), re.IGNORECASE)),
            ),
            (
                f"label={name}",
                lambda: self.page.get_by_label(name, exact=exact),
            ),
        ]
        for desc, factory in strategies:
            try:
                locator = factory().first
                locator.click(timeout=5000)
                try:
                    locator.fill(text)
                except Exception:  # noqa: BLE001
                    # Some combobox widgets are not fill-compatible; fallback to keyboard typing.
                    locator.press("Control+A")
                    locator.press("Backspace")
                    locator.type(text, delay=5)
                return desc
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

        for relaxed_role in ("textbox", "searchbox", "combobox"):
            if relaxed_role == role:
                continue
            try:
                loc = self.page.get_by_role(relaxed_role, name=name, exact=exact)
                if loc.count() == 0:
                    continue
                loc.first.fill(text)
                return f"Typed into role={relaxed_role} name={name} via=relaxed_role"
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

        # Final fallback for search-like controls: use first visible control by role, ignoring exact name.
        if role in {"textbox", "searchbox", "combobox"}:
            try:
                generic = self.page.get_by_role(role)
                if generic.count() > 0:
                    field = generic.first
                    field.click(timeout=4000)
                    try:
                        field.fill(text)
                    except Exception:  # noqa: BLE001
                        field.press("Control+A")
                        field.press("Backspace")
                        field.type(text, delay=5)
                    return f"Typed into role={role} via=generic_role_fallback"
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        if last_error:
            raise last_error
        raise RuntimeError("type_text: no matching field")

    def back(self) -> str:
        if not self.page:
            raise RuntimeError("Browser is not started")
        self.page.go_back(wait_until="domcontentloaded")
        return "Went back"

    def wait(self, ms: int) -> str:
        if not self.page:
            raise RuntimeError("Browser is not started")
        self.page.wait_for_timeout(ms)
        return f"Waited {ms} ms"

    def scroll(self, direction: str = "down", amount: int = 1200) -> str:
        if not self.page:
            raise RuntimeError("Browser is not started")
        amount = int(amount)
        if direction == "up":
            delta = -abs(amount)
        else:
            delta = abs(amount)
            direction = "down"
        self.page.mouse.wheel(0, delta)
        self.page.wait_for_timeout(250)
        return f"Scrolled {direction} by {abs(delta)}"

    def _capture_aria_snapshot(self) -> str:
        if not self.page:
            return ""
        try:
            snap = self.page.locator("body").aria_snapshot(timeout=15_000)
            return (snap or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    def _generic_dom_scan(self) -> Dict[str, Any]:
        """Site-agnostic DOM harvest (tag-level, not page-specific selectors)."""
        if not self.page:
            return {}
        script = """() => {
          const clip = (s, n) => (s || '').replace(/\\s+/g, ' ').trim().slice(0, n);
          const headings = [];
          document.querySelectorAll('h1,h2,h3').forEach((el) => {
            const t = clip(el.innerText, 240);
            if (t) headings.push(t);
          });
          const buttons = [];
          document.querySelectorAll('button,[role="button"],input[type="submit"],input[type="button"]').forEach((el) => {
            const name = clip(el.getAttribute('aria-label') || el.innerText || el.value, 200);
            if (name) buttons.push(name);
          });
          const links = [];
          document.querySelectorAll('a[href]').forEach((el) => {
            const name = clip(el.getAttribute('aria-label') || el.innerText, 200);
            if (name) links.push(name);
          });
          const inputs = [];
          document.querySelectorAll('input:not([type="hidden"]),textarea,select').forEach((el) => {
            const ph = clip(el.getAttribute('placeholder'), 120);
            const al = clip(el.getAttribute('aria-label'), 120);
            const name = clip(el.getAttribute('name'), 80);
            const typ = (el.getAttribute('type') || el.tagName || '').toLowerCase();
            const parts = [typ, al, ph, name].filter(Boolean);
            if (parts.length) inputs.push(parts.join(' | '));
          });
          const textSnippets = [];
          document.querySelectorAll('p,li,td,th').forEach((el) => {
            const t = clip(el.innerText, 220);
            if (t && t.length > 3) textSnippets.push(t);
          });

          const toCandidate = (el, kind, fallbackName='') => {
            const rect = el.getBoundingClientRect();
            const name = clip(
              el.getAttribute('aria-label') ||
              el.getAttribute('placeholder') ||
              el.innerText ||
              el.value ||
              fallbackName,
              180
            );
            const role = (el.getAttribute('role') || '').toLowerCase();
            const type = (el.getAttribute('type') || el.tagName || '').toLowerCase();
            const id = clip(el.getAttribute('id'), 80);
            const form = el.closest('form');
            const formName = form ? clip(form.getAttribute('id') || form.getAttribute('name') || form.className, 120) : '';
            const className = clip(el.className, 120);
            if (!name) return null;
            return {
              kind,
              name,
              role,
              type,
              id,
              className,
              formName,
              x: Math.round(rect.x),
              y: Math.round(rect.y),
              width: Math.round(rect.width),
              height: Math.round(rect.height),
              visible: rect.width > 2 && rect.height > 2,
            };
          };

          const inputCandidates = [];
          document.querySelectorAll('input:not([type="hidden"]),textarea,[role="textbox"],[role="searchbox"],[role="combobox"]').forEach((el) => {
            const c = toCandidate(el, 'input');
            if (c && c.visible) inputCandidates.push(c);
          });
          const buttonCandidates = [];
          document.querySelectorAll('button,[role="button"],input[type="submit"],input[type="button"]').forEach((el) => {
            const c = toCandidate(el, 'button');
            if (c && c.visible) buttonCandidates.push(c);
          });
          const linkCandidates = [];
          document.querySelectorAll('a[href]').forEach((el) => {
            const c = toCandidate(el, 'link');
            if (!c || !c.visible) return;
            const href = clip(el.getAttribute('href') || '', 240);
            c.href = href;
            linkCandidates.push(c);
          });

          return { headings, buttons, links, inputs, textSnippets, inputCandidates, buttonCandidates, linkCandidates };
        }"""
        try:
            data = self.page.evaluate(script)
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 20] + "\n...[truncated]..."

    def _merge_dom_summary(self, aria: str, dom_scan: Dict[str, Any]) -> str:
        parts: List[str] = []
        if aria:
            parts.append("=== ARIA snapshot (body) ===\n" + aria)
        scan_lines = self._format_dom_scan(dom_scan)
        if scan_lines:
            parts.append("=== Generic DOM scan ===\n" + "\n".join(scan_lines))
        if not parts:
            return "No ARIA snapshot or DOM scan available (empty page or blocked content)."
        return "\n\n".join(parts)

    @staticmethod
    def _format_dom_scan(dom_scan: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        for key in ("headings", "buttons", "links", "inputs", "textSnippets"):
            items = dom_scan.get(key) or []
            if not isinstance(items, list):
                continue
            uniq = BrowserController._unique_keep_order([str(x) for x in items if x], limit=25)
            if uniq:
                lines.append(f"{key}: " + "; ".join(uniq))
        return lines

    def _hints_from_aria_and_scan(self, aria: str, dom_scan: Dict[str, Any]) -> List[str]:
        hints: List[str] = []
        for line in aria.splitlines():
            m = re.match(r"^\s*-\s*(link|button|textbox|searchbox|combobox|menuitem)\s+\"([^\"]+)\"", line)
            if m:
                hints.append(f"{m.group(1)}:{m.group(2)}")
            m2 = re.match(r"^\s*-\s*(link|button|textbox|searchbox|combobox|menuitem)\s+'([^']+)'", line)
            if m2:
                hints.append(f"{m2.group(1)}:{m2.group(2)}")
        for label, key in (
            ("button", "buttons"),
            ("link", "links"),
        ):
            for name in dom_scan.get(key) or []:
                n = str(name).strip()
                if n:
                    hints.append(f"{label}:{n}")
        seen: set[str] = set()
        out: List[str] = []
        for h in hints:
            if h not in seen:
                seen.add(h)
                out.append(h)
            if len(out) >= 35:
                break
        return out

    def _build_page_structure(self, aria: str, dom_scan: Dict[str, Any]) -> Dict[str, Any]:
        headings = self._unique_keep_order(
            [str(x) for x in (dom_scan.get("headings") or []) if x],
            limit=12,
        )
        buttons = self._unique_keep_order(
            [str(x) for x in (dom_scan.get("buttons") or []) if x],
            limit=20,
        )
        links = self._unique_keep_order(
            [str(x) for x in (dom_scan.get("links") or []) if x],
            limit=20,
        )
        inputs = self._unique_keep_order(
            [str(x) for x in (dom_scan.get("inputs") or []) if x],
            limit=20,
        )
        visible_text = self._unique_keep_order(
            [str(x) for x in (dom_scan.get("textSnippets") or []) if x],
            limit=24,
        )
        summary = (
            f"title={self.page.title() if self.page else ''}; "
            f"headings={len(headings)}, buttons={len(buttons)}, links={len(links)}, inputs={len(inputs)}; "
            f"aria_chars={len(aria)}"
        )
        return {
            "summary": summary,
            "headings": headings,
            "visible_text": visible_text,
            "buttons": buttons,
            "links": links,
            "inputs": inputs,
            "input_candidates": dom_scan.get("inputCandidates") or [],
            "button_candidates": dom_scan.get("buttonCandidates") or [],
            "link_candidates": dom_scan.get("linkCandidates") or [],
        }

    @staticmethod
    def _unique_keep_order(items: List[str], limit: int) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())
