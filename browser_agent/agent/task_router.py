from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote_plus


@dataclass
class RouteDecision:
    start_url: str
    source: str


_URL_RE = re.compile(r"(https?://[^\s]+)", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"\b([a-z0-9][a-z0-9.-]+\.[a-z]{2,})(/[^\s]*)?\b", re.IGNORECASE)


def infer_start_url(task: str) -> RouteDecision:
    text = (task or "").strip()
    lowered = text.lower()

    direct_url = _extract_direct_url(text)
    if direct_url:
        return RouteDecision(start_url=direct_url, source="task_url")

    domain_url = _extract_domain_url(text)
    if domain_url:
        return RouteDecision(start_url=domain_url, source="task_domain")

    routed = _route_by_intent(lowered)
    if routed:
        return routed

    query = quote_plus(text or "web task")
    return RouteDecision(
        start_url=f"https://www.google.com/search?q={query}",
        source="intent_search_fallback",
    )


def _extract_direct_url(text: str) -> str | None:
    m = _URL_RE.search(text)
    if not m:
        return None
    return m.group(1).rstrip(".,);")


def _extract_domain_url(text: str) -> str | None:
    m = _DOMAIN_RE.search(text)
    if not m:
        return None
    domain = m.group(1).lower()
    if domain.startswith(("www.", "m.")):
        domain = domain.split(".", 1)[1]
    return f"https://{domain}"


def _route_by_intent(lowered_task: str) -> RouteDecision | None:
    intent_routes: list[tuple[list[str], str, str]] = [
        (
            ["почт", "email", "mail", "inbox", "gmail", "outlook"],
            "https://mail.google.com",
            "intent_email",
        ),
        (
            ["ваканс", "работ", "job", "vacanc", "resume", "hh.ru", "linkedin"],
            "https://hh.ru",
            "intent_jobs",
        ),
        (
            ["заказ", "достав", "еда", "food", "delivery", "restaurant"],
            "https://www.google.com/search?q=food+delivery",
            "intent_food_delivery",
        ),
        (
            ["playwright", "docs", "documentation", "документац"],
            "https://playwright.dev",
            "intent_docs",
        ),
    ]
    for keywords, url, source in intent_routes:
        if any(keyword in lowered_task for keyword in keywords):
            return RouteDecision(start_url=url, source=source)
    return None
