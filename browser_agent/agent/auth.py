from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .tools import Observation


@dataclass
class AuthGateDecision:
    is_auth_gate: bool
    confidence: int
    rationale: str


_AUTH_URL_TOKENS = ("login", "signin", "sign-in", "auth", "oauth", "session", "account")
_AUTH_TEXT_TOKENS = (
    "login",
    "log in",
    "sign in",
    "войти",
    "вход",
    "авториза",
    "unauthorized",
    "access denied",
    "please sign in",
    "please log in",
)
_PASSWORD_TOKENS = ("password", "пароль", "passcode")
_IDENTITY_TOKENS = ("email", "e-mail", "mail", "phone", "телефон", "username", "логин", "otp", "2fa")


def detect_auth_gate(observation: Observation) -> AuthGateDecision:
    url = (observation.url or "").lower()
    title = (observation.title or "").lower()

    inputs = _as_joined(observation.page_structure.get("inputs") or [])
    buttons = _as_joined(observation.page_structure.get("buttons") or [])
    links = _as_joined(observation.page_structure.get("links") or [])
    headings = _as_joined(observation.page_structure.get("headings") or [])
    visible_text = _as_joined(observation.page_structure.get("visible_text") or [])
    dom_summary = (observation.dom_summary or "").lower()
    blob = " ".join([title, headings, buttons, links, inputs, visible_text, dom_summary])

    score = 0
    reasons: list[str] = []

    if any(token in url for token in _AUTH_URL_TOKENS):
        score += 2
        reasons.append("auth-like url")

    if _contains_any(blob, _AUTH_TEXT_TOKENS):
        score += 2
        reasons.append("auth text markers")

    has_password = _contains_any(blob, _PASSWORD_TOKENS)
    if has_password:
        score += 2
        reasons.append("password field markers")

    if _contains_any(blob, _IDENTITY_TOKENS):
        score += 1
        reasons.append("identity input markers")

    # Typical auth wall pattern: explicit auth copy + password or auth-like URL.
    auth_gate = score >= 4 and (has_password or any(token in url for token in _AUTH_URL_TOKENS))
    rationale = ", ".join(reasons) if reasons else "no auth signals"
    return AuthGateDecision(is_auth_gate=auth_gate, confidence=score, rationale=rationale)


def _as_joined(values: Iterable[object]) -> str:
    return " ".join(str(v).lower() for v in values if v is not None)


def _contains_any(text: str, tokens: Iterable[str]) -> bool:
    normalized = re.sub(r"\s+", " ", text)
    return any(token in normalized for token in tokens)
