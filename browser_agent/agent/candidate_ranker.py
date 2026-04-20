from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class RankedCandidate:
    role: str
    name: str
    score: float
    reasons: list[str]
    raw: dict[str, Any]


NEGATIVE_INPUT_TOKENS = (
    "исключить",
    "exclude",
    "salary",
    "зарплат",
    "от ",
    "до ",
    "location",
    "регион",
    "город",
    "sort",
    "сортиров",
)
POSITIVE_INPUT_TOKENS = (
    "поиск",
    "search",
    "вакан",
    "должност",
    "професс",
    "position",
    "job",
    "company",
    "компани",
)
POSITIVE_BUTTON_TOKENS = ("поиск", "найти", "search", "show", "показать")
NEGATIVE_BUTTON_TOKENS = ("создать", "publish", "new", "add", "chat", "write", "post")


def rank_primary_input(page_structure: dict[str, Any], query: str) -> list[RankedCandidate]:
    candidates = _as_list(page_structure.get("input_candidates"))
    ranked: list[RankedCandidate] = []
    for candidate in candidates:
        name = str(candidate.get("name", "")).strip()
        role = _resolve_input_role(candidate)
        if not name or role not in {"searchbox", "textbox", "combobox"}:
            continue
        lowered = name.lower()
        score = 0.0
        reasons: list[str] = []
        if role in {"searchbox", "combobox"}:
            score += 3.0
            reasons.append("search-capable-role")
        if _contains_any(lowered, POSITIVE_INPUT_TOKENS):
            score += 4.0
            reasons.append("search-semantics")
        if _contains_any(lowered, NEGATIVE_INPUT_TOKENS):
            score -= 5.0
            reasons.append("aux-filter-penalty")
        if int(candidate.get("y", 9999)) < 380:
            score += 1.5
            reasons.append("top-page-position")
        if int(candidate.get("width", 0)) >= 220:
            score += 1.5
            reasons.append("wide-primary-like")
        if query and len(query) <= 40:
            score += 0.3
        ranked.append(RankedCandidate(role=role, name=name, score=score, reasons=reasons, raw=candidate))
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked


def rank_submit_buttons(page_structure: dict[str, Any], chosen_input: RankedCandidate | None) -> list[RankedCandidate]:
    candidates = _as_list(page_structure.get("button_candidates"))
    ranked: list[RankedCandidate] = []
    in_form = str(chosen_input.raw.get("formName", "")).strip().lower() if chosen_input else ""
    ix = int(chosen_input.raw.get("x", 0)) if chosen_input else 0
    iy = int(chosen_input.raw.get("y", 0)) if chosen_input else 0
    for candidate in candidates:
        name = str(candidate.get("name", "")).strip()
        if not name:
            continue
        lowered = name.lower()
        score = 0.0
        reasons: list[str] = []
        if _contains_any(lowered, POSITIVE_BUTTON_TOKENS):
            score += 5.0
            reasons.append("search-submit-semantics")
        if _contains_any(lowered, NEGATIVE_BUTTON_TOKENS):
            score -= 6.0
            reasons.append("unrelated-action-penalty")
        form = str(candidate.get("formName", "")).strip().lower()
        if in_form and form and form == in_form:
            score += 3.0
            reasons.append("same-form")
        cx = int(candidate.get("x", 0))
        cy = int(candidate.get("y", 0))
        dist = abs(cx - ix) + abs(cy - iy)
        if chosen_input and dist <= 420:
            score += 2.0
            reasons.append("near-input")
        ranked.append(RankedCandidate(role="button", name=name, score=score, reasons=reasons, raw=candidate))
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked


def rank_vacancy_links(page_structure: dict[str, Any]) -> list[RankedCandidate]:
    candidates = _as_list(page_structure.get("link_candidates"))
    ranked: list[RankedCandidate] = []
    for candidate in candidates:
        name = str(candidate.get("name", "")).strip()
        href = str(candidate.get("href", "")).lower()
        if not name:
            continue
        lowered = name.lower()
        score = 0.0
        reasons: list[str] = []
        if "vacanc" in href or "vakans" in href or "/vacancy/" in href:
            score += 4.0
            reasons.append("vacancy-link-url")
        if any(token in lowered for token in ("инженер", "engineer", "ai", "ml", "data")):
            score += 2.0
            reasons.append("role-like-title")
        if len(name) > 22:
            score += 1.0
        ranked.append(RankedCandidate(role="link", name=name, score=score, reasons=reasons, raw=candidate))
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked


def rank_profile_links(page_structure: dict[str, Any]) -> list[RankedCandidate]:
    candidates = _as_list(page_structure.get("link_candidates"))
    ranked: list[RankedCandidate] = []
    for candidate in candidates:
        name = str(candidate.get("name", "")).strip()
        href = str(candidate.get("href", "")).lower()
        if not name:
            continue
        lowered = name.lower()
        score = 0.0
        reasons: list[str] = []
        if any(token in lowered for token in ("резюме", "профиль", "resume", "profile", "cv")):
            score += 5.0
            reasons.append("profile-semantics")
        if any(token in href for token in ("resume", "profile", "cv", "resumes")):
            score += 3.0
            reasons.append("profile-url")
        if any(token in lowered for token in ("создать", "create", "new resume")):
            score -= 4.0
            reasons.append("create-resume-penalty")
        if any(token in lowered for token in ("мое", "готовое", "my resume", "my profile")):
            score += 2.0
            reasons.append("existing-profile-priority")
        if score > 0:
            ranked.append(RankedCandidate(role="link", name=name, score=score, reasons=reasons, raw=candidate))
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked


def _resolve_input_role(candidate: dict[str, Any]) -> str:
    role = str(candidate.get("role", "")).strip().lower()
    typ = str(candidate.get("type", "")).strip().lower()
    if role in {"searchbox", "textbox", "combobox"}:
        return role
    if typ in {"search"}:
        return "searchbox"
    if typ in {"text", "textarea"}:
        return "textbox"
    return "combobox"


def _contains_any(text: str, tokens: Iterable[str]) -> bool:
    return any(token in text for token in tokens)


def _as_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, dict)]
