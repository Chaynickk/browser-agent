from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class VacancyValidation:
    is_valid: bool
    canonical_url: str
    normalized_title: str
    relevance_score: float
    quality_score: float
    reasons: list[str]
    structured: dict


_SEARCH_PAGE_TOKENS = ("найдено", "ваканс", "search", "результат")
_RELEVANT_TOKENS = ("ai", "ml", "llm", "machine learning", "python", "data", "engineer", "инженер")
_NEGATIVE_ROLE_TOKENS = (
    "sales",
    "hr",
    "recruiter",
    "call center",
    "оператор",
    "продавец",
    "product manager",
    "project manager",
    "менеджер по продажам",
    "product owner",
)


def validate_vacancy_observation(obs: object, query: str, known_urls: set[str], known_titles: set[str]) -> VacancyValidation:
    url = str(getattr(obs, "url", "") or "")
    title = str(getattr(obs, "title", "") or "")
    structure = getattr(obs, "page_structure", {}) or {}
    visible = [str(x).strip() for x in (structure.get("visible_text") or []) if str(x).strip()]
    headings = [str(x).strip() for x in (structure.get("headings") or []) if str(x).strip()]

    canonical = _canonicalize_url(url)
    candidate_title = _sanitize_title(headings[0] if headings else title)
    reasons: list[str] = []
    quality = 0.0
    relevance = _score_relevance(candidate_title, visible, query)

    if not _looks_like_vacancy_url(canonical):
        reasons.append("url_not_vacancy_like")
    else:
        quality += 1.0

    if _is_search_summary_title(candidate_title):
        reasons.append("search_summary_title")
    else:
        quality += 1.0

    if not _is_human_readable_title(candidate_title):
        reasons.append("bad_title")
    else:
        quality += 1.0

    company = _extract_company(visible)
    salary = _extract_salary(visible)
    requirements = _extract_requirements(visible)
    location = _extract_location(visible)
    experience = _extract_experience(visible)

    if company or requirements:
        quality += 1.0
    else:
        reasons.append("missing_company_and_requirements")

    if canonical in known_urls:
        reasons.append("duplicate_url")
    if candidate_title.lower() in known_titles:
        reasons.append("duplicate_title")

    if relevance < 1.8:
        reasons.append("low_relevance")

    is_valid = (
        quality >= 3.2
        and relevance >= 1.8
        and "duplicate_url" not in reasons
        and "duplicate_title" not in reasons
        and "search_summary_title" not in reasons
        and "bad_title" not in reasons
    )
    structured = {
        "url": canonical,
        "title": candidate_title,
        "company": company,
        "location": location,
        "salary": salary,
        "requirements": requirements[:5],
        "experience": experience,
    }
    return VacancyValidation(
        is_valid=is_valid,
        canonical_url=canonical,
        normalized_title=candidate_title,
        relevance_score=relevance,
        quality_score=quality,
        reasons=reasons,
        structured=structured,
    )


def _canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}" if parsed.scheme and parsed.netloc else url


def _looks_like_vacancy_url(url: str) -> bool:
    lower = url.lower()
    return "/vacancy/" in lower or "/vacancies/" in lower


def _sanitize_title(value: str) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    text = re.sub(r"\|.*$", "", text).strip()
    text = re.sub(r"^hh\.ru\s*[-:]\s*", "", text, flags=re.IGNORECASE).strip()
    return text[:180]


def _is_human_readable_title(title: str) -> bool:
    if not title:
        return False
    if re.search(r"https?://|[?&=]|^\d{6,}$", title):
        return False
    if re.match(r"^[a-z0-9_-]+$", title.lower()):
        return False
    return len(title) >= 5


def _is_search_summary_title(title: str) -> bool:
    lowered = title.lower()
    if any(token in lowered for token in _SEARCH_PAGE_TOKENS) and ("ваканс" in lowered or "vacanc" in lowered):
        return True
    return False


def _extract_company(visible: list[str]) -> str:
    for item in visible:
        if re.search(r"(ооо|зао|ип|ltd|llc|inc|компан)", item, re.IGNORECASE):
            return item[:160]
    return ""


def _extract_salary(visible: list[str]) -> str:
    for item in visible:
        if re.search(r"(₽|руб|usd|eur|\$|kzt|зарплат)", item, re.IGNORECASE):
            return item[:120]
    return ""


def _extract_location(visible: list[str]) -> str:
    for item in visible:
        if re.search(r"(москва|санкт|remote|удален|гибрид|офис)", item, re.IGNORECASE):
            return item[:120]
    return ""


def _extract_experience(visible: list[str]) -> str:
    for item in visible:
        if re.search(r"(опыт|experience|junior|middle|senior)", item, re.IGNORECASE):
            return item[:120]
    return ""


def _extract_requirements(visible: list[str]) -> list[str]:
    return [item[:200] for item in visible if re.search(r"(требован|опыт|python|ml|ai|llm|nlp|sql|backend)", item, re.IGNORECASE)]


def _score_relevance(title: str, visible: list[str], query: str) -> float:
    blob = " ".join([title, query, *visible[:25]]).lower()
    score = 0.0
    title_lower = (title or "").lower()
    if any(token in title_lower for token in ("engineer", "инженер", "developer", "разработ")):
        score += 1.0
    if any(token in title_lower for token in ("manager", "менеджер")) and not any(token in title_lower for token in ("engineering manager",)):
        score -= 1.2
    for token in _RELEVANT_TOKENS:
        if token in blob:
            score += 0.5
    for token in _NEGATIVE_ROLE_TOKENS:
        if token in blob:
            score -= 0.8
    return score
