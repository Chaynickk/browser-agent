from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class NormalizedIntent:
    raw_task: str
    search_query: str
    target_domain: str
    is_search_like: bool


_SEARCH_TOKENS = ("find", "search", "job", "vacanc", "ваканс", "работ", "найд", "поиск")
_DOMAIN_RE = re.compile(r"\b([a-z0-9][a-z0-9.-]+\.[a-z]{2,})\b", re.IGNORECASE)


def normalize_task_intent(task: str) -> NormalizedIntent:
    raw = (task or "").strip()
    lower = raw.lower()
    domain_match = _DOMAIN_RE.search(raw)
    target_domain = domain_match.group(1).lower() if domain_match else ""
    is_search = any(token in lower for token in _SEARCH_TOKENS)
    search_query = _extract_query(raw)
    return NormalizedIntent(
        raw_task=raw,
        search_query=search_query,
        target_domain=target_domain,
        is_search_like=is_search,
    )


def _extract_query(task: str) -> str:
    # 1) Keep common role-like Latin spans such as "AI Engineer", "ML Engineer", "Data Scientist".
    role_patterns = [
        r"\b(ai|ml|data|prompt|llm)\s*(engineer|developer|scientist|researcher|specialist)\b",
        r"\b(machine learning|artificial intelligence)\s*(engineer|developer|scientist)\b",
    ]
    for pattern in role_patterns:
        m = re.search(pattern, task, flags=re.IGNORECASE)
        if m:
            value = re.sub(r"\s+", " ", m.group(0)).strip()
            return _title_case_ascii(value)

    # 2) If the task has a short quoted segment, prefer it.
    quote_match = re.search(r"[\"'«](.{2,40}?)[\"'»]", task)
    if quote_match:
        candidate = _sanitize_query(quote_match.group(1))
        if candidate:
            return candidate

    # 3) Fallback: extract a compact role phrase around vacancy/job markers.
    candidate = re.sub(
        r"(?i)\b(найди|подбери|ищи|ваканси[ияеию]|работ[ауы]|позици[яию]|на|в|и|с|по|для|откликнись|отклик)\b",
        " ",
        task,
    )
    candidate = _sanitize_query(candidate)
    if candidate:
        words = candidate.split()
        if len(words) > 4:
            words = words[:4]
        return " ".join(words)
    return "AI Engineer"


def _sanitize_query(value: str) -> str:
    text = re.sub(r"https?://\S+|\b[a-z0-9.-]+\.[a-z]{2,}\b", " ", value, flags=re.IGNORECASE)
    text = re.sub(r"[^0-9A-Za-zА-Яа-я+\- ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if len(text) > 60:
        text = text[:60].strip()
    return text


def _title_case_ascii(value: str) -> str:
    words = re.sub(r"\s+", " ", value).strip().split(" ")
    return " ".join(word.capitalize() if word.isascii() else word for word in words)
