from __future__ import annotations

import re


def extract_profile_summary(obs: object) -> str:
    title = str(getattr(obs, "title", "") or "")
    url = str(getattr(obs, "url", "") or "")
    structure = getattr(obs, "page_structure", {}) or {}
    visible = [str(x).strip() for x in (structure.get("visible_text") or []) if str(x).strip()]

    if not re.search(r"(resume|profile|cv|резюме|профил)", f"{url} {title}", re.IGNORECASE):
        return ""

    role = next((x for x in visible if re.search(r"(engineer|разработ|инженер|data|ml|ai)", x, re.IGNORECASE)), "")
    skills = [x for x in visible if re.search(r"(python|sql|ml|ai|llm|docker|kubernetes|backend|api)", x, re.IGNORECASE)]
    exp = next((x for x in visible if re.search(r"(опыт|experience|лет|year)", x, re.IGNORECASE)), "")
    achievements = [x for x in visible if re.search(r"(проект|project|достижен|deliver|result)", x, re.IGNORECASE)]

    parts = []
    if role:
        parts.append(f"Role: {role[:180]}")
    if skills:
        parts.append(f"Skills: {'; '.join(skills[:5])[:320]}")
    if exp:
        parts.append(f"Experience: {exp[:200]}")
    if achievements:
        parts.append(f"Achievements: {'; '.join(achievements[:3])[:320]}")
    summary = "\n".join(parts).strip()
    return summary if _is_profile_summary_meaningful(summary) else ""


def is_profile_summary_meaningful(summary: str) -> bool:
    return _is_profile_summary_meaningful(summary)


def _is_profile_summary_meaningful(summary: str) -> bool:
    text = (summary or "").strip()
    if len(text) < 60:
        return False
    labels = sum(1 for marker in ("Role:", "Skills:", "Experience:", "Achievements:") if marker in text)
    if labels < 2:
        return False
    if "headhunter api" in text.lower() and labels < 3:
        return False
    return True
