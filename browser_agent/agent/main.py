from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from .auth import detect_auth_gate
from .candidate_ranker import rank_primary_input, rank_profile_links, rank_submit_buttons, rank_vacancy_links
from .env_bootstrap import load_dotenv_if_available
from .loop_control import LoopMonitor
from .profile_capture import extract_profile_summary, is_profile_summary_meaningful
from .query_normalizer import normalize_task_intent
from .runtime_debug import RuntimeDebugWriter
from .strategy import choose_stuck_override, detect_search_ui, is_search_like_task
from .task_router import infer_start_url
from .vacancy_quality import validate_vacancy_observation

load_dotenv_if_available()

from .browser import BrowserController
from .error_reflector import ErrorReflector
from .executor import Executor
from .llm import LLMClient
from .memory import MemoryStore
from .planner import Planner
from .safety_evaluator import SafetyEvaluator
from .tools import ExecutionResult, PlanStep, ToolAction

_STRUCTURE_JSON_LIMIT = int(os.getenv("OBS_STRUCTURE_JSON_LIMIT", "2500"))
_OBS_ACTION_HINT_LIMIT = int(os.getenv("OBS_ACTION_HINT_LIMIT", "18"))
_OBS_DOM_SUMMARY_LIMIT = int(os.getenv("OBS_DOM_SUMMARY_LIMIT", "2800"))


@dataclass
class SearchFlowState:
    stage: str = "bootstrap"
    query_typed: bool = False
    search_submitted: bool = False
    profile_checked: bool = False
    failed_input_targets: set[str] = field(default_factory=set)
    failed_link_targets: set[str] = field(default_factory=set)
    opened_links: set[str] = field(default_factory=set)
    profile_attempted: bool = False
    valid_vacancies: list[dict] = field(default_factory=list)
    rejected_vacancies: list[dict] = field(default_factory=list)
    profile_summary: str = ""
    search_results_url: str = ""
    candidate_queue: list[dict] = field(default_factory=list)
    seen_vacancy_ids: set[str] = field(default_factory=set)
    processed_vacancy_ids: set[str] = field(default_factory=set)
    rejected_vacancy_ids: set[str] = field(default_factory=set)
    no_progress_steps: int = 0
    pending_vacancy_id: str = ""
    profile_capture_failures: int = 0


def _compact_page_structure(page_structure: dict) -> dict:
    if not isinstance(page_structure, dict):
        return {}
    compact: dict = {}
    for key, limit in (
        ("summary", 1),
        ("headings", 8),
        ("inputs", 8),
        ("buttons", 10),
        ("links", 10),
        ("visible_text", 8),
    ):
        value = page_structure.get(key)
        if isinstance(value, list):
            compact[key] = [str(x) for x in value[:limit]]
        elif isinstance(value, str):
            compact[key] = value[:220]
    return compact


def _observation_signature(obs: object) -> str:
    raw = json.dumps(_compact_page_structure(obs.page_structure), ensure_ascii=True, sort_keys=True)
    compact = f"{obs.url}|{obs.title}|{raw}"
    compact = re.sub(r"\s+", " ", compact)
    return compact[:900]


def _short_candidates(items: list, limit: int = 4) -> list[dict]:
    out: list[dict] = []
    for item in items[:limit]:
        out.append(
            {
                "role": item.role,
                "name": item.name,
                "score": round(item.score, 2),
                "reasons": item.reasons[:3],
            }
        )
    return out


def _extract_vacancy_details(obs: object) -> dict:
    structure = obs.page_structure if isinstance(obs.page_structure, dict) else {}
    title = str(obs.title or "").strip()
    visible = [str(x).strip() for x in (structure.get("visible_text") or []) if str(x).strip()]
    headings = [str(x).strip() for x in (structure.get("headings") or []) if str(x).strip()]
    salary = next((line for line in visible if re.search(r"(₽|руб|usd|eur|\$|kzt|зарплат)", line, re.IGNORECASE)), "")
    company = next((line for line in visible if re.search(r"(ооо|зао|ип|ltd|llc|inc)", line, re.IGNORECASE)), "")
    requirements = [line for line in visible if re.search(r"(требован|опыт|python|ml|ai|llm|nlp|sql)", line, re.IGNORECASE)]
    return {
        "url": obs.url,
        "title": headings[0] if headings else title,
        "company": company[:160],
        "location": next((line for line in visible if re.search(r"(москва|санкт|remote|удален|гибрид)", line, re.IGNORECASE)), "")[:120],
        "salary": salary[:120],
        "requirements": requirements[:4],
    }


def _build_cover_letters(llm: LLMClient, profile_summary: str, vacancies: list[dict], task: str) -> str:
    if not vacancies or not profile_summary.strip():
        return "No vacancies were collected."
    if os.getenv("USE_LLM_COVER_LETTERS", "0") != "1":
        letters: list[str] = []
        profile_line = profile_summary[:260]
        for idx, vacancy in enumerate(vacancies[:3], start=1):
            title = vacancy.get("title") or "позиция"
            company = vacancy.get("company") or "компания"
            req = ", ".join(vacancy.get("requirements") or [])[:220]
            letters.append(
                f"{idx}) Для вакансии '{title}' ({company}):\n"
                f"Здравствуйте! Меня заинтересовала позиция {title}. "
                f"Мой релевантный опыт: {profile_line}. "
                f"С учетом требований ({req or 'требования уточняются на странице'}) я могу быстро включиться в задачи и принести практический результат."
            )
        return "\n\n".join(letters)
    prompt = {
        "task": task[:260],
        "profile_summary": profile_summary[:700],
        "vacancies": vacancies[:3],
    }
    text = llm.chat_text(
        "You are a professional job assistant. Create concise personalized cover letters in Russian for each vacancy. "
        "Use factual vacancy details and profile summary only. Return plain text with numbered sections.",
        json.dumps(prompt, ensure_ascii=True),
        temperature=0.2,
        max_tokens=int(os.getenv("LM_COVER_MAX_TOKENS", "420")),
    )
    return text.strip() or "Cover letters generation returned empty output."


def _has_meaningful_vacancy(vacancy: dict) -> bool:
    title = str(vacancy.get("title", "")).strip()
    if not title or re.search(r"https?://|[?&=]|^\d+$", title):
        return False
    has_company_or_req = bool(vacancy.get("company")) or bool(vacancy.get("requirements"))
    return has_company_or_req


def _is_search_results_page(url: str, title: str) -> bool:
    u = (url or "").lower()
    t = (title or "").lower()
    return "/search/vacancy" in u or "/vacancies" in u or ("ваканс" in t and "найден" in t)


def _vacancy_id_from_url(url: str) -> str:
    m = re.search(r"/vacancy/(\d+)", url or "", re.IGNORECASE)
    return m.group(1) if m else ""


def _canonical_vacancy_url(url: str) -> str:
    parsed = urlparse(url or "")
    vid = _vacancy_id_from_url(url)
    if parsed.scheme and parsed.netloc and vid:
        return f"{parsed.scheme}://{parsed.netloc}/vacancy/{vid}"
    return url


def _collect_search_candidates(obs: object) -> list[dict]:
    structure = obs.page_structure if isinstance(obs.page_structure, dict) else {}
    links = structure.get("link_candidates") or []
    out: list[dict] = []
    for raw in links:
        if not isinstance(raw, dict):
            continue
        href = str(raw.get("href", "")).strip()
        if not href:
            continue
        if href.startswith("/"):
            href = f"https://hh.ru{href}"
        if "/employer/" in href.lower():
            continue
        vid = _vacancy_id_from_url(href)
        if not vid:
            continue
        name = str(raw.get("name", "")).strip()
        if not name or re.search(r"найдено\s+\d+\s+ваканс", name, re.IGNORECASE):
            continue
        out.append({"vacancy_id": vid, "url": _canonical_vacancy_url(href), "name": name})
    unique: dict[str, dict] = {}
    for item in out:
        unique[item["vacancy_id"]] = item
    return list(unique.values())


def _rank_candidates_with_llm(llm: LLMClient, query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    payload = {"query": query, "candidates": [{"id": c["vacancy_id"], "title": c["name"][:120]} for c in candidates[:15]]}
    response = llm.chat_text(
        "Rank vacancy candidates for AI Engineer relevance. Return JSON array of candidate ids in best-first order.",
        json.dumps(payload, ensure_ascii=True),
        temperature=0.0,
        max_tokens=int(os.getenv("LM_RANK_MAX_TOKENS", "120")),
    )
    try:
        parsed = json.loads(response)
        ids = parsed if isinstance(parsed, list) else parsed.get("ids", [])
        if not isinstance(ids, list):
            return candidates
        rank_map = {str(v): i for i, v in enumerate(ids)}
        return sorted(candidates, key=lambda c: rank_map.get(c["vacancy_id"], 999))
    except Exception:  # noqa: BLE001
        return candidates


def _safe_console(value: object) -> str:
    return str(value).encode("cp1251", errors="replace").decode("cp1251", errors="replace")


def format_structured_observation(obs: object) -> str:
    compact_structure = _compact_page_structure(obs.page_structure)
    raw = json.dumps(compact_structure, ensure_ascii=True)
    if len(raw) > _STRUCTURE_JSON_LIMIT:
        raw = raw[: _STRUCTURE_JSON_LIMIT - 24] + "...[structure truncated]"
    dom_summary = str(obs.dom_summary or "")
    if len(dom_summary) > _OBS_DOM_SUMMARY_LIMIT:
        dom_summary = dom_summary[: _OBS_DOM_SUMMARY_LIMIT - 22] + "\n...[observation truncated]"
    return (
        f"URL: {obs.url}\n"
        f"Title: {obs.title}\n"
        f"Action hints: {', '.join(obs.available_actions[:_OBS_ACTION_HINT_LIMIT]) if obs.available_actions else 'none'}\n"
        f"Page structure: {raw}\n"
        f"Page observation (compact):\n{dom_summary}"
    )


def _print_banner(task: str, start_url: Optional[str], max_steps: int, headless: bool, route_source: str) -> None:
    line = "=" * 72
    print(line)
    print(" Browser Agent - autonomous observe -> plan -> act loop")
    print(line)
    print(f" Task: {task}")
    if start_url:
        print(f" Start URL: {start_url} ({route_source})")
    print(f" Max steps: {max_steps} | Headless: {headless}")
    print(line + "\n")


def _print_step_plan(step: int, thought: str, action: ToolAction, verbose: bool) -> None:
    print(f"\n{'-' * 72}")
    print(f"[step {step}] model thought:\n{thought or '(none)'}")
    print(f"[step {step}] planned tool: {action.tool}")
    if verbose:
        print(f"[step {step}] tool args: {json.dumps(action.args, ensure_ascii=True)}")
        print(f"[step {step}] reason: {action.reason or '-'}")
    else:
        if action.args:
            print(f"[step {step}] tool args (summary): {json.dumps(action.args, ensure_ascii=True)[:400]}{'...' if len(json.dumps(action.args)) > 400 else ''}")
    print(f"{'-' * 72}")


def run_agent(
    task: str,
    start_url: Optional[str],
    max_steps: int,
    headless: bool,
    profile_dir: Optional[str] = None,
    pause_for_login: bool = False,
    verbose: bool = False,
    reflect_on_error: bool = True,
    allow_manual_login: bool = True,
) -> str:
    route_source = "cli_override"
    if not start_url:
        route = infer_start_url(task)
        start_url = route.start_url
        route_source = route.source
    _print_banner(task, start_url, max_steps, headless, route_source)

    llm = LLMClient()
    planner = Planner(llm=llm)
    reflector = ErrorReflector(llm=llm)
    safety_evaluator = SafetyEvaluator(llm=llm)
    memory = MemoryStore(max_events=200)
    browser = BrowserController(headless=headless, profile_dir=profile_dir)
    executor = Executor(browser=browser)
    debug_writer = RuntimeDebugWriter(enabled=True)
    loop_monitor = LoopMonitor(window=6)
    memory.start_task(task)
    intent = normalize_task_intent(task)
    flow_state = SearchFlowState()

    browser.start()
    try:
        if start_url:
            nav_result = executor.execute(ToolAction(tool="navigate", args={"url": start_url}), step=0)
            memory.add_text(0, "action_result", nav_result.message, success=nav_result.success)
            memory.record_action_with_progress(
                0,
                "navigate",
                f"initial route ({route_source})",
                nav_result.success,
                nav_result.message,
                useful_progress=nav_result.success,
            )
            if pause_for_login:
                print("[pause/debug] Manual login is allowed on auth-required states (no unconditional pause).")

        final_answer = "Stopped: max steps reached."
        stop_reason = "max_steps_reached"
        auth_prompted_for_url: Optional[str] = None
        search_like_task = is_search_like_task(task) or intent.is_search_like
        previous_observation_signature: Optional[str] = None

        for step in range(1, max_steps + 1):
            valid_before_step = len(flow_state.valid_vacancies)
            obs = browser.observe(step=step)
            obs_text = format_structured_observation(obs)
            memory.add_text(step, "observation", f"{obs.title} @ {obs.url}")
            memory.record_page_visit(obs.url)
            known_urls = {str(v.get("url", "")) for v in flow_state.valid_vacancies}
            known_titles = {str(v.get("title", "")).strip().lower() for v in flow_state.valid_vacancies}
            current_vacancy_id = _vacancy_id_from_url(obs.url)
            if flow_state.pending_vacancy_id and current_vacancy_id == flow_state.pending_vacancy_id:
                flow_state.stage = "process_vacancies"
            if flow_state.stage == "process_vacancies" and current_vacancy_id:
                if current_vacancy_id in flow_state.processed_vacancy_ids or current_vacancy_id in flow_state.rejected_vacancy_ids:
                    memory.add_text(step, "vacancy_skip_duplicate", current_vacancy_id)
                    flow_state.pending_vacancy_id = ""
                else:
                    validation = validate_vacancy_observation(obs, intent.search_query, known_urls, known_titles)
                    if validation.is_valid:
                        vacancy = {
                            **validation.structured,
                            "vacancy_id": current_vacancy_id,
                            "relevance_score": round(validation.relevance_score, 2),
                            "quality_score": round(validation.quality_score, 2),
                        }
                        flow_state.valid_vacancies.append(vacancy)
                        flow_state.valid_vacancies = sorted(
                            flow_state.valid_vacancies,
                            key=lambda x: float(x.get("relevance_score", 0.0)) + float(x.get("quality_score", 0.0)),
                            reverse=True,
                        )[:8]
                        flow_state.processed_vacancy_ids.add(current_vacancy_id)
                        flow_state.no_progress_steps = 0
                        flow_state.pending_vacancy_id = ""
                        memory.add_text(step, "vacancy_collected", vacancy.get("title", ""), data=vacancy)
                    else:
                        flow_state.rejected_vacancy_ids.add(current_vacancy_id)
                        flow_state.pending_vacancy_id = ""
                        flow_state.rejected_vacancies.append(
                            {
                                "url": validation.canonical_url,
                                "title": validation.normalized_title,
                                "reasons": validation.reasons[:4],
                            }
                        )
            if flow_state.stage in {"search_results", "process_vacancies"} and _is_search_results_page(obs.url, obs.title):
                candidates = _collect_search_candidates(obs)
                candidates = [c for c in candidates if c["vacancy_id"] not in flow_state.processed_vacancy_ids and c["vacancy_id"] not in flow_state.rejected_vacancy_ids]
                ranked = _rank_candidates_with_llm(llm, intent.search_query, candidates)
                flow_state.candidate_queue = ranked
                flow_state.search_results_url = obs.url
                if ranked:
                    flow_state.stage = "profile_capture" if not flow_state.profile_summary else "process_vacancies"
            if debug_writer.enabled:
                try:
                    browser.save_screenshot(str(debug_writer.step_file(step, "screen.png")))
                except Exception as screenshot_exc:  # noqa: BLE001
                    memory.add_text(step, "debug_warning", f"screenshot failed: {screenshot_exc}")

            auth_gate = detect_auth_gate(obs)
            if auth_prompted_for_url and obs.url != auth_prompted_for_url and not auth_gate.is_auth_gate:
                auth_prompted_for_url = None
            if allow_manual_login and auth_gate.is_auth_gate and auth_prompted_for_url != obs.url:
                print(
                    f"[auth] Detected auth-required state (confidence={auth_gate.confidence}; {auth_gate.rationale}). "
                    "Complete login in browser, then press Enter to continue..."
                )
                input()
                auth_prompted_for_url = obs.url
                obs = browser.observe(step=step)
                obs_text = format_structured_observation(obs)
                memory.add_text(
                    step,
                    "auth_resume",
                    "Manual login checkpoint completed; resuming autonomous flow.",
                    confidence=auth_gate.confidence,
                    rationale=auth_gate.rationale,
                )

            ranked_inputs = rank_primary_input(obs.page_structure, intent.search_query)
            ranked_buttons = rank_submit_buttons(obs.page_structure, ranked_inputs[0] if ranked_inputs else None)
            ranked_links = rank_vacancy_links(obs.page_structure)
            ranked_profile_links = rank_profile_links(obs.page_structure)
            if not flow_state.profile_summary:
                extracted_profile = extract_profile_summary(obs)
                if extracted_profile:
                    flow_state.profile_summary = extracted_profile[:1200]
                    memory.add_text(step, "profile_summary_captured", flow_state.profile_summary[:220])
                    if flow_state.stage == "profile_capture":
                        flow_state.stage = "search_results"
            instrumentation_payload = {
                "url": obs.url,
                "title": obs.title,
                "task_intent": {
                    "search_query": intent.search_query,
                    "target_domain": intent.target_domain,
                    "is_search_like": intent.is_search_like,
                },
                "top_inputs": _short_candidates(ranked_inputs),
                "top_buttons": _short_candidates(ranked_buttons),
                "top_links": _short_candidates(ranked_links),
                "top_profile_links": _short_candidates(ranked_profile_links),
                "flow_state": {
                    "stage": flow_state.stage,
                    "query_typed": flow_state.query_typed,
                    "search_submitted": flow_state.search_submitted,
                    "profile_checked": flow_state.profile_checked,
                    "vacancies_collected": len(flow_state.valid_vacancies),
                    "vacancies_rejected": len(flow_state.rejected_vacancies),
                    "candidate_queue": len(flow_state.candidate_queue),
                    "profile_summary_present": bool(flow_state.profile_summary),
                },
            }
            debug_writer.write_json(step, "instrumentation.json", instrumentation_payload)
            debug_writer.write_text(step, "observation.txt", obs_text)

            search_ui_present = detect_search_ui(obs)
            deterministic_action: Optional[ToolAction] = None
            deterministic_thought = ""
            if search_like_task and flow_state.stage == "bootstrap" and ranked_inputs and not flow_state.query_typed:
                if len(flow_state.failed_input_targets) >= 3:
                    deterministic_action = ToolAction(
                        tool="navigate",
                        args={"url": f"https://hh.ru/search/vacancy?text={intent.search_query.replace(' ', '+')}"},
                        reason="fallback route refinement after repeated input targeting failures",
                    )
                    deterministic_thought = "Input targeting repeatedly failed; navigate directly to search results route."
                
                best = None
                if deterministic_action is None:
                    for candidate in ranked_inputs:
                        sig = f"{candidate.role}:{candidate.name}"
                        if sig not in flow_state.failed_input_targets:
                            best = candidate
                            break
                    if best is None:
                        best = ranked_inputs[0]
                    deterministic_action = ToolAction(
                        tool="type",
                        args={"role": best.role, "name": best.name, "text": intent.search_query},
                        reason=f"search bootstrap: primary input by rank ({', '.join(best.reasons[:2]) or 'score'})",
                    )
                    deterministic_thought = "Deterministic search bootstrap: type normalized query into primary input."
            elif search_like_task and flow_state.stage == "bootstrap" and flow_state.query_typed and not flow_state.search_submitted and ranked_buttons:
                best_btn = ranked_buttons[0]
                deterministic_action = ToolAction(
                    tool="click",
                    args={"role": "button", "name": best_btn.name},
                    reason=f"search submit by rank ({', '.join(best_btn.reasons[:2]) or 'score'})",
                )
                deterministic_thought = "Deterministic search submit: click ranked search button."
            elif search_like_task and flow_state.stage == "profile_capture" and not flow_state.profile_summary:
                if flow_state.profile_capture_failures >= 2:
                    flow_state.stage = "search_results"
                if "/applicant/resumes" not in obs.url:
                    deterministic_action = ToolAction(
                        tool="navigate",
                        args={"url": "https://hh.ru/applicant/resumes"},
                        reason="mandatory profile capture before personalized letters",
                    )
                    deterministic_thought = "Navigate to profile/resume area to capture personalization context."
                    flow_state.profile_attempted = True
                elif ranked_profile_links and flow_state.profile_capture_failures < 2:
                    profile_link = ranked_profile_links[0]
                    deterministic_action = ToolAction(
                        tool="click",
                        args={"role": "link", "name": profile_link.name},
                        reason="open best resume/profile link for richer context extraction",
                    )
                    deterministic_thought = "Open resume/profile link and capture structured profile context."
            elif search_like_task and flow_state.stage == "process_vacancies" and len(flow_state.valid_vacancies) < 3:
                candidate = next(
                    (
                        c for c in flow_state.candidate_queue
                        if c["vacancy_id"] not in flow_state.processed_vacancy_ids
                        and c["vacancy_id"] not in flow_state.rejected_vacancy_ids
                        and c["vacancy_id"] not in flow_state.seen_vacancy_ids
                    ),
                    None,
                )
                if candidate:
                    flow_state.seen_vacancy_ids.add(candidate["vacancy_id"])
                    flow_state.pending_vacancy_id = candidate["vacancy_id"]
                    deterministic_action = ToolAction(
                        tool="navigate",
                        args={"url": candidate["url"]},
                        reason=f"open candidate from stored search queue id={candidate['vacancy_id']}",
                    )
                    deterministic_thought = "Open next vacancy strictly from stored search-results queue."
                elif flow_state.search_results_url:
                    deterministic_action = ToolAction(
                        tool="navigate",
                        args={"url": flow_state.search_results_url},
                        reason="queue exhausted; return to stored search results page",
                    )
                    deterministic_thought = "Rebuild candidate queue from search-results page (no page-local mining)."
            elif search_like_task and flow_state.stage == "search_results" and not _is_search_results_page(obs.url, obs.title):
                deterministic_action = ToolAction(
                    tool="navigate",
                    args={"url": flow_state.search_results_url or f"https://hh.ru/search/vacancy?text={intent.search_query.replace(' ', '+')}"},
                    reason="return to canonical search-results page before candidate acquisition",
                )
                deterministic_thought = "Navigate to search results page to build candidate queue."

            planning_hints = (
                f"task_search_like={search_like_task}; "
                f"search_ui_present={search_ui_present}; "
                "if repeated scrolling yields no page-state change, avoid another scroll and switch strategy."
            )
            if deterministic_action:
                plan = None
            else:
                plan = planner.plan(
                    task=task,
                    observation_text=obs_text,
                    memory=memory,
                    planning_hints=planning_hints,
                )

            if plan and plan.done and plan.final_response is not None:
                final_answer = str(plan.final_response).strip() or "Task complete."
                stop_reason = "model_done_final_response"
                memory.add_text(step, "completion", final_answer, via="done_flag")
                print(f"[step {step}] completion: model returned done=true with final_response (no tool run)")
                if verbose:
                    print(f"[step {step}] final_response: {final_answer}")
                break

            if plan and not plan.steps:
                if plan.done:
                    final_answer = (plan.final_response or "").strip() if plan.final_response else "Task complete."
                    if not final_answer:
                        final_answer = "Task complete."
                    stop_reason = "model_done_empty_steps"
                    memory.add_text(step, "completion", final_answer, via="done_empty_steps")
                    print(f"[step {step}] completion: done=true with empty steps")
                    if verbose:
                        print(f"[step {step}] final_response: {final_answer}")
                    break
                memory.add_text(step, "planner_error", "No plan steps returned")
                memory.record_action(step, "planner", "empty plan", False, "No plan steps returned")
                print(f"[step {step}] planner returned no steps; continuing...")
                continue

            if deterministic_action:
                next_step = PlanStep(thought=deterministic_thought, action=deterministic_action)
            else:
                next_step = plan.steps[0]
            if next_step.action.tool == "type":
                text_value = str(next_step.action.args.get("text", ""))
                target_name = str(next_step.action.args.get("name", "")).lower()
                if len(text_value) > 90 or text_value.strip().lower() == task.strip().lower():
                    next_step.action.args["text"] = intent.search_query
                    next_step.action.reason = "normalized query guard: avoid dumping full task into input"
                if any(bad in target_name for bad in ("исключить", "exclude", "зарплат", " от", " до", "region", "город")) and ranked_inputs:
                    best = ranked_inputs[0]
                    next_step.action.args["role"] = best.role
                    next_step.action.args["name"] = best.name
                    next_step.action.args["text"] = intent.search_query
                    next_step.action.reason = "input ranking guard: redirect typing into primary search input"
            if next_step.action.tool == "click":
                click_name = str(next_step.action.args.get("name", "")).lower()
                click_role = str(next_step.action.args.get("role", "")).lower()
                if click_role == "button" and any(bad in click_name for bad in ("создать", "create", "new", "add", "chat", "опубликовать")) and ranked_buttons:
                    best_btn = ranked_buttons[0]
                    next_step.action.args["role"] = "button"
                    next_step.action.args["name"] = best_btn.name
                    next_step.action.reason = "button ranking guard: avoid unrelated global action button"
            if next_step.action.tool == "navigate":
                nav_url = str(next_step.action.args.get("url", ""))
                nav_vid = _vacancy_id_from_url(nav_url)
                if "/employer/" in nav_url.lower():
                    next_step.action = ToolAction(
                        tool="navigate",
                        args={"url": flow_state.search_results_url or f"https://hh.ru/search/vacancy?text={intent.search_query.replace(' ', '+')}"},
                        reason="guard: employer pages are forbidden as vacancy candidates",
                    )
                    next_step.thought = "Guard override: reject employer page and return to search results."
                elif nav_vid and (nav_vid in flow_state.processed_vacancy_ids or nav_vid in flow_state.rejected_vacancy_ids or nav_vid in flow_state.seen_vacancy_ids):
                    if nav_vid != flow_state.pending_vacancy_id:
                        next_step.action = ToolAction(
                            tool="navigate",
                            args={"url": flow_state.search_results_url or f"https://hh.ru/search/vacancy?text={intent.search_query.replace(' ', '+')}"},
                            reason=f"guard: repeated vacancy id {nav_vid} rejected; rebuild queue",
                        )
                        next_step.thought = "Guard override: repeated vacancy id rejected; return to search results."

            # Deterministic search orchestration before planner drift.
            if search_like_task and search_ui_present and next_step.action.tool == "scroll":
                forced = choose_stuck_override(task=task, observation=obs, normalized_query=intent.search_query)
                next_step.action = forced.action
                next_step.thought = (
                    (next_step.thought + " | ") if next_step.thought else ""
                ) + f"search-first override: {forced.reason}"
                memory.add_text(
                    step,
                    "strategy_override",
                    "Search-like task with visible search UI: forced non-scroll action.",
                    override_reason=forced.reason,
                )
            memory.add_text(step, "thought", next_step.thought)
            _print_step_plan(step, next_step.thought, next_step.action, verbose)
            debug_writer.write_json(
                step,
                "chosen_action.json",
                {
                    "action": {"tool": next_step.action.tool, "args": next_step.action.args, "reason": next_step.action.reason},
                    "why": next_step.thought,
                },
            )

            safety = safety_evaluator.evaluate(
                task=task,
                observation=obs,
                action=next_step.action,
                memory=memory,
            )
            print(
                f"[step {step}] safety: route={safety.route} verdict={safety.verdict} source={safety.source} "
                f"rationale={safety.rationale}"
            )
            memory.add_text(
                step,
                "safety_evaluation",
                f"verdict={safety.verdict}: {safety.rationale}",
                source=safety.source,
                tool=next_step.action.tool,
            )

            if safety.verdict == "block":
                result = ExecutionResult(
                    False,
                    "Blocked by safety layer",
                    details={
                        "tool": next_step.action.tool,
                        "stop_reason": "safety_blocked",
                        "safety_rationale": safety.rationale,
                        "safety_source": safety.source,
                    },
                )
            elif safety.verdict == "confirm":
                allowed = executor.confirm_action(next_step.action, rationale=safety.rationale)
                if not allowed:
                    result = ExecutionResult(
                        False,
                        "Blocked by user safety confirmation",
                        details={
                            "tool": next_step.action.tool,
                            "stop_reason": "security_confirmation_denied",
                            "safety_rationale": safety.rationale,
                            "safety_source": safety.source,
                        },
                    )
                else:
                    result = executor.execute(next_step.action, step=step)
            else:
                result = executor.execute(next_step.action, step=step)
            memory.add_text(
                step,
                "action_result",
                result.message,
                success=result.success,
                tool=next_step.action.tool,
            )
            stagnation_signal = loop_monitor.record(next_step.action, obs)
            current_obs_signature = _observation_signature(obs)
            useful_progress = bool(
                result.success
                and (
                    next_step.action.tool != "scroll"
                    or obs.url != (memory.visited_pages[-2] if len(memory.visited_pages) > 1 else "")
                    or current_obs_signature != previous_observation_signature
                )
            )
            if stagnation_signal.is_stuck and next_step.action.tool == "scroll":
                useful_progress = False
            memory.record_action_with_progress(
                step,
                next_step.action.tool,
                next_step.action.reason,
                result.success,
                result.message,
                useful_progress=useful_progress,
            )
            previous_observation_signature = current_obs_signature
            if result.success and next_step.action.tool == "type":
                if str(next_step.action.args.get("text", "")).strip() == intent.search_query:
                    flow_state.query_typed = True
            if (not result.success) and next_step.action.tool == "type":
                fail_sig = f"{next_step.action.args.get('role','')}:{next_step.action.args.get('name','')}"
                flow_state.failed_input_targets.add(fail_sig)
            if (not result.success) and next_step.action.tool == "navigate":
                pending = flow_state.pending_vacancy_id
                if pending:
                    flow_state.rejected_vacancy_ids.add(pending)
                    flow_state.pending_vacancy_id = ""
            if (not result.success) and flow_state.stage == "profile_capture" and next_step.action.tool in {"click", "navigate"}:
                flow_state.profile_capture_failures += 1
            if (not result.success) and next_step.action.tool == "click" and next_step.action.args.get("role") == "link":
                flow_state.failed_link_targets.add(str(next_step.action.args.get("name", "")))
            if result.success and next_step.action.tool == "click":
                name_lower = str(next_step.action.args.get("name", "")).lower()
                if any(token in name_lower for token in ("поиск", "найти", "search")):
                    flow_state.search_submitted = True
                    if flow_state.stage == "bootstrap":
                        flow_state.stage = "search_results"
                if any(token in name_lower for token in ("резюме", "профиль", "resume", "profile", "cv")):
                    flow_state.profile_checked = True
                if next_step.action.args.get("role") == "link":
                    flow_state.opened_links.add(str(next_step.action.args.get("name", "")))
                    if flow_state.stage == "profile_capture":
                        flow_state.profile_capture_failures = 0
            if stagnation_signal.is_stuck:
                memory.add_text(
                    step,
                    "stagnation_detected",
                    "Loop monitor detected stagnation; strategy override will be preferred.",
                    reason=stagnation_signal.reason,
                    repeated_action=stagnation_signal.repeated_action_count,
                    same_url=stagnation_signal.same_url_count,
                    same_state=stagnation_signal.same_state_count,
                )
                print(f"[stagnation] {_safe_console(stagnation_signal.reason)}")
                if next_step.action.tool == "scroll":
                    forced = choose_stuck_override(task=task, observation=obs, normalized_query=intent.search_query)
                    print(f"[stagnation] Prepared next-step strategy override: {forced.reason}")
                elif search_like_task and ranked_buttons:
                    alt = ranked_buttons[0]
                    memory.add_text(
                        step,
                        "stagnation_strategy",
                        "Forcing submit-click strategy on next loop.",
                        button=alt.name,
                    )

            print(f"[step {step}] executor: success={result.success} | {_safe_console(result.message)}")
            print(f"[progress] {memory.progress_summary()}")

            if len(flow_state.valid_vacancies) > valid_before_step:
                flow_state.no_progress_steps = 0
            else:
                flow_state.no_progress_steps += 1

            if (
                flow_state.stage == "process_vacancies"
                and not flow_state.pending_vacancy_id
                and flow_state.no_progress_steps > 2
                and flow_state.search_results_url
            ):
                memory.add_text(step, "recovery", "No new valid vacancy; returning to search results for queue rebuild.")
                flow_state.stage = "search_results"

            if not result.success:
                if verbose:
                    print(f"[step {step}] error detail: {_safe_console(result.message)}")
                    if result.details:
                        print(f"[step {step}] error details: {json.dumps(result.details, ensure_ascii=True)}")
                stop_reason = str(result.details.get("stop_reason", "")) if result.details else ""
                if reflect_on_error and not stop_reason.startswith("safety_") and stop_reason != "security_confirmation_denied":
                    hint = reflector.reflect(
                        task=task,
                        observation_excerpt=obs_text,
                        action=next_step.action,
                        error_message=result.message,
                    )
                    memory.add_text(step, "recovery_hint", hint, from_subagent="error_reflector")
                    print(f"[sub-agent error_reflector]\n{hint}\n")

            if next_step.action.tool == "finish_task" and result.success:
                final_answer = (
                    next_step.action.args.get("message")
                    or (plan.final_response if plan else None)
                    or "Task complete."
                )
                stop_reason = "finish_task"
                if verbose:
                    print(f"[step {step}] finish_task message: {final_answer}")
                break

            clean_vacancies = [v for v in flow_state.valid_vacancies if _has_meaningful_vacancy(v)]
            if search_like_task and len(clean_vacancies) >= 3 and flow_state.profile_attempted and not is_profile_summary_meaningful(flow_state.profile_summary):
                final_answer = (
                    "Vacancies collected but personalization context is incomplete.\n"
                    "Profile/resume summary could not be captured after retry. "
                    "Run requires manual profile access confirmation before final personalized application drafts."
                )
                stop_reason = "incomplete_personalization_context"
                break

            if search_like_task and len(clean_vacancies) >= 3 and is_profile_summary_meaningful(flow_state.profile_summary):
                top_vacancies = clean_vacancies[:3]
                letters = _build_cover_letters(
                    llm=llm,
                    profile_summary=flow_state.profile_summary,
                    vacancies=top_vacancies,
                    task=task,
                )
                if letters and "No vacancies were collected." not in letters:
                    final_answer = (
                        "Collected 3 validated relevant vacancies.\n\n"
                        f"Vacancies:\n{json.dumps(top_vacancies, ensure_ascii=False, indent=2)}\n\n"
                        f"Profile summary:\n{flow_state.profile_summary[:900]}\n\n"
                        f"Personalized cover letters:\n{letters}"
                    )
                    stop_reason = "vacancies_and_letters_ready"
                    break

        print(f"\n{'=' * 72}")
        print(f"[agent_stop] reason={stop_reason}")
        print(f"{'=' * 72}")
        return final_answer
    finally:
        browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Playwright browser agent.")
    parser.add_argument("--task", default=None, help="Task for the browser agent. If omitted, interactive prompt is used.")
    parser.add_argument("--start-url", default=None, help="Optional starting URL override (debug/developer)")
    parser.add_argument("--max-steps", type=int, default=20, help="Max agent iterations")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode (not recommended for demo)")
    parser.add_argument("--profile-dir", default=None, help="Path to persistent Chromium user profile directory")
    parser.add_argument(
        "--pause-for-login",
        action="store_true",
        help="Debug override: force manual login pause right after initial navigation",
    )
    parser.add_argument(
        "--no-manual-login",
        action="store_true",
        help="Disable auth-on-demand login pause and keep full autonomous mode",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full tool arguments and longer logs (good for screen recording)",
    )
    parser.add_argument(
        "--no-reflect",
        action="store_true",
        help="Disable error-reflector sub-agent after failed actions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task = (args.task or "").strip()
    if not task:
        task = input("What should the browser agent do? ").strip()
    if not task:
        print("Task is required.", file=sys.stderr)
        sys.exit(2)
    try:
        answer = run_agent(
            task=task,
            start_url=args.start_url,
            max_steps=args.max_steps,
            headless=args.headless,
            profile_dir=args.profile_dir,
            pause_for_login=args.pause_for_login,
            verbose=args.verbose,
            reflect_on_error=not args.no_reflect,
            allow_manual_login=not args.no_manual_login,
        )
    except KeyboardInterrupt:
        print("\n[agent_stop] interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    print(_safe_console(f"\n--- Final report ---\n{answer}\n"))
