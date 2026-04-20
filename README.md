# Browser Agent

Автономный AI-агент для **видимого** браузера: Python, Playwright, цикл *observe → plan → act* с памятью, сжатием контекста и подтверждением опасных кликов.

## Возможности (соответствие ТЗ)

- Программное управление Chromium через Playwright; **по умолчанию окно видно** (`headless` только по флагу).
- Ввод задачи из **CLI**; агент сам планирует шаги и вызывает инструменты (`navigate`, `click`, `type`, `scroll`, `wait`, `back`, `finish_task`).
- **Persistent session**: каталог профиля `--profile-dir` и пауза для ручного входа `--pause-for-login`.
- **Контекст**: в LLM уходит сжатое наблюдение — ARIA snapshot + обобщённый DOM-скан (без «скармливания» сырой страницы целиком); лимиты через `OBS_TEXT_CHAR_LIMIT` / `OBS_STRUCTURE_JSON_LIMIT`.
- **Security layer**: перед потенциально деструктивным `click` запрашивается подтверждение в терминале.
- **Продвинутый паттерн**: субагент **`error_reflector`** — после неудачного действия короткая подсказка по восстановлению (отключается флагом `--no-reflect`).
- Без привязки к конкретному сайту: взаимодействие через **role + accessible name** (Playwright `get_by_role` / fallback `get_by_label`).

## Быстрый старт

```bash
cd browser-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
copy .env.example .env
```

Настройте в `.env` или в системе переменные `LM_BASE_URL`, `LM_API_KEY`, `LM_MODEL` (см. пример).

### LM Studio

1. Запустите LM Studio → вкладка **Developer** → поднимите сервер, например `http://127.0.0.1:1234/v1`.
2. Загрузите инструкт-модель с устойчивым JSON (например семейство Qwen2.5-Instruct).

### Проверка без LLM

```bash
python smoke_test.py
```

## Запуск агента

**Не передавайте `--headless`**, если готовите демо-видео — проверяющему должен быть виден браузер.

```bash
python -m browser_agent.agent.main --task "YOUR TASK" --max-steps 25
```

Опционально для debug/dev:

```bash
python -m browser_agent.agent.main --task "YOUR TASK" --start-url "https://example.com" --pause-for-login --profile-dir ".profiles/demo"
```

Подробные логи шага (мысль модели, инструмент, аргументы):

```bash
python -m browser_agent.agent.main --verbose --task "..."
```

Профиль и ручной логин перед автономными шагами:

```bash
python -m browser_agent.agent.main --verbose --task "..." --start-url "https://example.com/login" --profile-dir ".profiles/demo" --pause-for-login
```

## Рекомендуемый демо-сценарий (стабильный, без оплаты)

Цель: показать навигацию, клики, ввод, скролл и итоговый отчёт на нейтральном сайте.

1. Откройте терминал и браузер рядом (разрешение экрана, чтобы в кадре попали оба окна).
2. Запустите (подставьте свою модель в `.env`):

```bash
python -m browser_agent.agent.main --verbose --max-steps 18 --start-url "https://playwright.dev" --task "Open the Playwright documentation site, find and open the Python section or Python-related docs page, then briefly summarize what Playwright is in 2-3 sentences in Russian."
```

3. Дождитесь завершения или шага `finish_task` в логах.
4. В консоли появится блок `--- Final report ---` — его должно быть видно на записи.

Если модель слабо следует JSON-схеме, увеличьте `LM_JSON_RETRIES` в окружении.

### Как записать короткое видео для сдачи

1. **OBS Studio**, **ShareX** или встроенная запись Windows (**Win+G**).
2. Кадр: слева/сверху — терминал с флагом `--verbose`, справа/снизу — окно Chromium.
3. В начале кадра запустите одну команду из раздела выше; не редактируйте видео так, чтобы скрывать реальные логи инструментов.
4. 60–120 секунд обычно достаточно: старт команды → несколько шагов `[step N]` → финальный отчёт.
5. Залейте видео (YouTube unlisted / Google Drive / др.) и приложите ссылку вместе с **ссылкой на этот репозиторий**.

## Переменные окружения

См. `.env.example`. Дополнительно:

- `BROWSER_EXECUTABLE_PATH` или `CHROMIUM_EXECUTABLE_PATH` — путь к системному Chrome/Chromium, если не хотите качать браузер через `playwright install`.

## Ограничения

- Качество планов сильно зависит от модели и от JSON-ответов.
- Сложные SPA / теневой DOM / нестандартные виджеты могут требовать больше шагов или ручного входа.
- Ключевые слова деструктивных действий для security layer эвристические — возможны ложные срабатывания.
- Полный e2e на произвольном внешнем сайте зависит от сети, гео-блокировок и изменений вёрстки.

## Структура

- `browser_agent/agent/main.py` — точка входа, цикл агента, логирование.
- `browser_agent/agent/browser.py` — Playwright, наблюдение страницы.
- `browser_agent/agent/planner.py` / `llm.py` — запрос к модели, валидация JSON.
- `browser_agent/agent/executor.py` — выполнение инструментов, security.
- `browser_agent/agent/error_reflector.py` — субагент подсказок после ошибки.
- `browser_agent/agent/memory.py` — память для промпта.

## Лицензия

Укажите лицензию по необходимости перед публикацией на GitHub.
