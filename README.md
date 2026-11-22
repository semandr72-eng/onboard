# onboard

[![CI](https://github.com/semandr72-eng/onboard/actions/workflows/ci.yml/badge.svg)](https://github.com/semandr72-eng/onboard/actions/workflows/ci.yml)

AI-наставник для обучения сотрудников новому материалу и автоматического тестирования.
Два канала доступа: Telegram-бот и веб-чат (мини-сайт). Работает на DeepSeek через OpenAI-совместимый API,
отвечает на основе базы знаний (RAG: pgvector + полнотекстовый поиск), обезличивает персональные
данные перед отправкой во внешний LLM, запрашивает согласие на обработку ПДн (152-ФЗ)
и поддерживает права субъекта: выгрузку и удаление данных.

AI-наставник:
- объясняет материал простыми сообщениями на основе релевантных фрагментов базы знаний;
- отвечает на вопросы сотрудника строго по загруженным документам;
- сам решает, когда можно переходить к тесту;
- задает вопросы по одному;
- проверяет ответы по материалу;
- сохраняет результат теста в Postgres.

## Как устроено

```
Telegram -> bot (aiogram) --+
                             +-> AITrainingService -> Anonymizer -> LLMGateway -> DeepSeek
Браузер -> web (FastAPI) ----+         |   ^
                                       |   | RAGService (hybrid search, RRF)
                                       v   |
                              ConsentService + pgvector: knowledge_chunks (семантика + FTS)
                              PersonalDataService      knowledge_documents
                                       ^
                              invite_codes (вход по коду приглашения)
```

Оба канала используют одно сервис-ядро. Состояние сессии — FSM (Telegram) или память с TTL 24 ч (веб).

**Единая модель субъекта данных:** везде вместо «telegram user id» используется
`subject_id` вида `tg:<id>` или `web:<uuid>` плюс канал (`telegram` / `web`).

### Компоненты

1. **RAG** (`services/rag_service.py`) — гибридный поиск по базе знаний: векторная близость
   (pgvector, косинусная) + полнотекстовый поиск PostgreSQL (`russian`), склейка рангов через RRF.
   Эмбеддинги считаются локально (`fastembed`, `multilingual-e5-large`, префиксы
   `passage:`/`query:`) — наружу не уходит ничего. Запрос на поиск строится из сообщения
   сотрудника и текущего вопроса теста. Если база пуста — фолбэк на статический `TRAINING_MATERIAL`.
2. **Ingestion** (`services/knowledge_ingest.py`) — загрузка документов через админ-API.
   Понимает YAML-frontmatter (метаданные: тип документа, роль, версия) и разбивку по
   маркерам «# ЧАНК N. ...»; для обычных markdown-документов — fallback на заголовки `##`.
3. **LLM Gateway** (`services/llm_gateway.py`) — клиент к OpenAI-совместимому API DeepSeek.
   `response_format=json_schema` с автоматическим откатом на `json_object`; один повтор
   при невалидном JSON.
4. **Обезличивание** (`services/anonymizer.py`) — ФИО, телефоны, e-mail и ники заменяются
   на плейсхолдеры до отправки во внешний LLM. Имя сотрудника в промпт не попадает никогда.
   Маппинг живёт только в памяти процесса; ответ подставляется обратно локально.
5. **Согласие и права субъекта** (`ConsentService`, `PersonalDataService`) — версионируемое
   согласие, `/export` и `/delete` в боте, retention по `RETENTION_DAYS`.
6. **Аудит** (`audit_log`) — обращения к LLM (с числом RAG-чанков и статистикой обезличивания),
   согласия, выгрузки и удаления. Без значений ПДн.
7. **Веб-канал** (`web/`, FastAPI) — чат в браузере, вход по одноразовым кодам приглашения.
8. **Аналитика для HR** (`training_answers` + admin API) — каждый проверенный ответ сохраняется
   с текстом вопроса, верностью и привязкой к чанку базы знаний (семантическая атрибуция).
   Отчёты: результаты с фильтрами, сводная статистика, «пробелы знаний» (вопросы с самой
   низкой долей верных ответов), CSV-выгрузка. HR видит не «все прошли курс», а где именно
   сотрудники ошибаются и какой фрагмент регламента стоит переписать.
9. **Контроль качества** (`services/quality_service.py`, `QualityJudge`) — независимый судья
   на том же LLM Gateway: (1) пересматривает оценки ответов теста по тройке «вопрос + ответ +
   материал» — при расхождении вердикт судьи приоритетен, сотрудник видит «Ответ засчитан
   после пересмотра»; (2) faithfulness-проверка каждой реплики наставника: выдуманные факты
   фиксируются. Все проверки пишутся в `quality_checks`, провал — алерт `quality.warn` в аудит,
   сводка — в `/api/admin/quality`. Отключается флагом `QUALITY_CONTROL=false`.
   Судья получает только обезличенный текст — периметр ПДн не расширяется.

Ограничение обезличивания: распознаются известные ФИО, ФИО с отчеством, телефоны,
e-mail и ники; произвольные пары «Имя Фамилия» без отчества требуют NER (следующий шаг).

## Веб-канал: API

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/` | одностраничный чат |
| POST | `/api/session` | начать сессию по коду приглашения (cookie HttpOnly) |
| GET | `/api/state` | текущее состояние сессии (возобновление) |
| POST | `/api/consent` | согласие (true/false) |
| POST | `/api/name` | имя сотрудника, старт обучения |
| POST | `/api/message` | сообщение в диалог |
| POST | `/api/logout` | завершить сессию |
| POST | `/api/admin/codes` | создать код приглашения (Bearer `ADMIN_API_TOKEN`) |
| GET | `/api/admin/codes` | список кодов |
| POST | `/api/admin/knowledge` | загрузить документ в базу знаний `{source, content}` |
| GET | `/api/admin/knowledge` | список документов с числом чанков |
| DELETE | `/api/admin/knowledge/{id}` | удалить документ и его чанки |
| GET | `/api/admin/results` | результаты тестов с фильтрами `date_from`, `date_to`, `channel`, `min_score` |
| GET | `/api/admin/stats` | сводка: всего сессий, средний балл, по каналам и по дням (30 дней) |
| GET | `/api/admin/gaps` | «пробелы знаний»: вопросы с самой низкой долей верных ответов |
| GET | `/api/admin/quality` | контроль качества: pass rate проверок faithfulness/grade + последние провалы |
| GET | `/api/admin/export/results.csv` | CSV-выгрузка результатов |

Загрузка документа в базу знаний:

```bash
curl -X POST http://localhost:8000/api/admin/knowledge \
  -H "Authorization: Bearer $ADMIN_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"source": "sbor_order_picking.md", "content": "---\n..."}'
```

## Что умеет продукт

- запросить согласие на обработку персональных данных (152-ФЗ);
- отвечать и тестировать на основе загруженных документов (RAG), а не фиксированного текста;
- работать в Telegram и в браузере (вход по коду приглашения);
- обезличить персональные данные перед отправкой в DeepSeek;
- выгрузить данные сотрудника (`/export`) и удалить их (`/delete`);
- удалять результаты старше `RETENTION_DAYS`;
- вести аудит значимых событий.

## Структура данных

Таблица `training_results`:
- `subject_id` (`tg:<id>` / `web:<uuid>`), `channel` (`telegram` / `web`)
- `employee_name`, `topic`
- `total_questions`, `correct_answers`, `score_percent`, `final_summary`
- `created_at`

Таблица `consents`:
- `subject_id`, `version`, `text_sha256`
- `status` (`granted` / `revoked`), `created_at`, `revoked_at`

Таблица `training_answers`:
- `result_id` (FK на `training_results`, каскадное удаление)
- `question` — текст вопроса теста
- `is_correct`, `feedback`
- `chunk_title` — к какому чанку базы знаний отнесен вопрос (семантическая атрибуция)
- `created_at`

Таблица `quality_checks`:
- `kind` (`faithfulness` — достоверность реплик наставника | `grade` — оценка ответа судьёй)
- `passed`, `subject_id`, `detail_json` (вердикт, обоснование, неподтверждённые утверждения)
- `created_at`

Таблица `audit_log`:- `action` (`llm.request`, `consent.granted`, `data.exported`, `data.deleted`, ...)
- `status` (`ok` / `error`), `subject_id`
- `meta_json` — модель, sha256 промпта, статистика обезличивания, число RAG-чанков, токены
- `created_at`

Таблицы `knowledge_documents` / `knowledge_chunks`:
- документ: `source`, `meta_json` (frontmatter), `chunk_count`
- чанк: `document_id`, `ordinal`, `title`, `content`, `embedding vector(1024)`
- GIN-индекс полнотекстового поиска (`russian`) по `content`

Таблица `invite_codes`:
- `code`, `max_uses`, `used_count`, `active`, `expires_at`, `created_at`

## Переменные окружения

```env
BOT_TOKEN=your_telegram_bot_token
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/onboarding
LLM_API_KEY=your_deepseek_api_key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
EMBEDDING_MODEL=intfloat/multilingual-e5-large
KNOWLEDGE_TOP_K=4
QUALITY_CONTROL=true
TRAINING_TOPIC=Регламент работы команды
TRAINING_MATERIAL=...  # фолбэк, если база знаний пуста
TRAINING_MATERIAL_FILE=./material.txt
QUIZ_QUESTION_COUNT=5
RETENTION_DAYS=180
PRIVACY_NOTICE_VERSION=1.0
ADMIN_API_TOKEN=change-me
LOG_LEVEL=INFO
```

Если `LLM_BASE_URL` и `LLM_MODEL` не заданы, используется официальный API DeepSeek. Подойдет любой
OpenAI-совместимый эндпоинт, включая self-hosted vLLM/Ollama. Первый запуск скачивает модель
эмбеддингов (~1.2 ГБ) в локальный кэш.

## Локальный запуск

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env

# бот
python main.py

# веб-канал (второй терминал)
python -m uvicorn web.app:create_app --factory --host 0.0.0.0 --port 8000
```

## Запуск через Docker

```powershell
docker compose up --build
# бот + веб на http://localhost:8000, Postgres с pgvector
```

## Развёртывание on-prem (закрытый контур)

Всё работает в сети заказчика без единого внешнего API: LLM — локальная
(DeepSeek-R1-Distill через Ollama, OpenAI-совместимый интерфейс уже поддержан
через `LLM_BASE_URL`), эмбеддинги — локальная модель (fastembed), БД — Postgres
с pgvector. Персональные данные обезличиваются перед отправкой в LLM как обычно.

**Требования:** Docker с compose-плагином; GPU необязателен, но для
DeepSeek-R1:8b на CPU ответы будут медленными (игровая видеокарта 8+ ГБ VRAM
или CPU с терпением — см. размеры моделей Ollama).

```powershell
Copy-Item .env.example .env
# задать в .env: ADMIN_API_TOKEN (обязательно), BOT_TOKEN,
# LLM_API_KEY (для Ollama — любое непустое значение)

docker compose -f docker-compose.onprem.yml up -d --build
docker compose -f docker-compose.onprem.yml exec llm ollama pull deepseek-r1:8b
```

Миграции БД применяются автоматически при старте контейнеров. Веб-канал —
на `http://<сервер>:8000`, Telegram-бот начинает отвечать после задания валидного
`BOT_TOKEN` (перезапуск не нужен: `docker compose ... restart bot`).

**Бэкап базы** (рекомендуется ежедневно, например cron/systemd-timer):

```powershell
docker compose -f docker-compose.onprem.yml exec db pg_dump -U postgres onboarding > backup_$(Get-Date -Format yyyyMMdd).sql
```

**Обновление версии** (код + схема БД):

```powershell
git pull
docker compose -f docker-compose.onprem.yml up -d --build
# миграции до head применятся сами при старте; откат: alembic downgrade -1
```

**Чек-лист настройки:** задан `ADMIN_API_TOKEN`; загружен регламент в базу знаний
(`POST /api/admin/knowledge`); создан код приглашения (`POST /api/admin/codes`);
настроен бэкап; проведён пробный прогон сотрудником и просмотр
`/api/admin/results` + `/api/admin/gaps`.

**Что осознанно не входит (roadmap):** мониторинг/алёртинг (Sentry, Grafana),
нагрузочные тесты, Kubernetes-манифесты, SSO.

## Миграции БД (Alembic)
Схема Postgres управляется версионированными миграциями (`alembic/versions/`).
Миграции до `head` применяются автоматически при старте бота и веб-канала
(`init_db` в `database/db.py`). Базы, созданные раньше через `create_all`
(до появления Alembic), распознаются автоматически и фиксируются на `head`
без изменений (`alembic stamp head`), данные не трогаются.

Новая миграция после изменения моделей:

```powershell
$env:DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/onboarding"
.venv\Scripts\alembic revision --autogenerate -m "описание изменения"
# проверить файл в alembic/versions/, затем применить:
.venv\Scripts\alembic upgrade head
```

SQLite (локальные тесты) миграциями не управляется — схема собирается
`create_all` без pgvector-таблиц.

## Тесты

```powershell
pip install -r requirements-dev.txt
pytest
```

- `tests/unit/` — анонимайзер, RRF-склейка, `apply_ai_turn`, парсер документов,
  судья на фейковом gateway. Без БД и сети.
- `tests/integration/` — репозитории на SQLite; сквозной веб-флоу
  («код → согласие → имя → диалог → результат») через ASGI с фейковой моделью;
  RAG с реальным pgvector — через временный docker-контейнер (если docker
  недоступен, тест пропускается).

## Пример сценария

```text
HR загружает инструкцию сборщика заказов в базу знаний (admin API)
Пользователь: /start (или код приглашения на сайте)
Бот: [уведомление об обработке ПДн, кнопка «Согласен»]
Пользователь: (согласие)
Бот: Напишите имя сотрудника, которого нужно обучить.
Пользователь: Иван Петров
Бот: Сегодня разберем регламент сборки онлайн-заказов...
Пользователь: А сколько заморозка может быть вне холодильника?
Бот: [RAG: чанк «Температурный режим»] Не более 20 минут, в термопакете.
Пользователь: Готов к тесту
Бот: Вопрос 1. Куда упаковывать бытовую химию?
...
Бот: Тест завершен. Результат сохранен в Postgres.
```
