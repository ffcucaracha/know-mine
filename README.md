# Know Mine — R&D Knowledge Graph Assistant

Know Mine превращает архив научно-технических документов в карту знаний и помогает отвечать на естественные вопросы с указанием источников.

MVP реализует GraphRAG-подход: релевантные фрагменты ищутся через vector search в ChromaDB, а извлеченные сущности, факты и связи сохраняются в SQLite как упрощенный граф знаний.

## Стек

- Streamlit
- Yandex AI Studio
- Ollama для локального LLM-режима
- ChromaDB
- SQLite
- PyMuPDF
- python-docx

## Запуск

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Заполните в `.env`:

```env
YANDEX_API_KEY=...
YANDEX_FOLDER_ID=...
```

Затем запустите приложение:

```bash
streamlit run app.py
```

## UI structure

Know Mine разделяет пользовательские и административные сценарии.

Верхние вкладки:

- `Вопрос` — задать вопрос, посмотреть ответ, источники и добавить источник в избранное.
- `Избранное` — сохраненные источники.
- `Карта знаний` — поиск сущностей и граф связей.
- `Администрирование` — инструменты обслуживания базы знаний.

Внутри `Администрирование`:

- `Загрузка` — загрузка архивов, локальные источники, извлечение текста, чанкинг и индекс.
- `Факты` — запуск extraction facts/relations и просмотр фактов.
- `LLM Usage` — статистика LLM-запросов, токенов, стоимости и ошибок.
- `Состояние` — storage status, routing моделей, healthcheck и системные счетчики.

Sidebar в обычном режиме показывает только выбранные модели, storage summary и кнопку `Проверить модели`. Технические поля вроде DB path, raw model URI, demo limits и reset-кнопки скрыты.

Чтобы включить debug-детали администратора:

```env
SHOW_ADMIN_DEBUG=true
```

## Формат `.env`

```env
LLM_PROVIDER=yandex

YANDEX_API_KEY=
YANDEX_FOLDER_ID=

YANDEX_GENERATION_MODEL=yandexgpt-lite
YANDEX_GENERATION_MODEL_URI=
YANDEX_EMBEDDING_MODEL=text-search-doc
YANDEX_EMBEDDING_MODEL_URI=

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_GENERATION_MODEL=mistral
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_EMBEDDING_MAX_CHARS=1000

LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000
LLM_TIMEOUT_SECONDS=60
LLM_RETRY_COUNT=3

LLM_USAGE_TRACKING_ENABLED=true
LLM_USAGE_STORE_HASHES=true
LLM_APPROX_CHARS_PER_TOKEN=4
LLM_COST_INPUT_PER_1K=0
LLM_COST_OUTPUT_PER_1K=0
LLM_COST_EMBEDDING_PER_1K=0
LLM_COST_CURRENCY=RUB

MOCK_EMBEDDING_DIM=384

CHUNK_SIZE=1000
CHUNK_OVERLAP=150

EMBEDDING_MAX_CHARS=2000
EMBEDDING_BATCH_SIZE=8

DEMO_MAX_DOCUMENTS=5
DEMO_MAX_CHUNKS=30

SOURCE_PATH=data/raw/sources
MAX_UPLOAD_SIZE_MB=200
SUPPORTED_EXTENSIONS=.pdf,.docx
SHOW_ADMIN_DEBUG=false
```

Для локальной отладки без внешнего API можно использовать:

```env
LLM_PROVIDER=mock
```

`mock` нужен для проверки UI, загрузки, чанкинга, retrieval и графа без Yandex credentials. Для реального извлечения фактов, embeddings и ответов используйте `LLM_PROVIDER=yandex`.

## Ollama Local Provider

Know Mine поддерживает локальный Ollama-провайдер без изменения архитектуры пайплайна: приложение по-прежнему работает через общий `LLMClient`, а провайдер выбирается переменной `LLM_PROVIDER`.

Установите и запустите Ollama:

```bash
ollama serve
```

В отдельном терминале загрузите модели:

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

Пример `.env` для локального режима:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_GENERATION_MODEL=mistral
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_EMBEDDING_MAX_CHARS=1000
```

Проверьте доступность:

```bash
curl http://localhost:11434/api/tags
```

Затем запустите приложение обычной командой:

```bash
streamlit run app.py
```

Для `ollama` ключи Yandex не требуются. В sidebar будет показан base URL и выбранные generation/embedding модели.

## Ollama Embedding Context Errors

Если при построении векторного индекса возникает ошибка:

```text
the input length exceeds the context length
```

уменьшите размер чанков и лимит текста для embeddings:

```env
CHUNK_SIZE=800
CHUNK_OVERLAP=100
OLLAMA_EMBEDDING_MAX_CHARS=800
DEMO_MAX_CHUNKS=20
```

После изменения настроек удалите старый индекс и пересоздайте чанки. В UI используйте кнопку `Сбросить чанки и индекс`, затем снова выполните:

- `Создать чанки`
- `Построить векторный индекс`

Кнопка сброса сохраняет documents, но очищает chunks, facts, nodes, edges и локальный `chroma_db`, чтобы старые факты не ссылались на удаленные чанки.

## Что реализовано

- Загрузка `.zip` архива с `.pdf` и `.docx`.
- Режим локального пути для больших папок и `.zip` архивов без browser upload.
- Распаковка архива и рекурсивный поиск поддерживаемых документов.
- Извлечение текста из PDF через PyMuPDF и DOCX через python-docx.
- Чанкинг документов с сохранением страниц для PDF.
- SQLite-хранилище для документов, чанков, фактов, nodes и edges.
- Локальный ChromaDB vector index.
- Provider-agnostic LLM-слой через общий `LLMClient`.
- Извлечение сущностей, фактов и связей из чанков.
- Нормализация терминов и словарь синонимов для горно-металлургического домена.
- Ответы на вопросы с источниками, фактами и найденными фрагментами.
- Визуализация мини-графа знаний через pyvis.
- Экспорт ответа в Markdown.
- Локальная статистика использования LLM в SQLite.
- Demo mode с лимитами `DEMO_MAX_DOCUMENTS` и `DEMO_MAX_CHUNKS`.

## Large source archives

Для архивов и папок источников размером 1+ ГБ не рекомендуется использовать browser upload в Streamlit: загрузка через браузер медленнее, сильнее расходует память и чаще упирается в таймауты.

Лучше положить архив или распакованную папку в локальный каталог проекта, например `data/raw/sources`:

```bash
mkdir -p data/raw/sources
unzip archive.zip -d data/raw/sources
```

Пример `.env` для демо-обработки большого корпуса:

```env
SOURCE_PATH=data/raw/sources
DEMO_MAX_DOCUMENTS=30
DEMO_MAX_CHUNKS=100
SUPPORTED_EXTENSIONS=.pdf,.docx
MAX_UPLOAD_SIZE_MB=200
```

Затем запустите приложение, откройте вкладку `Загрузка`, выберите режим `Большой архив / папка по локальному пути`, проверьте путь и нажмите `Сканировать путь`. Приложение быстро соберет только metadata файлов, покажет первые 100 строк и при обработке возьмет не больше `DEMO_MAX_DOCUMENTS` документов.

Для полного корпуса увеличивайте `DEMO_MAX_DOCUMENTS` и `DEMO_MAX_CHUNKS` осторожно: извлечение текста, чанкинг, embeddings и LLM-извлечение фактов масштабируются по числу документов и чанков.

## Minimal deduplication

Know Mine выполняет минимальную защиту от повторной обработки:

- documents пропускаются по `file_hash`, который считается потоково из файла;
- chunks пропускаются по `chunk_hash`, который считается из текста чанка;
- vector index пропускает `chunk_id`, которые уже есть в ChromaDB collection;
- LLM extraction пропускает `chunk_id`, по которым уже есть facts в SQLite.

Если LLM вернула 0 facts для чанка, такой chunk в минимальном варианте может быть обработан повторно, потому что признаком завершенной extraction сейчас являются сохраненные facts.

Чтобы пересобрать всё с нуля, остановите приложение и удалите SQLite и ChromaDB:

```bash
rm -f knowmine.sqlite
rm -rf chroma_db
```

## Entity normalization

Сущности в графе дедуплицируются по `canonical_name`: имя очищается от лишних пробелов и кавычек, приводится к lowercase, а `ё` заменяется на `е`. Поэтому `Медь`, ` медь ` и `МЕДЬ` сохраняются как одна node с `canonical_name=медь`; латиница также приводится к lowercase, например `Copper` -> `copper`.

Если уже существующая node имеет `type=Unknown`, а при следующем extraction приходит та же сущность с конкретным типом, например `Material`, Know Mine обновляет type у существующей node вместо создания дубля.

## Taxonomy

Допустимые типы сущностей и связей лежат в `src/graph/taxonomy.py`. Этот справочник используется одновременно в extraction prompt, JSON validation, SQLite saving и UI.

LLM не может записать произвольный тип: неизвестный entity type превращается в `Unknown`, неизвестный relation type превращается в `mentions`. Чтобы добавить новый тип сущности или связи, обновите `ENTITY_TYPES` или `RELATION_TYPES` и соответствующий словарь описаний в `taxonomy.py`.

## Favorites

В ответах Know Mine можно добавлять найденные источники и чанки в избранное. Кнопка `☆ Добавить в избранное` доступна рядом с найденными фрагментами во вкладке `Вопрос`, а сохраненные элементы отображаются во вкладке `Избранное`.

Избранное хранится локально в SQLite в таблице `favorites`. Это fallback вместо browser `localStorage`: он не требует авторизации, сохраняется между перезапусками приложения и не добавляет frontend-зависимостей. В избранное сохраняется только metadata источника и preview текста до 1000 символов, без API keys и полных документов.

Дубликаты не добавляются: стабильный `favorite_id` строится по `chunk_id`, либо по `document_id` и диапазону страниц, либо по hash от filename/snippet. Во вкладке `Избранное` можно искать, удалять отдельные источники и очищать весь список.

## Report history

Каждый успешно сформированный Markdown-отчёт автоматически сохраняется в локальную SQLite базу в таблицу `report_history`. Пользовательская вкладка `История отчётов` находится рядом с `Избранное` и позволяет найти прежний отчёт по вопросу, открыть preview, скачать Markdown повторно, удалить один отчёт или очистить всю историю.

По умолчанию хранится до `MAX_REPORT_HISTORY=200` последних отчётов. Лимит можно изменить в `.env`:

```env
MAX_REPORT_HISTORY=200
```

История сохраняет финальный ответ, Markdown-отчёт, counts источников/фактов и дату актуализации источников. API keys, raw prompts, tracebacks и provider credentials в историю не записываются.

## LLM Usage Statistics

Статистика использования LLM хранится локально в SQLite в таблице `llm_usage_events`.

Учитываются операции:

- `generation`
- `extraction`
- `answer`
- `embedding`
- `healthcheck`

В UI доступна вкладка `LLM Usage`, где показаны:

- количество запросов;
- успешные и ошибочные запросы;
- примерные токены;
- примерная стоимость;
- группировка по операциям;
- группировка по провайдерам;
- последние события.

Стоимость является приблизительной, если провайдер не возвращает фактический token usage. Коэффициенты задаются через `.env`:

```env
LLM_COST_INPUT_PER_1K=0
LLM_COST_OUTPUT_PER_1K=0
LLM_COST_EMBEDDING_PER_1K=0
LLM_COST_CURRENCY=RUB
```

По умолчанию cost = `0`, чтобы не имитировать официальные тарифы. Для реального бюджетного контроля команда должна вручную вписать актуальные коэффициенты.

Включение/выключение статистики:

```env
LLM_USAGE_TRACKING_ENABLED=true
```

Хранение hashes prompt/response:

```env
LLM_USAGE_STORE_HASHES=true
```

Полные prompt, response, документы и API key не сохраняются. Сохраняются только длины, approximate tokens, hashes, provider/model, operation, latency и error metadata.

## Ограничения MVP

- Нет OCR для сканов и изображений внутри PDF.
- SQLite используется как упрощенное graph storage, не как промышленная graph database.
- Качество фактов зависит от качества исходного текста и LLM-ответа.
- Нет дедупликации фактов на уровне экспертной валидации.
- Нет пользовательских ролей и разграничения доступа.
- Нет инкрементального обновления индекса по измененным документам.
- Нет интеграции с корпоративными DMS/ECM/SharePoint/S3-хранилищами.

## Production Roadmap

- Neo4j или другой промышленный graph backend.
- OCR для сканов, изображений и табличных PDF.
- RBAC и разграничение доступа к источникам.
- Аудит действий пользователей и LLM-вызовов.
- Экспертная корректировка графа знаний.
- Инкрементальное обновление документов, индекса и графа.
- Интеграция с внутренними хранилищами документов.

## Security

- Секреты передаются только через `.env`.
- `.env` добавлен в `.gitignore`.
- API key не хранится в коде и не выводится в Streamlit UI.
- Для демо без ключей используйте `LLM_PROVIDER=mock`.
