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
OLLAMA_GENERATION_MODEL=qwen2.5:1.5b
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
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

Пример `.env` для локального режима:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_GENERATION_MODEL=qwen2.5:1.5b
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
