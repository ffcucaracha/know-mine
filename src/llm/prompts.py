from __future__ import annotations


ENTITY_TYPES = (
    "Material",
    "Process",
    "Equipment",
    "Property",
    "Experiment",
    "Publication",
    "Expert",
    "Facility",
    "Condition",
    "Result",
    "Unknown",
)

RELATION_TYPES = (
    "uses_material",
    "operates_at_condition",
    "produces_output",
    "described_in",
    "validated_by",
    "contradicts",
    "applies_to",
    "has_effect",
    "mentions",
)

EXTRACTION_SYSTEM_PROMPT = """
Ты извлекаешь структурированные знания из R&D и научно-технических текстов
горно-металлургического домена.

Верни только валидный JSON. Не добавляй markdown, комментарии или пояснения.
Не выдумывай сущности, факты, числа, годы, географию и связи.
Если в тексте нет данных для извлечения, верни пустые массивы.

Учитывай домен:
- материалы, руды, металлы, растворы, реагенты;
- процессы, технологии, эксперименты и режимы;
- оборудование, установки, фабрики и площадки;
- свойства, показатели качества, извлечение, концентрации, температуры;
- публикации, экспертов, организации и географию.

Допустимые entity.type:
Material, Process, Equipment, Property, Experiment, Publication, Expert, Facility, Condition, Result, Unknown.

Допустимые relations.relation:
uses_material, operates_at_condition, produces_output, described_in, validated_by,
contradicts, applies_to, has_effect, mentions.

Требования к значениям:
- confidence всегда число от 0 до 1.
- numeric_value всегда число или null.
- year всегда число или null.
- numeric_unit содержит единицу измерения, если она указана: мг/л, °C, т/сут, %, м/с, МПа и т.п.
- geography заполняй только если география явно указана: Россия, зарубежная практика, страна, регион.
- evidence должен быть коротким фрагментом из исходного текста, подтверждающим связь.
""".strip()

EXTRACTION_USER_PROMPT_TEMPLATE = """
Файл-источник: {filename}

Извлеки entities, facts и relations из текста ниже.

Верни JSON строго такой структуры:
{{
  "entities": [
    {{
      "label": "...",
      "type": "Material|Process|Equipment|Property|Experiment|Publication|Expert|Facility|Condition|Result|Unknown"
    }}
  ],
  "facts": [
    {{
      "statement": "...",
      "material": "...",
      "process": "...",
      "equipment": "...",
      "property": "...",
      "condition_text": "...",
      "numeric_value": null,
      "numeric_unit": "...",
      "geography": "...",
      "year": null,
      "confidence": 0.0
    }}
  ],
  "relations": [
    {{
      "source": "...",
      "relation": "uses_material|operates_at_condition|produces_output|described_in|validated_by|contradicts|applies_to|has_effect|mentions",
      "target": "...",
      "evidence": "..."
    }}
  ]
}}

Если значения нет, используй null для numeric_value/year и пустую строку или null для текстовых полей.
Если данных нет, верни:
{{"entities": [], "facts": [], "relations": []}}

Текст:
{chunk_text}
""".strip()

FACT_EXTRACTION_PROMPT = EXTRACTION_SYSTEM_PROMPT

ANSWER_SYSTEM_PROMPT = """
Ты R&D Knowledge Graph Assistant. Отвечай только по предоставленному контексту:
фрагментам документов и фактам из графа знаний.

Правила:
- Не выдумывай факты, числа, источники и страницы.
- Если данных недостаточно, явно напиши: «В предоставленных источниках не найдено достаточно данных».
- Если источники противоречат друг другу, укажи противоречие.
- В источниках указывай filename и страницы, если они есть.
- Ответ должен быть на русском языке.
""".strip()

ANSWER_USER_PROMPT_TEMPLATE = """
Вопрос пользователя:
{question}

Эвристика запроса:
{query_hints}

Контекст из документов:
{document_context}

Факты из графа знаний:
{facts_context}

Сформируй ответ строго в формате:
1. Краткий вывод
2. Найденные методы / эксперименты / технологии
3. Условия и числовые параметры
4. Эффект / результат
5. Источники
6. Уверенность
7. Пробелы и противоречия
""".strip()

ANSWER_PROMPT = ANSWER_SYSTEM_PROMPT


def build_extraction_prompt(chunk_text: str, filename: str) -> tuple[str, str]:
    user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
        filename=filename,
        chunk_text=chunk_text,
    )
    return EXTRACTION_SYSTEM_PROMPT, user_prompt


def build_answer_prompt(question: str, context: str) -> tuple[str, str]:
    user_prompt = """
Вопрос пользователя:
{question}

Контекст:
{context}

Сформируй ответ строго в формате:
1. Краткий вывод
2. Найденные методы / эксперименты / технологии
3. Условия и числовые параметры
4. Эффект / результат
5. Источники
6. Уверенность
7. Пробелы и противоречия
""".strip().format(question=question, context=context)
    return ANSWER_SYSTEM_PROMPT, user_prompt
