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
    "Country",
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

DEFAULT_ENTITY_TYPE = "Unknown"
DEFAULT_RELATION_TYPE = "mentions"

ENTITY_TYPE_DESCRIPTIONS = {
    "Material": "Материалы, сырье, продукты, вещества, сплавы, концентраты.",
    "Process": "Технологические процессы и операции.",
    "Equipment": "Оборудование, установки, аппараты.",
    "Property": "Свойства материалов, параметров или процессов.",
    "Experiment": "Эксперименты, испытания, лабораторные исследования.",
    "Publication": "Статьи, отчеты, патенты, нормативные документы.",
    "Expert": "Эксперты, авторы, организации или специалисты.",
    "Facility": "Площадки, лаборатории, предприятия, месторождения.",
    "Condition": "Условия проведения процесса: температура, давление, среда, время.",
    "Result": "Результаты, выводы, эффекты, показатели.",
    "Country": "Страны и явно указанные национальные юрисдикции.",
    "Unknown": "Неопределенный тип, если классификация неясна.",
}

RELATION_TYPE_DESCRIPTIONS = {
    "uses_material": "Процесс, оборудование или эксперимент использует материал.",
    "operates_at_condition": "Процесс или оборудование работает при заданном условии.",
    "produces_output": "Процесс или эксперимент производит результат или продукт.",
    "described_in": "Сущность описана в публикации, отчете или документе.",
    "validated_by": "Факт, процесс или результат подтвержден экспериментом/источником.",
    "contradicts": "Факт или результат противоречит другому факту/источнику.",
    "applies_to": "Свойство, условие или вывод применимы к сущности.",
    "has_effect": "Процесс, условие или материал оказывает эффект на результат.",
    "mentions": "Общая слабая связь упоминания, если точный тип связи неясен.",
}


def get_entity_types() -> tuple[str, ...]:
    return ENTITY_TYPES


def get_relation_types() -> tuple[str, ...]:
    return RELATION_TYPES


def is_valid_entity_type(value: str) -> bool:
    return value in ENTITY_TYPES


def is_valid_relation_type(value: str) -> bool:
    return value in RELATION_TYPES


def normalize_entity_type(value: str | None) -> str:
    if value is None:
        return DEFAULT_ENTITY_TYPE
    cleaned = str(value).strip()
    return cleaned if is_valid_entity_type(cleaned) else DEFAULT_ENTITY_TYPE


def normalize_relation_type(value: str | None) -> str:
    if value is None:
        return DEFAULT_RELATION_TYPE
    cleaned = str(value).strip()
    return cleaned if is_valid_relation_type(cleaned) else DEFAULT_RELATION_TYPE


def format_entity_types_for_prompt() -> str:
    return ", ".join(ENTITY_TYPES)


def format_relation_types_for_prompt() -> str:
    return ", ".join(RELATION_TYPES)


def get_entity_type_table() -> list[dict[str, str]]:
    return [
        {
            "type": entity_type,
            "description": ENTITY_TYPE_DESCRIPTIONS.get(entity_type, ""),
        }
        for entity_type in ENTITY_TYPES
    ]


def get_relation_type_table() -> list[dict[str, str]]:
    return [
        {
            "relation_type": relation_type,
            "description": RELATION_TYPE_DESCRIPTIONS.get(relation_type, ""),
        }
        for relation_type in RELATION_TYPES
    ]
