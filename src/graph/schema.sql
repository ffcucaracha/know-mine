PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    title TEXT,
    file_hash TEXT,
    text_hash TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    chunk_hash TEXT,
    page_start INTEGER,
    page_end INTEGER,
    FOREIGN KEY(document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    type TEXT NOT NULL,
    normalized_label TEXT NOT NULL,
    canonical_name TEXT
);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    document_id TEXT,
    chunk_id TEXT,
    statement TEXT NOT NULL,
    material TEXT,
    process TEXT,
    equipment TEXT,
    property TEXT,
    condition_text TEXT,
    numeric_value REAL,
    numeric_unit TEXT,
    geography TEXT,
    year INTEGER,
    confidence REAL,
    FOREIGN KEY(document_id) REFERENCES documents(id),
    FOREIGN KEY(chunk_id) REFERENCES chunks(id)
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    fact_id TEXT,
    evidence TEXT,
    FOREIGN KEY(source_node_id) REFERENCES nodes(id),
    FOREIGN KEY(target_node_id) REFERENCES nodes(id),
    FOREIGN KEY(fact_id) REFERENCES facts(id)
);

CREATE TABLE IF NOT EXISTS llm_usage_events (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT,
    operation TEXT NOT NULL,
    request_chars INTEGER NOT NULL DEFAULT 0,
    response_chars INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    estimated_input_tokens INTEGER,
    estimated_output_tokens INTEGER,
    estimated_total_tokens INTEGER,
    cost_currency TEXT DEFAULT 'RUB',
    estimated_cost REAL DEFAULT 0,
    latency_ms INTEGER,
    success INTEGER NOT NULL DEFAULT 1,
    error_type TEXT,
    error_message TEXT,
    prompt_hash TEXT,
    response_hash TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS favorites (
    favorite_id TEXT PRIMARY KEY,
    chunk_id TEXT,
    document_id TEXT,
    filename TEXT NOT NULL,
    source_path TEXT,
    page_start INTEGER,
    page_end INTEGER,
    score REAL,
    snippet TEXT,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS report_history (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    answer_preview TEXT,
    markdown TEXT NOT NULL,
    sources_count INTEGER DEFAULT 0,
    facts_count INTEGER DEFAULT 0,
    actualization_date TEXT,
    filename TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_text_hash ON documents(text_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_hash ON chunks(chunk_hash);
CREATE INDEX IF NOT EXISTS idx_facts_document_id ON facts(document_id);
CREATE INDEX IF NOT EXISTS idx_facts_chunk_id ON facts(chunk_id);
CREATE INDEX IF NOT EXISTS idx_nodes_normalized_label ON nodes(normalized_label);
CREATE INDEX IF NOT EXISTS idx_nodes_canonical_name ON nodes(canonical_name);
CREATE INDEX IF NOT EXISTS idx_edges_source_node_id ON edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_target_node_id ON edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_llm_usage_created_at ON llm_usage_events(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_provider ON llm_usage_events(provider);
CREATE INDEX IF NOT EXISTS idx_llm_usage_operation ON llm_usage_events(operation);
CREATE INDEX IF NOT EXISTS idx_llm_usage_success ON llm_usage_events(success);
CREATE INDEX IF NOT EXISTS idx_favorites_added_at ON favorites(added_at);
CREATE INDEX IF NOT EXISTS idx_favorites_filename ON favorites(filename);
CREATE INDEX IF NOT EXISTS idx_report_history_created_at ON report_history(created_at);
CREATE INDEX IF NOT EXISTS idx_report_history_question ON report_history(question);
