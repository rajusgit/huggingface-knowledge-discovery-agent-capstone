-- hf_discovery: schema for the HF Knowledge & Discovery Agent. Idempotent.
-- No pre-existing schema or table is touched.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS hf_discovery;

-- License/permission registry: authority for what may be indexed.
CREATE TABLE IF NOT EXISTS hf_discovery.sources (
    source_id      text PRIMARY KEY,
    canonical_url  text NOT NULL,
    owner          text,
    license        text,
    access_class   text NOT NULL DEFAULT 'public',
    permitted_use  text NOT NULL CHECK (permitted_use IN
                   ('pending_review','index_text','reference_only','rejected')),
    store_text     boolean NOT NULL DEFAULT false,
    version        text,
    attribution    text,
    category       text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);

-- One row per ingest; at most one 'active' run is what retrieval reads by default.
CREATE TABLE IF NOT EXISTS hf_discovery.ingest_runs (
    run_id                    bigserial PRIMARY KEY,
    corpus_revision           text NOT NULL,
    chunk_strategy            text NOT NULL,
    embedding_model           text NOT NULL,
    embedding_model_revision  text,
    status                    text NOT NULL CHECK (status IN
                              ('running','validated','active','superseded','failed')),
    stats                     jsonb NOT NULL DEFAULT '{}',
    started_at                timestamptz NOT NULL DEFAULT now(),
    finished_at               timestamptz,
    UNIQUE (corpus_revision, chunk_strategy)
);
CREATE UNIQUE INDEX IF NOT EXISTS ingest_runs_one_active
    ON hf_discovery.ingest_runs ((true)) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS hf_discovery.doc_chunks (
    chunk_id        text PRIMARY KEY,
    run_id          bigint NOT NULL REFERENCES hf_discovery.ingest_runs,
    source_id       text   NOT NULL REFERENCES hf_discovery.sources,
    source_url      text   NOT NULL,
    heading_path    text[] NOT NULL DEFAULT '{}',
    revision        text   NOT NULL,
    license         text   NOT NULL,
    category        text,
    access          text   NOT NULL DEFAULT 'public',
    chunk_strategy  text   NOT NULL,
    chunk_index     int    NOT NULL,
    start_offset    int,
    end_offset      int,
    content         text,
    content_hash    text   NOT NULL,
    embedding       vector(384) NOT NULL,
    tsv             tsvector GENERATED ALWAYS AS
                    (to_tsvector('english', coalesce(content, ''))) STORED,
    indexed_at      timestamptz NOT NULL DEFAULT now(),
    deleted_at      timestamptz
);
-- VECTOR_INDEX
CREATE INDEX IF NOT EXISTS doc_chunks_tsv_gin ON hf_discovery.doc_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS doc_chunks_run     ON hf_discovery.doc_chunks (run_id);
CREATE INDEX IF NOT EXISTS doc_chunks_source  ON hf_discovery.doc_chunks (source_id);

CREATE TABLE IF NOT EXISTS hf_discovery.sessions (
    thread_id       uuid PRIMARY KEY,
    memory_enabled  boolean NOT NULL DEFAULT true,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS hf_discovery.messages (
    message_id        bigserial PRIMARY KEY,
    thread_id         uuid NOT NULL REFERENCES hf_discovery.sessions ON DELETE CASCADE,
    role              text NOT NULL CHECK (role IN ('user','assistant','tool')),
    content           text NOT NULL,
    pipeline_variant  text,
    citations         jsonb NOT NULL DEFAULT '[]',
    guard_events      jsonb NOT NULL DEFAULT '[]',
    llm_model         text,
    token_usage       jsonb,
    cost_usd          numeric(12,6),
    latency_ms        int,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_thread ON hf_discovery.messages (thread_id, created_at);

CREATE TABLE IF NOT EXISTS hf_discovery.benchmark_runs (
    job_id            uuid PRIMARY KEY,
    model_id          text NOT NULL,
    model_revision    text,
    dataset_id        text NOT NULL,
    dataset_revision  text,
    status            text NOT NULL CHECK (status IN ('queued','running','succeeded','failed','timeout')),
    seed              int,
    hardware          jsonb,
    metrics           jsonb,
    error             text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz
);
