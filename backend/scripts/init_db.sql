-- ============================================================
-- fire-safety-rag 数据库初始化 SQL
-- 目标库：Neon PostgreSQL（免费套餐自带 pgvector）
-- 用法：
--   1) Neon 控制台 -> SQL Editor，粘贴执行；或
--   2) psql "<Neon 的 -pooler 连接串>" -f scripts/init_db.sql
-- 说明：业务表由 Alembic / SQLAlchemy 建；本脚本只做"扩展 + 兜底手建表 + 校验"
-- ============================================================

-- ---------- 1. 扩展 ----------
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector：向量列与相似度算子(<=>)
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- 可选：中文/英文模糊匹配兜底
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- 可选：服务端生成 uuid

-- 校验扩展是否可用
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_trgm');

-- ---------- 2. 业务表（与 app/db/models.py 对齐；Alembic 未跑时可直接执行）----------
CREATE TABLE IF NOT EXISTS documents (
    id              VARCHAR(36)  PRIMARY KEY,
    title           VARCHAR(512) NOT NULL,
    source_type     VARCHAR(32)  DEFAULT 'pdf',
    source_path     TEXT         DEFAULT '',
    file_hash       VARCHAR(64)  DEFAULT '',
    doc_category    VARCHAR(64)  DEFAULT 'general',
    status          VARCHAR(32)  DEFAULT 'pending',
    is_scanned      BOOLEAN      DEFAULT FALSE,
    page_count      INTEGER      DEFAULT 0,
    chunk_count     INTEGER      DEFAULT 0,
    metadata        JSONB        DEFAULT '{}'::jsonb,
    error_message   TEXT         DEFAULT '',
    created_at      TIMESTAMPTZ  DEFAULT now(),
    updated_at      TIMESTAMPTZ  DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_documents_hash_status ON documents (file_hash, status);
CREATE INDEX IF NOT EXISTS ix_documents_status      ON documents (status);

CREATE TABLE IF NOT EXISTS chunks (
    id                      VARCHAR(36) PRIMARY KEY,
    document_id             VARCHAR(36) REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index             INTEGER     DEFAULT 0,
    content                 TEXT        NOT NULL,
    heading_path            TEXT        DEFAULT '',
    token_count             INTEGER     DEFAULT 0,
    page_no                 INTEGER     DEFAULT 0,
    hypothetical_questions  JSONB       DEFAULT '[]'::jsonb,
    metadata                JSONB       DEFAULT '{}'::jsonb,
    embedding               vector(1024),        -- 维度必须与 EMBEDDING_DIM 一致（bge-m3=1024）
    created_at              TIMESTAMPTZ DEFAULT now(),
    updated_at              TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_chunks_document_id ON chunks (document_id);

-- 自建检索时用的 HNSW 索引（第一批先不建，数据量 > 1 万条再建，构建较慢）
-- CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw
--     ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id          VARCHAR(36) PRIMARY KEY,
    user_id     VARCHAR(64) DEFAULT 'anonymous',
    title       VARCHAR(255) DEFAULT '新会话',
    summary     TEXT        DEFAULT '',
    turn_count  INTEGER     DEFAULT 0,
    metadata    JSONB       DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_id ON chat_sessions (user_id);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL   PRIMARY KEY,
    session_id  VARCHAR(36) REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        VARCHAR(16) NOT NULL,
    content     TEXT        NOT NULL,
    sources     JSONB       DEFAULT '[]'::jsonb,
    latency_ms  INTEGER     DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_chat_messages_session_created ON chat_messages (session_id, created_at);

CREATE TABLE IF NOT EXISTS eval_records (
    id              VARCHAR(36) PRIMARY KEY,
    run_name        VARCHAR(128) DEFAULT 'default',
    eval_type       VARCHAR(32) NOT NULL,
    metrics         JSONB       DEFAULT '{}'::jsonb,
    dataset_size    INTEGER     DEFAULT 0,
    config_snapshot JSONB       DEFAULT '{}'::jsonb,
    duration_sec    DOUBLE PRECISION DEFAULT 0,
    detail          JSONB       DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_eval_records_run_name ON eval_records (run_name);

-- ---------- 3. 说明 ----------
-- 3.1 LlamaIndex PGVectorStore 会自建一张表（默认 data_<table_name>，如 data_fire_chunks），
--     结构为 (id, node_id, text, metadata_, embedding vector(dim))，**不需要手工创建**，
--     首次 ingest 时由 PGVectorStore 自动建表并建 HNSW 索引。
-- 3.2 连接串请用 Neon 的 pooler 串（主机名含 -pooler），并带 sslmode=require。
-- 3.3 清库重来（谨慎）：
--     TRUNCATE chat_messages, chat_sessions, chunks, documents RESTART IDENTITY CASCADE;
--     DROP TABLE IF EXISTS data_fire_chunks;
