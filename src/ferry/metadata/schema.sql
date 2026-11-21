-- Ferry v1 coordinator schema (DESIGN §5). Idempotent.

CREATE TABLE IF NOT EXISTS nodes (
    node_id          TEXT PRIMARY KEY,
    node_type        TEXT NOT NULL CHECK (node_type IN ('worker', 'aggregator')),
    state            TEXT NOT NULL,               -- JOINING | IDLE | TRAINING | UPLOADING | AGGREGATING | SUSPECT | DEAD
    advertise_addr   TEXT,                        -- aggregator only
    reported_config  JSONB,                       -- worker DescribeResponse
    last_heartbeat   JSONB,                       -- latest Heartbeat payload
    lease_expires_at TIMESTAMPTZ,
    node_state_id    TEXT,                        -- last persisted resumable state (blob ID)
    registered_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Single row: the active aggregator and the latest committed global checkpoint.
CREATE TABLE IF NOT EXISTS aggregator_state (
    id                   BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    active_aggregator_id TEXT REFERENCES nodes(node_id),
    epoch                BIGINT NOT NULL DEFAULT 0,
    latest_checkpoint_id TEXT,
    global_version       BIGINT NOT NULL DEFAULT 0,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO aggregator_state (id) VALUES (TRUE) ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id   TEXT PRIMARY KEY,
    global_version  BIGINT NOT NULL UNIQUE,
    parent_version  BIGINT,
    uri             TEXT NOT NULL,
    aggregator_id   TEXT NOT NULL,
    epoch           BIGINT NOT NULL,
    delta_ids       TEXT[] NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Audit log of deltas the aggregator accepted or rejected.
CREATE TABLE IF NOT EXISTS deltas (
    delta_id            TEXT PRIMARY KEY,
    worker_id           TEXT NOT NULL,
    base_global_version BIGINT NOT NULL,
    inner_steps         INT NOT NULL,
    num_samples         BIGINT NOT NULL,
    status              TEXT NOT NULL,              -- accepted | rejected | applied
    reason              TEXT,
    received_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
