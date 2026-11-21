from ferry.metadata.base import (
    AggregatorState,
    CheckpointRecord,
    CommitRequest,
    CommitResult,
    DeltaRecord,
    MetadataStore,
    NodeRecord,
)
from ferry.metadata.memory import InMemoryMetadataStore


def open_metadata_store(dsn: str) -> MetadataStore:
    """``memory://`` for the in-memory store, otherwise a Postgres DSN."""
    if dsn in ("memory", "memory://"):
        return InMemoryMetadataStore()
    from ferry.metadata.postgres import PostgresMetadataStore

    return PostgresMetadataStore(dsn)


__all__ = [
    "AggregatorState",
    "CheckpointRecord",
    "CommitRequest",
    "CommitResult",
    "DeltaRecord",
    "InMemoryMetadataStore",
    "MetadataStore",
    "NodeRecord",
    "open_metadata_store",
]
