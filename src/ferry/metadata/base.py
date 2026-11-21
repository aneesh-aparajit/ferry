"""``MetadataStore``: the coordinator's source of truth (DESIGN §5, §6).

Postgres in production; an in-memory implementation for tests and single-process runs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

WORKER = "worker"
AGGREGATOR = "aggregator"

# Node states as stored (DESIGN §5 `nodes.state`).
JOINING = "JOINING"
IDLE = "IDLE"
TRAINING = "TRAINING"
UPLOADING = "UPLOADING"
AGGREGATING = "AGGREGATING"
SUSPECT = "SUSPECT"
DEAD = "DEAD"

CHECKPOINT_CHANNEL = "ferry_checkpoint"


@dataclass
class NodeRecord:
    node_id: str
    node_type: str
    state: str
    advertise_addr: str | None = None
    reported_config: dict | None = None
    last_heartbeat: dict | None = None
    lease_expires_at: datetime | None = None
    node_state_id: str | None = None
    registered_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class AggregatorState:
    active_aggregator_id: str | None
    epoch: int
    latest_checkpoint_id: str | None
    global_version: int
    active_aggregator_addr: str | None = None
    latest_checkpoint_uri: str | None = None


@dataclass
class CheckpointRecord:
    checkpoint_id: str
    global_version: int
    parent_version: int | None
    uri: str
    aggregator_id: str
    epoch: int
    delta_ids: list[str] = field(default_factory=list)
    created_at: datetime | None = None


@dataclass
class DeltaRecord:
    delta_id: str
    worker_id: str
    base_global_version: int
    inner_steps: int
    num_samples: int
    status: str  # accepted | rejected | applied
    reason: str | None = None


@dataclass
class CommitRequest:
    aggregator_id: str
    epoch: int
    checkpoint_id: str
    uri: str
    parent_version: int
    delta_ids: list[str] = field(default_factory=list)
    delta_records: list[DeltaRecord] = field(default_factory=list)
    initial: bool = False


@dataclass
class CommitResult:
    accepted: bool
    global_version: int = 0
    reason: str = ""


def rejection_reason(state: AggregatorState, req: CommitRequest, enforce_epoch: bool) -> str:
    """Explain why a conditional commit UPDATE matched zero rows."""
    if state.active_aggregator_id != req.aggregator_id:
        return "not_active_aggregator"
    if enforce_epoch and state.epoch != req.epoch:
        return "stale_epoch"
    if req.initial:
        return "already_initialized"
    if state.latest_checkpoint_id is None:
        return "not_initialized"
    return f"version_conflict: current={state.global_version} parent={req.parent_version}"


class MetadataStore(Protocol):
    async def init(self) -> None: ...

    async def close(self) -> None: ...

    # nodes
    async def upsert_node(self, node_id: str, node_type: str, advertise_addr: str | None, state: str) -> NodeRecord:
        """Create the node, or resume it if ``node_id`` already exists (keeps ``node_state_id``)."""
        ...

    async def get_node(self, node_id: str) -> NodeRecord | None: ...

    async def list_nodes(self) -> list[NodeRecord]: ...

    async def set_reported_config(self, node_id: str, config: dict) -> None: ...

    async def record_heartbeat(self, node_id: str, state: str, heartbeat: dict, lease_expires_at: datetime) -> None: ...

    async def set_node_state(self, node_id: str, state: str) -> None: ...

    async def set_node_state_id(self, node_id: str, node_state_id: str) -> None: ...

    async def expire_leases(self, now: datetime, dead_after_s: float) -> list[tuple[str, str]]:
        """Mark expired leases ``SUSPECT`` and long-expired ones ``DEAD``. Returns ``[(node_id, new_state)]``."""
        ...

    # aggregator / checkpoints
    async def register_aggregator(self, node_id: str) -> int:
        """Make ``node_id`` the active aggregator and increment ``epoch``. Returns the new epoch."""
        ...

    async def get_aggregator_state(self) -> AggregatorState: ...

    async def commit_checkpoint(self, req: CommitRequest, enforce_epoch: bool = False) -> CommitResult: ...

    async def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None: ...

    async def list_checkpoints(self, limit: int = 100) -> list[CheckpointRecord]:
        """Newest first."""
        ...

    async def prune_checkpoints(self, keep: int) -> list[CheckpointRecord]:
        """Delete registry rows beyond the newest ``keep``; return them so blobs can be GC'd."""
        ...

    async def list_deltas(self) -> list[DeltaRecord]: ...

    def subscribe_checkpoints(self) -> AsyncIterator[str]:
        """Yield checkpoint IDs as they are committed (Postgres: ``LISTEN ferry_checkpoint``)."""
        ...
