"""In-memory ``MetadataStore`` with the same commit semantics as Postgres. For tests and single-process runs."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

from ferry.metadata.base import (
    DEAD,
    SUSPECT,
    AggregatorState,
    CheckpointRecord,
    CommitRequest,
    CommitResult,
    DeltaRecord,
    NodeRecord,
    rejection_reason,
)


class InMemoryMetadataStore:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeRecord] = {}
        self._agg = AggregatorState(None, 0, None, 0)
        self._checkpoints: dict[str, CheckpointRecord] = {}
        self._deltas: dict[str, DeltaRecord] = {}
        self._subscribers: list[asyncio.Queue[str]] = []
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert_node(self, node_id, node_type, advertise_addr, state) -> NodeRecord:
        now = datetime.now(UTC)
        async with self._lock:
            n = self._nodes.get(node_id)
            if n is None:
                n = NodeRecord(node_id, node_type, state, advertise_addr, registered_at=now, updated_at=now)
                self._nodes[node_id] = n
            else:
                if n.node_type != node_type:
                    raise ValueError(f"node {node_id} is a {n.node_type}, not a {node_type}")
                n.state = state
                n.advertise_addr = advertise_addr
                n.updated_at = now
            return copy.deepcopy(n)

    async def get_node(self, node_id) -> NodeRecord | None:
        n = self._nodes.get(node_id)
        return copy.deepcopy(n) if n else None

    async def list_nodes(self) -> list[NodeRecord]:
        return [copy.deepcopy(n) for n in self._nodes.values()]

    async def set_reported_config(self, node_id, config) -> None:
        self._nodes[node_id].reported_config = dict(config)
        self._nodes[node_id].updated_at = datetime.now(UTC)

    async def record_heartbeat(self, node_id, state, heartbeat, lease_expires_at) -> None:
        n = self._nodes[node_id]
        n.state = state
        n.last_heartbeat = dict(heartbeat)
        n.lease_expires_at = lease_expires_at
        n.updated_at = datetime.now(UTC)

    async def set_node_state(self, node_id, state) -> None:
        self._nodes[node_id].state = state
        self._nodes[node_id].updated_at = datetime.now(UTC)

    async def set_node_state_id(self, node_id, node_state_id) -> None:
        self._nodes[node_id].node_state_id = node_state_id
        self._nodes[node_id].updated_at = datetime.now(UTC)

    async def expire_leases(self, now, dead_after_s) -> list[tuple[str, str]]:
        changed = []
        async with self._lock:
            for n in self._nodes.values():
                if n.lease_expires_at is None or n.state == DEAD:
                    continue
                if n.lease_expires_at + timedelta(seconds=dead_after_s) < now:
                    n.state = DEAD
                    changed.append((n.node_id, DEAD))
                elif n.lease_expires_at < now and n.state != SUSPECT:
                    n.state = SUSPECT
                    changed.append((n.node_id, SUSPECT))
        return changed

    async def register_aggregator(self, node_id) -> int:
        async with self._lock:
            self._agg.active_aggregator_id = node_id
            self._agg.epoch += 1
            return self._agg.epoch

    async def get_aggregator_state(self) -> AggregatorState:
        s = copy.deepcopy(self._agg)
        if s.active_aggregator_id and s.active_aggregator_id in self._nodes:
            s.active_aggregator_addr = self._nodes[s.active_aggregator_id].advertise_addr
        if s.latest_checkpoint_id and s.latest_checkpoint_id in self._checkpoints:
            s.latest_checkpoint_uri = self._checkpoints[s.latest_checkpoint_id].uri
        return s

    async def commit_checkpoint(self, req: CommitRequest, enforce_epoch: bool = False) -> CommitResult:
        async with self._lock:
            s = self._agg
            ok = s.active_aggregator_id == req.aggregator_id and (not enforce_epoch or s.epoch == req.epoch)
            if req.initial:
                ok = ok and s.latest_checkpoint_id is None
                new_version = 0
            else:
                ok = ok and s.latest_checkpoint_id is not None and s.global_version == req.parent_version
                new_version = s.global_version + 1
            if not ok or req.checkpoint_id in self._checkpoints:
                reason = "duplicate_checkpoint_id" if ok else rejection_reason(s, req, enforce_epoch)
                return CommitResult(False, s.global_version, reason)
            s.latest_checkpoint_id = req.checkpoint_id
            s.global_version = new_version
            self._checkpoints[req.checkpoint_id] = CheckpointRecord(
                req.checkpoint_id,
                new_version,
                None if req.initial else req.parent_version,
                req.uri,
                req.aggregator_id,
                req.epoch,
                list(req.delta_ids),
                datetime.now(UTC),
            )
            for r in req.delta_records:
                self._deltas[r.delta_id] = copy.copy(r)
            for d in req.delta_ids:
                if d in self._deltas:
                    self._deltas[d].status = "applied"
        for q in list(self._subscribers):
            q.put_nowait(req.checkpoint_id)
        return CommitResult(True, new_version, "")

    async def get_checkpoint(self, checkpoint_id) -> CheckpointRecord | None:
        c = self._checkpoints.get(checkpoint_id)
        return copy.deepcopy(c) if c else None

    async def list_checkpoints(self, limit=100) -> list[CheckpointRecord]:
        rows = sorted(self._checkpoints.values(), key=lambda c: c.global_version, reverse=True)
        return [copy.deepcopy(c) for c in rows[:limit]]

    async def prune_checkpoints(self, keep) -> list[CheckpointRecord]:
        async with self._lock:
            rows = sorted(self._checkpoints.values(), key=lambda c: c.global_version, reverse=True)
            pruned = rows[keep:]
            for c in pruned:
                del self._checkpoints[c.checkpoint_id]
            return pruned

    async def list_deltas(self) -> list[DeltaRecord]:
        return [copy.copy(d) for d in self._deltas.values()]

    def subscribe_checkpoints(self) -> AsyncIterator[str]:
        # Register the queue eagerly so commits made before the first ``__anext__`` are not missed.
        q: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers.append(q)

        async def gen():
            try:
                while True:
                    yield await q.get()
            finally:
                self._subscribers.remove(q)

        return gen()
