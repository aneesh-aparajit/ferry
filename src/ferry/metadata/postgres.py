"""Postgres ``MetadataStore`` (DESIGN §5). Uses psycopg 3 (async) and ``LISTEN/NOTIFY`` for checkpoint fan-out."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from importlib import resources

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from ferry.metadata.base import (
    CHECKPOINT_CHANNEL,
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

log = logging.getLogger(__name__)

_NODE_COLS = (
    "node_id, node_type, state, advertise_addr, reported_config, last_heartbeat, lease_expires_at, "
    "node_state_id, registered_at, updated_at"
)
_CKPT_COLS = "checkpoint_id, global_version, parent_version, uri, aggregator_id, epoch, delta_ids, created_at"


def _node(row: dict) -> NodeRecord:
    return NodeRecord(**row)


def _ckpt(row: dict) -> CheckpointRecord:
    return CheckpointRecord(**{**row, "delta_ids": list(row["delta_ids"] or [])})


class PostgresMetadataStore:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 8) -> None:
        self.dsn = dsn
        self.pool = AsyncConnectionPool(
            dsn, min_size=min_size, max_size=max_size, open=False, kwargs={"row_factory": dict_row}
        )

    async def init(self) -> None:
        await self.pool.open(wait=True)
        ddl = resources.files("ferry.metadata").joinpath("schema.sql").read_text()
        async with self.pool.connection() as conn:
            await conn.execute(ddl)

    async def close(self) -> None:
        await self.pool.close()

    # ───────────── nodes ─────────────

    async def upsert_node(self, node_id, node_type, advertise_addr, state) -> NodeRecord:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"""
                INSERT INTO nodes (node_id, node_type, state, advertise_addr)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (node_id) DO UPDATE
                   SET state = EXCLUDED.state, advertise_addr = EXCLUDED.advertise_addr, updated_at = now()
                 WHERE nodes.node_type = EXCLUDED.node_type
                RETURNING {_NODE_COLS}
                """,
                (node_id, node_type, state, advertise_addr),
            )
            row = await cur.fetchone()
        if row is None:
            raise ValueError(f"node {node_id} exists with a different node_type than {node_type}")
        return _node(row)

    async def get_node(self, node_id) -> NodeRecord | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(f"SELECT {_NODE_COLS} FROM nodes WHERE node_id = %s", (node_id,))
            row = await cur.fetchone()
        return _node(row) if row else None

    async def list_nodes(self) -> list[NodeRecord]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(f"SELECT {_NODE_COLS} FROM nodes ORDER BY registered_at")
            return [_node(r) for r in await cur.fetchall()]

    async def _update(self, sql: str, params: tuple) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(sql, params)

    async def set_reported_config(self, node_id, config) -> None:
        await self._update(
            "UPDATE nodes SET reported_config = %s, updated_at = now() WHERE node_id = %s", (Jsonb(config), node_id)
        )

    async def record_heartbeat(self, node_id, state, heartbeat, lease_expires_at) -> None:
        await self._update(
            "UPDATE nodes SET state = %s, last_heartbeat = %s, lease_expires_at = %s, updated_at = now() "
            "WHERE node_id = %s",
            (state, Jsonb(heartbeat), lease_expires_at, node_id),
        )

    async def set_node_state(self, node_id, state) -> None:
        await self._update("UPDATE nodes SET state = %s, updated_at = now() WHERE node_id = %s", (state, node_id))

    async def set_node_state_id(self, node_id, node_state_id) -> None:
        await self._update(
            "UPDATE nodes SET node_state_id = %s, updated_at = now() WHERE node_id = %s", (node_state_id, node_id)
        )

    async def expire_leases(self, now: datetime, dead_after_s: float) -> list[tuple[str, str]]:
        dead_cutoff = now - timedelta(seconds=dead_after_s)
        async with self.pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                "UPDATE nodes SET state = %s, updated_at = now() "
                "WHERE lease_expires_at < %s AND state <> %s RETURNING node_id",
                (DEAD, dead_cutoff, DEAD),
            )
            dead = [(r["node_id"], DEAD) for r in await cur.fetchall()]
            cur = await conn.execute(
                "UPDATE nodes SET state = %s, updated_at = now() "
                "WHERE lease_expires_at < %s AND state NOT IN (%s, %s) RETURNING node_id",
                (SUSPECT, now, SUSPECT, DEAD),
            )
            suspect = [(r["node_id"], SUSPECT) for r in await cur.fetchall()]
        return dead + suspect

    # ───────────── aggregator / checkpoints ─────────────

    async def register_aggregator(self, node_id) -> int:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE aggregator_state SET active_aggregator_id = %s, epoch = epoch + 1, updated_at = now() "
                "RETURNING epoch",
                (node_id,),
            )
            return (await cur.fetchone())["epoch"]

    async def get_aggregator_state(self) -> AggregatorState:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT s.active_aggregator_id, s.epoch, s.latest_checkpoint_id, s.global_version,
                       n.advertise_addr AS active_aggregator_addr, c.uri AS latest_checkpoint_uri
                  FROM aggregator_state s
                  LEFT JOIN nodes n ON n.node_id = s.active_aggregator_id
                  LEFT JOIN checkpoints c ON c.checkpoint_id = s.latest_checkpoint_id
                """
            )
            return AggregatorState(**await cur.fetchone())

    async def commit_checkpoint(self, req: CommitRequest, enforce_epoch: bool = False) -> CommitResult:
        """DESIGN §5.1: conditional UPDATE + INSERT + NOTIFY in one transaction."""
        epoch_clause = " AND epoch = %(epoch)s" if enforce_epoch else ""
        if req.initial:
            update = (
                "UPDATE aggregator_state SET latest_checkpoint_id = %(cid)s, global_version = 0, updated_at = now() "
                "WHERE active_aggregator_id = %(agg)s AND latest_checkpoint_id IS NULL"
            )
        else:
            update = (
                "UPDATE aggregator_state SET latest_checkpoint_id = %(cid)s, global_version = global_version + 1, "
                "updated_at = now() "
                "WHERE active_aggregator_id = %(agg)s AND latest_checkpoint_id IS NOT NULL "
                "AND global_version = %(parent)s"
            )
        params = {"cid": req.checkpoint_id, "agg": req.aggregator_id, "parent": req.parent_version, "epoch": req.epoch}
        try:
            async with self.pool.connection() as conn, conn.transaction():
                cur = await conn.execute(update + epoch_clause + " RETURNING global_version", params)
                row = await cur.fetchone()
                if row is None:
                    raise _Rejected()
                version = row["global_version"]
                await conn.execute(
                    f"INSERT INTO checkpoints ({_CKPT_COLS.replace(', created_at', '')}) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        req.checkpoint_id,
                        version,
                        None if req.initial else req.parent_version,
                        req.uri,
                        req.aggregator_id,
                        req.epoch,
                        list(req.delta_ids),
                    ),
                )
                for r in req.delta_records:
                    await conn.execute(
                        "INSERT INTO deltas (delta_id, worker_id, base_global_version, inner_steps, num_samples, "
                        "status, reason) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (delta_id) DO UPDATE SET status = EXCLUDED.status, reason = EXCLUDED.reason",
                        (
                            r.delta_id,
                            r.worker_id,
                            r.base_global_version,
                            r.inner_steps,
                            r.num_samples,
                            r.status,
                            r.reason,
                        ),
                    )
                if req.delta_ids:
                    await conn.execute(
                        "UPDATE deltas SET status = 'applied' WHERE delta_id = ANY(%s)", (list(req.delta_ids),)
                    )
                await conn.execute("SELECT pg_notify(%s, %s)", (CHECKPOINT_CHANNEL, req.checkpoint_id))
        except _Rejected:
            state = await self.get_aggregator_state()
            return CommitResult(False, state.global_version, rejection_reason(state, req, enforce_epoch))
        except psycopg.errors.UniqueViolation:
            state = await self.get_aggregator_state()
            return CommitResult(False, state.global_version, "duplicate_checkpoint_id")
        return CommitResult(True, version, "")

    async def get_checkpoint(self, checkpoint_id) -> CheckpointRecord | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(f"SELECT {_CKPT_COLS} FROM checkpoints WHERE checkpoint_id = %s", (checkpoint_id,))
            row = await cur.fetchone()
        return _ckpt(row) if row else None

    async def list_checkpoints(self, limit=100) -> list[CheckpointRecord]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_CKPT_COLS} FROM checkpoints ORDER BY global_version DESC LIMIT %s", (limit,)
            )
            return [_ckpt(r) for r in await cur.fetchall()]

    async def prune_checkpoints(self, keep) -> list[CheckpointRecord]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"""
                DELETE FROM checkpoints WHERE checkpoint_id IN (
                    SELECT checkpoint_id FROM checkpoints ORDER BY global_version DESC OFFSET %s
                ) AND checkpoint_id <> (SELECT COALESCE(latest_checkpoint_id, '') FROM aggregator_state)
                RETURNING {_CKPT_COLS}
                """,
                (keep,),
            )
            return [_ckpt(r) for r in await cur.fetchall()]

    async def list_deltas(self) -> list[DeltaRecord]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT delta_id, worker_id, base_global_version, inner_steps, num_samples, status, reason "
                "FROM deltas ORDER BY received_at"
            )
            return [DeltaRecord(**r) for r in await cur.fetchall()]

    async def subscribe_checkpoints(self) -> AsyncIterator[str]:
        """``LISTEN ferry_checkpoint`` on a dedicated connection; reconnects on failure."""
        while True:
            try:
                async with await psycopg.AsyncConnection.connect(self.dsn, autocommit=True) as conn:
                    await conn.execute(f"LISTEN {CHECKPOINT_CHANNEL}")
                    async for note in conn.notifies():
                        yield note.payload
            except (psycopg.OperationalError, OSError) as e:
                log.warning("checkpoint LISTEN connection lost (%s); reconnecting", e)
                await asyncio.sleep(1.0)


class _Rejected(Exception):
    pass
