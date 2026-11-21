"""Coordinator: membership, health, leases and the checkpoint registry (DESIGN §3.1). Holds no tensors."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import grpc
from google.protobuf.json_format import MessageToDict

from ferry import __version__, metrics
from ferry.blobstore import open_blob_store
from ferry.config import CoordinatorConfig
from ferry.metadata import CommitRequest, DeltaRecord, MetadataStore, open_metadata_store
from ferry.metadata.base import AGGREGATOR, DEAD, JOINING, WORKER
from ferry.rpc import node_type_name, server, state_name
from ferry.v1 import ferry_pb2 as pb
from ferry.v1 import ferry_pb2_grpc as pb_grpc

log = logging.getLogger(__name__)

_CLOSE = object()


@dataclass(eq=False)
class _Session:
    node_id: str
    node_type: str
    outbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    described: asyncio.Event = field(default_factory=asyncio.Event)
    closed: bool = False

    def send(self, msg: pb.CoordinatorMsg) -> None:
        if not self.closed:
            self.outbox.put_nowait(msg)

    def close(self, reason: str | None = None) -> None:
        if self.closed:
            return
        if reason:
            self.outbox.put_nowait(pb.CoordinatorMsg(shutdown=pb.Shutdown(reason=reason)))
        self.closed = True
        self.outbox.put_nowait(_CLOSE)


class CoordinatorService(pb_grpc.CoordinatorServicer):
    def __init__(self, store: MetadataStore, config: CoordinatorConfig) -> None:
        self.store = store
        self.config = config
        self.sessions: dict[str, _Session] = {}
        self._gc_store = open_blob_store(config.gc_blob_store) if config.gc_blob_store else None

    # ───────────── Session ─────────────

    async def Session(self, request_iterator, context):  # noqa: N802 (gRPC naming)
        try:
            first = await anext(aiter(request_iterator))
        except StopAsyncIteration:
            return
        if first.WhichOneof("msg") != "register":
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "first message on Session must be Register")
        reg = first.register
        node_type = node_type_name(reg.type)
        if not node_type:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Register.type must be WORKER or AGGREGATOR")
        if node_type == AGGREGATOR and not reg.advertise_addr:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "aggregator must set advertise_addr")

        node_id = reg.node_id or f"{node_type[0]}-{uuid.uuid4().hex[:12]}"
        try:
            node = await self.store.upsert_node(node_id, node_type, reg.advertise_addr or None, JOINING)
        except ValueError as e:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        await self.store.record_heartbeat(node_id, JOINING, {}, self._lease_deadline())

        epoch = 0
        if node_type == AGGREGATOR:
            epoch = await self.store.register_aggregator(node_id)
        agg = await self.store.get_aggregator_state()

        sess = _Session(node_id, node_type)
        old = self.sessions.get(node_id)
        if old is not None:
            old.close("superseded by a new session for the same node_id")
        self.sessions[node_id] = sess
        log.info("registered %s %s (version %s, epoch %s)", node_type, node_id, reg.version, epoch or agg.epoch)

        if node_type == AGGREGATOR:
            self._on_new_aggregator(node_id, reg.advertise_addr, epoch)

        sess.send(
            pb.CoordinatorMsg(
                register_ack=pb.RegisterAck(
                    node_id=node_id,
                    epoch=epoch,
                    latest_checkpoint_id=agg.latest_checkpoint_id or "",
                    latest_global_version=agg.global_version,
                    node_state_id=node.node_state_id or "",
                    latest_checkpoint_uri=agg.latest_checkpoint_uri or "",
                )
            )
        )
        if node_type == WORKER:
            if agg.active_aggregator_addr:
                sess.send(
                    pb.CoordinatorMsg(aggregator=pb.AggregatorInfo(addr=agg.active_aggregator_addr, epoch=agg.epoch))
                )
            sess.send(pb.CoordinatorMsg(describe=pb.DescribeRequest()))

        reader = asyncio.create_task(self._read(sess, request_iterator))
        watchdog = asyncio.create_task(self._describe_watchdog(sess)) if node_type == WORKER else None
        try:
            while True:
                msg = await sess.outbox.get()
                if msg is _CLOSE:
                    break
                yield msg
        finally:
            reader.cancel()
            if watchdog:
                watchdog.cancel()
            if self.sessions.get(node_id) is sess:
                del self.sessions[node_id]
            log.info("session closed for %s", node_id)

    async def _read(self, sess: _Session, request_iterator) -> None:
        try:
            async for msg in request_iterator:
                await self._handle(sess, msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # stream broken
            log.debug("session reader for %s ended: %s", sess.node_id, e)
        finally:
            sess.close()

    async def _handle(self, sess: _Session, msg: pb.NodeMsg) -> None:
        kind = msg.WhichOneof("msg")
        if kind == "heartbeat":
            hb = msg.heartbeat
            await self.store.record_heartbeat(
                sess.node_id,
                state_name(hb.state) if hb.state else "IDLE",
                MessageToDict(hb, preserving_proto_field_name=True),
                self._lease_deadline(),
            )
        elif kind == "describe":
            await self.store.set_reported_config(
                sess.node_id, MessageToDict(msg.describe, preserving_proto_field_name=True)
            )
            sess.described.set()
        elif kind == "state":
            await self.store.set_node_state_id(sess.node_id, msg.state.node_state_id)
        elif kind == "register":
            log.warning("duplicate Register from %s ignored", sess.node_id)

    async def _describe_watchdog(self, sess: _Session) -> None:
        """DESIGN §4.1: a worker must answer DescribeRequest within ``describe_timeout``."""
        try:
            await asyncio.wait_for(sess.described.wait(), self.config.describe_timeout_s)
        except TimeoutError:
            log.warning("worker %s did not answer DescribeRequest in time; closing session", sess.node_id)
            sess.close("describe_timeout")

    def _lease_deadline(self) -> datetime:
        return datetime.now(UTC) + timedelta(seconds=self.config.lease_ttl_s)

    def _on_new_aggregator(self, node_id: str, addr: str, epoch: int) -> None:
        info = pb.CoordinatorMsg(aggregator=pb.AggregatorInfo(addr=addr, epoch=epoch))
        for s in list(self.sessions.values()):
            if s.node_type == WORKER:
                s.send(info)
            elif s.node_type == AGGREGATOR and s.node_id != node_id:
                s.close(f"replaced by aggregator {node_id} (epoch {epoch})")

    # ───────────── CommitCheckpoint ─────────────

    async def CommitCheckpoint(self, request, context):  # noqa: N802
        req = CommitRequest(
            aggregator_id=request.aggregator_id,
            epoch=request.epoch,
            checkpoint_id=request.checkpoint_id,
            uri=request.uri,
            parent_version=request.parent_version,
            delta_ids=list(request.delta_ids),
            delta_records=[
                DeltaRecord(
                    r.delta_id,
                    r.worker_id,
                    r.base_global_version,
                    r.inner_steps,
                    r.num_samples,
                    r.status,
                    r.reason or None,
                )
                for r in request.delta_records
            ],
            initial=request.initial,
        )
        res = await self.store.commit_checkpoint(req, enforce_epoch=self.config.enforce_epoch_fencing)
        if not res.accepted:
            metrics.commit_rejections_total.inc()
            log.warning("rejected commit %s from %s: %s", req.checkpoint_id, req.aggregator_id, res.reason)
        else:
            log.info("committed checkpoint %s as version %d", req.checkpoint_id, res.global_version)
        return pb.CommitCheckpointResponse(accepted=res.accepted, global_version=res.global_version, reason=res.reason)

    # ───────────── background loops ─────────────

    async def checkpoint_watcher(self) -> None:
        """Fan ``NewCheckpoint`` out to workers for every commit (``LISTEN ferry_checkpoint``)."""
        async for cid in self.store.subscribe_checkpoints():
            rec = await self.store.get_checkpoint(cid)
            if rec is None:
                continue
            msg = pb.CoordinatorMsg(
                new_checkpoint=pb.NewCheckpoint(
                    checkpoint_id=rec.checkpoint_id, global_version=rec.global_version, uri=rec.uri
                )
            )
            for s in list(self.sessions.values()):
                if s.node_type == WORKER:
                    s.send(msg)
            await self._gc()

    async def _gc(self) -> None:
        if self._gc_store is None or self.config.checkpoint_retention <= 0:
            return
        for rec in await self.store.prune_checkpoints(self.config.checkpoint_retention):
            try:
                await asyncio.to_thread(self._gc_store.delete, rec.uri)
            except Exception as e:
                log.warning("failed to GC checkpoint blob %s: %s", rec.uri, e)

    async def lease_sweeper(self) -> None:
        """Missed lease → SUSPECT; 2 × TTL → DEAD (DESIGN §8.1)."""
        while True:
            await asyncio.sleep(self.config.heartbeat_interval_s)
            try:
                changed = await self.store.expire_leases(datetime.now(UTC), self.config.lease_ttl_s)
                for node_id, state in changed:
                    metrics.lease_expirations_total.labels(state=state).inc()
                    log.info("node %s lease expired → %s", node_id, state)
                    if state == DEAD and node_id in self.sessions:
                        self.sessions[node_id].close("lease expired")
                counts: dict[tuple[str, str], int] = {}
                for n in await self.store.list_nodes():
                    counts[(n.node_type, n.state)] = counts.get((n.node_type, n.state), 0) + 1
                metrics.nodes.clear()
                for (t, s), c in counts.items():
                    metrics.nodes.labels(type=t, state=s).set(c)
            except Exception:
                log.exception("lease sweep failed")


class Coordinator:
    """Runs the coordinator gRPC server and its background loops."""

    def __init__(self, config: CoordinatorConfig, store: MetadataStore | None = None) -> None:
        self.config = config
        self.store = store or open_metadata_store(config.postgres_dsn or "memory://")
        self.service = CoordinatorService(self.store, config)
        self._server: grpc.aio.Server | None = None
        self._tasks: list[asyncio.Task] = []
        self.port: int | None = None

    async def start(self) -> None:
        await self.store.init()
        self._server = server()
        pb_grpc.add_CoordinatorServicer_to_server(self.service, self._server)
        self.port = self._server.add_insecure_port(self.config.listen_addr)
        await self._server.start()
        self._tasks = [
            asyncio.create_task(self.service.checkpoint_watcher()),
            asyncio.create_task(self.service.lease_sweeper()),
        ]
        metrics.maybe_start_server(self.config.metrics_port)
        log.info("coordinator %s listening on %s", __version__, self.config.listen_addr)

    async def wait(self) -> None:
        assert self._server is not None
        await self._server.wait_for_termination()

    async def stop(self, grace: float = 0.5) -> None:
        for s in list(self.service.sessions.values()):
            s.close()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        if self._server:
            await self._server.stop(grace)
        await self.store.close()
