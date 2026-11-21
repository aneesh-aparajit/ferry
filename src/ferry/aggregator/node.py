"""Aggregator: buffers worker deltas, applies the outer optimizer, persists then publishes checkpoints (DESIGN §7.2)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import grpc
import torch

from ferry import metrics
from ferry.blobstore import BlobStore, open_blob_store, read_blob
from ferry.config import AggregatorConfig
from ferry.metadata.base import DeltaRecord
from ferry.rpc import channel, server
from ferry.serialization import (
    Checkpoint,
    IntegrityError,
    Reassembler,
    decode_checkpoint,
    encode_checkpoint,
    loads,
    sha256_hex,
)
from ferry.session import SessionClient
from ferry.strategies import (
    AggregationPolicy,
    Aggregator,
    OuterOptimizer,
    build_aggregator,
    build_outer_optimizer,
    build_policy,
)
from ferry.types import Delta, DeltaMeta, StateDict
from ferry.v1 import ferry_pb2 as pb
from ferry.v1 import ferry_pb2_grpc as pb_grpc

log = logging.getLogger(__name__)

ModelFactory = Callable[[], "torch.nn.Module | StateDict"]

_SEEN_DELTA_CAPACITY = 100_000
_REJECT_FENCED = ("not_active_aggregator", "stale_epoch")


@dataclass
class _PendingCommit:
    checkpoint_id: str
    uri: str
    version: int
    delta_ids: list[str]
    initial: bool


class AggregatorShutdown(RuntimeError):
    """Raised (and logged) when this aggregator has been replaced and must stop."""


class AggregatorNode(pb_grpc.AggregatorServicer):
    def __init__(
        self,
        config: AggregatorConfig,
        model_factory: ModelFactory | None = None,
        *,
        blob_store: BlobStore | None = None,
        aggregator: Aggregator | None = None,
        outer_optimizer: OuterOptimizer | None = None,
        policy: AggregationPolicy | None = None,
    ) -> None:
        if not config.coordinator_addr or not config.advertise_addr:
            raise ValueError("aggregator requires coordinator_addr and advertise_addr")
        self.config = config
        self.model_factory = model_factory
        self.blob = blob_store or open_blob_store(config.blob_store)
        self.aggregator = aggregator or build_aggregator(config.aggregator)
        self.outer = outer_optimizer or build_outer_optimizer(config.outer_optimizer)
        self.policy = policy or build_policy(config.aggregation_policy)

        self.params: StateDict = {}
        self.version = -1  # local head (may be ahead of the coordinator while commits are pending)
        self.committed_version = -1
        self.committed_id = ""
        self.committed_uri = ""
        self.epoch = 0
        self.buffer: list[Delta] = []
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._audit: list[DeltaRecord] = []
        self._pending: list[_PendingCommit] = []
        self._ack: pb.RegisterAck | None = None
        self._ack_event = asyncio.Event()
        self._wake = asyncio.Event()
        self._state = pb.NODE_STATE_JOINING
        self._ready = asyncio.Event()
        self._stopped = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._server: grpc.aio.Server | None = None
        self._coord_channel: grpc.aio.Channel | None = None
        self._commit_lock = asyncio.Lock()
        self.shutdown_reason: str | None = None

        self.session = SessionClient(
            config.coordinator_addr,
            pb.NODE_TYPE_AGGREGATOR,
            self._on_coordinator_msg,
            self._heartbeat,
            node_id=config.node_id,
            advertise_addr=config.advertise_addr,
            heartbeat_interval_s=config.heartbeat_interval_s,
        )

    @property
    def node_id(self) -> str:
        return self.session.node_id

    # ───────────── lifecycle ─────────────

    async def start(self) -> None:
        self._server = server()
        pb_grpc.add_AggregatorServicer_to_server(self, self._server)
        self._server.add_insecure_port(self.config.resolved_listen_addr())
        await self._server.start()
        self._coord_channel = channel(self.config.coordinator_addr)
        self._tasks.append(asyncio.create_task(self.session.run()))
        await self._ack_event.wait()
        await self._restore_or_init(self._ack)
        self._tasks.append(asyncio.create_task(self._apply_loop()))
        self._tasks.append(asyncio.create_task(self._commit_loop()))
        metrics.maybe_start_server(self.config.metrics_port)
        self._ready.set()
        log.info("aggregator %s ready at version %d (epoch %d)", self.node_id, self.version, self.epoch)

    async def wait(self) -> None:
        await self._stopped.wait()

    async def stop(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        await self.session.stop()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        if self._server:
            await self._server.stop(0.5)
        if self._coord_channel:
            await self._coord_channel.close()

    # ───────────── coordinator session ─────────────

    async def _on_coordinator_msg(self, msg: pb.CoordinatorMsg) -> None:
        kind = msg.WhichOneof("msg")
        if kind == "register_ack":
            ack = msg.register_ack
            self.epoch = ack.epoch
            if self._ready.is_set():
                await self._on_reregister(ack)
            self._ack = ack
            self._ack_event.set()
        elif kind == "shutdown" and msg.shutdown.reason.startswith("replaced by aggregator"):
            # Another aggregator registered. v1 runs exactly one aggregator (DESIGN §8.3): stop.
            self.shutdown_reason = msg.shutdown.reason
            log.error("aggregator %s shutting down: %s", self.node_id, msg.shutdown.reason)
            asyncio.get_running_loop().create_task(self.stop())

    async def _on_reregister(self, ack: pb.RegisterAck) -> None:
        """Session came back. Keep local state if nobody else committed in the meantime; otherwise restore."""
        if ack.latest_checkpoint_id == self.committed_id or (not ack.latest_checkpoint_id and self.committed_id == ""):
            log.info("re-registered (epoch %d); %d pending commit(s) to flush", ack.epoch, len(self._pending))
            self._wake.set()
            return
        log.warning("coordinator head moved to %s while disconnected; restoring", ack.latest_checkpoint_id)
        await self._restore_or_init(ack)

    def _heartbeat(self) -> pb.Heartbeat:
        return pb.Heartbeat(state=self._state, base_global_version=max(self.version, 0))

    # ───────────── restore / init ─────────────

    async def _restore_or_init(self, ack: pb.RegisterAck) -> None:
        self.buffer.clear()
        self._pending.clear()
        if ack.latest_checkpoint_id:
            data = await asyncio.to_thread(read_blob, self.blob, ack.latest_checkpoint_uri)
            ckpt = decode_checkpoint(data)
            if ckpt.global_version != ack.latest_global_version:
                log.warning(
                    "checkpoint blob version %d != registry version %d; trusting registry",
                    ckpt.global_version,
                    ack.latest_global_version,
                )
            self.params = ckpt.params
            self.outer.load_state_dict(ckpt.outer_state)
            self.policy.load_state_dict(ckpt.policy_state)
            self.version = self.committed_version = ack.latest_global_version
            self.committed_id, self.committed_uri = ack.latest_checkpoint_id, ack.latest_checkpoint_uri
            metrics.global_version.set(self.version)
            log.info("restored global params + outer state from %s (v%d)", ack.latest_checkpoint_id, self.version)
            return
        if self.model_factory is None:
            raise RuntimeError("no checkpoint exists and no model_factory was given to initialize version 0")
        init = self.model_factory()
        sd = init.state_dict() if isinstance(init, torch.nn.Module) else init
        self.params = {k: v.detach().clone().cpu() for k, v in sd.items()}
        self.version = 0
        await self._persist(0, delta_ids=[], initial=True)
        await self._flush_commits()
        if self.committed_version < 0:
            # Could not commit v0 right now (coordinator down): the commit loop keeps retrying.
            log.warning("version 0 is pending commit")

    # ───────────── PushDelta ─────────────

    async def PushDelta(self, request_iterator, context):  # noqa: N802
        await self._ready.wait()
        header: pb.DeltaHeader | None = None
        reasm: Reassembler | None = None
        try:
            async for chunk in request_iterator:
                if chunk.seq == 0:
                    header = chunk.header
                    if not header.delta_id:
                        return self._ack_msg(False, "missing_header")
                    if header.delta_id in self._seen:
                        metrics.deltas_received_total.labels(status="duplicate").inc()
                        return self._ack_msg(True, "duplicate")
                    reasm = Reassembler(header.total_bytes, header.sha256)
                if reasm is None:
                    return self._ack_msg(False, "missing_header")
                reasm.add(chunk.seq, chunk.data)
                if chunk.last:
                    break
            if reasm is None or header is None:
                return self._ack_msg(False, "empty_stream")
            data = reasm.finish()
        except IntegrityError as e:
            metrics.deltas_received_total.labels(status="rejected").inc()
            return self._ack_msg(False, f"integrity_error: {e}")

        meta = DeltaMeta.from_proto(header)
        ok, reason = self.policy.accept(meta, self.version)
        tensors: StateDict = {}
        if ok:
            try:
                tensors = await asyncio.to_thread(loads, data)
            except Exception as e:
                ok, reason = False, f"invalid_safetensors: {e}"
        if ok:
            ok, reason = self._check_shapes(tensors)

        self._remember(meta.delta_id)
        metrics.delta_staleness.observe(max(0, self.version - meta.base_global_version))
        status = "accepted" if ok else "rejected"
        metrics.deltas_received_total.labels(status=status).inc()
        self._audit.append(
            DeltaRecord(
                meta.delta_id,
                meta.worker_id,
                meta.base_global_version,
                meta.inner_steps,
                meta.num_samples,
                status,
                reason or None,
            )
        )
        if not ok:
            log.info("rejected delta %s from %s: %s", meta.delta_id, meta.worker_id, reason)
            return self._ack_msg(False, reason)
        self.buffer.append(Delta(meta, tensors, received_at=time.monotonic()))
        self._wake.set()
        return self._ack_msg(True, "")

    def _check_shapes(self, tensors: StateDict) -> tuple[bool, str]:
        if set(tensors) != set(self.params):
            return False, "key_mismatch"
        for k, t in tensors.items():
            if t.shape != self.params[k].shape:
                return False, f"shape_mismatch: {k}"
        return True, ""

    def _remember(self, delta_id: str) -> None:
        self._seen[delta_id] = None
        while len(self._seen) > _SEEN_DELTA_CAPACITY:
            self._seen.popitem(last=False)

    def _ack_msg(self, accepted: bool, reason: str) -> pb.PushDeltaAck:
        return pb.PushDeltaAck(
            accepted=accepted,
            reason=reason,
            latest_checkpoint_id=self.committed_id,
            latest_global_version=max(self.committed_version, 0),
            latest_checkpoint_uri=self.committed_uri,
        )

    # ───────────── apply / persist / commit ─────────────

    async def _apply_loop(self) -> None:
        tick = 0.1
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), tick)
            except TimeoutError:
                pass
            self._wake.clear()
            if self.policy.should_apply(self.buffer, time.monotonic()):
                await self.apply_now()

    async def apply_now(self) -> None:
        """Aggregate the buffer, take one outer step, persist the checkpoint and queue its commit."""
        if not self.buffer:
            return
        batch, self.buffer = self.buffer, []
        self._state = pb.NODE_STATE_AGGREGATING
        try:
            outer_grad = self.aggregator.aggregate(batch, self.version)
            self.params = self.outer.step(self.params, outer_grad)
            self.version += 1
            metrics.outer_steps_total.inc()
            await self._persist(self.version, [d.meta.delta_id for d in batch], initial=False)
            log.info("outer step → v%d from %d delta(s)", self.version, len(batch))
        finally:
            self._state = pb.NODE_STATE_IDLE
        self._wake.set()
        await self._flush_commits()

    async def _persist(self, version: int, delta_ids: list[str], initial: bool) -> None:
        """Write ``{params, outer_state, global_version, policy_state}`` under an immutable key (DESIGN §7.2)."""
        blob = encode_checkpoint(Checkpoint(self.params, self.outer.state_dict(), version, self.policy.state_dict()))
        short = sha256_hex(blob)[:12]
        cid = f"{version:08d}-{short}"
        t0 = time.perf_counter()
        uri = await asyncio.to_thread(self.blob.put, f"checkpoints/{cid}.safetensors", blob)
        metrics.checkpoint_write_seconds.observe(time.perf_counter() - t0)
        self._pending.append(_PendingCommit(cid, uri, version, delta_ids, initial))

    async def _commit_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.commit_retry_s)
            if self._pending:
                await self._flush_commits()

    async def _flush_commits(self) -> None:
        """Publish pending checkpoints to the coordinator, in order. Unreachable coordinator → retry later."""
        async with self._commit_lock:
            await self._flush_commits_locked()

    async def _flush_commits_locked(self) -> None:
        assert self._coord_channel is not None
        stub = pb_grpc.CoordinatorStub(self._coord_channel)
        while self._pending:
            p = self._pending[0]
            audit = list(self._audit)
            req = pb.CommitCheckpointRequest(
                aggregator_id=self.node_id,
                epoch=self.epoch,
                checkpoint_id=p.checkpoint_id,
                uri=p.uri,
                parent_version=0 if p.initial else p.version - 1,
                delta_ids=p.delta_ids,
                delta_records=[
                    pb.DeltaRecord(
                        delta_id=r.delta_id,
                        worker_id=r.worker_id,
                        base_global_version=r.base_global_version,
                        inner_steps=r.inner_steps,
                        num_samples=r.num_samples,
                        status=r.status,
                        reason=r.reason or "",
                    )
                    for r in audit
                ],
                initial=p.initial,
            )
            try:
                res = await stub.CommitCheckpoint(req, timeout=10)
            except grpc.aio.AioRpcError as e:
                log.warning("commit of %s failed (%s); will retry", p.checkpoint_id, e.code().name)
                return
            if res.accepted:
                self._pending.pop(0)
                del self._audit[: len(audit)]
                self.committed_version = res.global_version
                self.committed_id, self.committed_uri = p.checkpoint_id, p.uri
                metrics.global_version.set(res.global_version)
                continue
            log.error("commit of %s rejected: %s", p.checkpoint_id, res.reason)
            if res.reason.startswith(_REJECT_FENCED):
                self.shutdown_reason = res.reason
                asyncio.get_running_loop().create_task(self.stop())
                return
            # Someone else moved the head (or v0 already exists): drop local work and restore from the registry.
            await self._resync()
            return

    async def _resync(self) -> None:
        """Re-register to fetch the current head; ``_on_reregister`` restores from it."""
        self._pending.clear()
        self.session.reconnect()


def new_delta_id() -> str:
    return str(uuid.uuid4())
