"""Worker: inner loop of ``H`` steps, push delta, pull latest, never block on peers (DESIGN §7.1)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import grpc
import torch

from ferry import metrics
from ferry.blobstore import BlobStore, open_blob_store, read_blob
from ferry.config import WorkerConfig
from ferry.rpc import channel
from ferry.serialization import (
    chunk,
    decode_checkpoint,
    decode_optimizer_state,
    dumps,
    encode_optimizer_state,
    sha256_hex,
)
from ferry.session import SessionClient
from ferry.types import StateDict
from ferry.v1 import ferry_pb2 as pb
from ferry.v1 import ferry_pb2_grpc as pb_grpc

log = logging.getLogger(__name__)


class WorkerStateHandler(Protocol):
    """Implemented by the user's training code. Persists and restores resumable worker state."""

    def save(self) -> bytes: ...  # inner optimizer state, data cursor, RNG state, step counters

    def restore(self, blob: bytes) -> None: ...


class OptimizerStateHandler:
    """Default handler: inner optimizer state + torch RNG + a user ``extra`` dict, as safetensors (no pickle)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        get_extra: Callable[[], dict] | None = None,
        set_extra: Callable[[dict], None] | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.get_extra = get_extra
        self.set_extra = set_extra

    def save(self) -> bytes:
        extra = {
            "torch_rng": base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode(),
            "user": self.get_extra() if self.get_extra else {},
        }
        return encode_optimizer_state(self.optimizer.state_dict(), extra)

    def restore(self, blob: bytes) -> None:
        state, extra = decode_optimizer_state(blob)
        self.optimizer.load_state_dict(state)
        if rng := extra.get("torch_rng"):
            torch.set_rng_state(torch.frombuffer(bytearray(base64.b64decode(rng)), dtype=torch.uint8))
        if self.set_extra and "user" in extra:
            self.set_extra(extra["user"])


@dataclass
class TrainTask:
    """What the user hands a worker.

    ``train_step()`` runs one inner optimizer step on ``model`` and returns ``(loss, num_samples)``.
    """

    model: torch.nn.Module
    train_step: Callable[[], tuple[float, int]]
    state_handler: WorkerStateHandler | None = None
    optimizer: torch.optim.Optimizer | None = None
    describe: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state_handler is None and self.optimizer is not None:
            self.state_handler = OptimizerStateHandler(self.optimizer)


@dataclass
class _CkptRef:
    checkpoint_id: str
    version: int
    uri: str


class Worker:
    def __init__(self, config: WorkerConfig, task: TrainTask, *, blob_store: BlobStore | None = None) -> None:
        if not config.coordinator_addr:
            raise ValueError("worker requires coordinator_addr")
        self.config = config
        self.task = task
        self.blob = blob_store or open_blob_store(config.blob_store)

        self.latest: _CkptRef | None = None
        self.latest_changed = asyncio.Event()
        self.aggregator_addr: str = ""
        self.aggregator_epoch = 0
        self.node_state_id = ""
        self._restored = False
        self._stopping = threading.Event()
        self._stopped = asyncio.Event()

        # progress (read by heartbeats; written by the training thread)
        self.state = pb.NODE_STATE_JOINING
        self.base_version = 0
        self.inner_step = 0
        self.loss = 0.0
        self.samples_per_sec = 0.0

        # counters (useful for tests and logs)
        self.runs = 0
        self.pushes_ok = 0
        self.pushes_failed = 0
        self.pushes_rejected = 0
        self.last_push_inner_steps = 0

        self._agg_channel: grpc.aio.Channel | None = None
        self._agg_channel_addr = ""
        self._tasks: list[asyncio.Task] = []
        self.session = SessionClient(
            config.coordinator_addr,
            pb.NODE_TYPE_WORKER,
            self._on_coordinator_msg,
            self._heartbeat,
            node_id=config.node_id,
            heartbeat_interval_s=config.heartbeat_interval_s,
            backoff_s=config.reconnect_backoff_s,
            backoff_max_s=config.reconnect_backoff_max_s,
        )

    @property
    def node_id(self) -> str:
        return self.session.node_id

    @property
    def _labels(self) -> dict:
        return {"node_id": self.node_id or "unregistered"}

    # ───────────── coordinator session ─────────────

    async def _on_coordinator_msg(self, msg: pb.CoordinatorMsg) -> None:
        kind = msg.WhichOneof("msg")
        if kind == "register_ack":
            ack = msg.register_ack
            self.node_state_id = ack.node_state_id
            if ack.latest_checkpoint_id:
                self._offer(_CkptRef(ack.latest_checkpoint_id, ack.latest_global_version, ack.latest_checkpoint_uri))
        elif kind == "aggregator":
            if msg.aggregator.addr != self.aggregator_addr:
                log.info("aggregator is now %s (epoch %d)", msg.aggregator.addr, msg.aggregator.epoch)
            self.aggregator_addr, self.aggregator_epoch = msg.aggregator.addr, msg.aggregator.epoch
        elif kind == "describe":
            c = self.config
            self.session.send(
                pb.NodeMsg(
                    describe=pb.DescribeResponse(
                        h=c.h,
                        inner_lr=c.inner_lr,
                        batch_size=c.batch_size,
                        device=c.device,
                        extra={**c.extra, **self.task.describe},
                    )
                )
            )
        elif kind == "new_checkpoint":
            nc = msg.new_checkpoint
            # Recorded, never interrupts the current local run (DESIGN §7.1).
            self._offer(_CkptRef(nc.checkpoint_id, nc.global_version, nc.uri))
        elif kind == "shutdown":
            log.warning("coordinator closed our session: %s", msg.shutdown.reason)

    def _offer(self, ref: _CkptRef) -> None:
        if self.latest is None or ref.version > self.latest.version:
            self.latest = ref
            self.latest_changed.set()

    def _heartbeat(self) -> pb.Heartbeat:
        return pb.Heartbeat(
            state=self.state,
            base_global_version=self.base_version,
            inner_step=self.inner_step,
            loss=self.loss,
            samples_per_sec=self.samples_per_sec,
        )

    # ───────────── lifecycle ─────────────

    async def run(self, max_runs: int | None = None) -> None:
        """Main loop. ``max_runs`` bounds the number of local runs (pushes attempted); ``None`` runs forever."""
        metrics.maybe_start_server(self.config.metrics_port)
        self._tasks.append(asyncio.create_task(self.session.run()))
        try:
            await self._loop(max_runs)
        finally:
            await self.stop()

    async def stop(self) -> None:
        self._stopping.set()
        await self.session.stop()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks.clear()
        if self._agg_channel is not None:
            await self._agg_channel.close()
            self._agg_channel = None
        self._stopped.set()

    async def _loop(self, max_runs: int | None) -> None:
        await self.session.ever_registered.wait()
        await self._maybe_restore()
        base = await self._load_latest()
        theta_base = {k: v.detach().clone() for k, v in self.task.model.state_dict().items()}
        steps_since_base = 0
        samples_since_base = 0

        while not self._stopping.is_set() and (max_runs is None or self.runs < max_runs):
            # 4. Train H inner steps.
            self.state = pb.NODE_STATE_TRAINING
            loss, samples = await asyncio.to_thread(self._run_steps, self.config.h)
            if self._stopping.is_set():
                break
            steps_since_base += self.config.h
            samples_since_base += samples
            self.runs += 1

            # 5. Push θ_local − θ_base.
            self.state = pb.NODE_STATE_UPLOADING
            ack = await self._push(base, theta_base, steps_since_base, samples_since_base, loss)
            self.state = pb.NODE_STATE_IDLE

            if ack is None:
                # 6b. Aggregator unreachable: keep θ_local and the same base, train more, push the larger delta.
                self.pushes_failed += 1
                metrics.push_failures_total.labels(**self._labels).inc()
                log.info("push failed; continuing from base v%d (%d steps accumulated)", base.version, steps_since_base)
                continue

            if ack.accepted:
                self.pushes_ok += 1
            else:
                self.pushes_rejected += 1
                log.info("delta rejected (%s); discarding and pulling latest", ack.reason)
            if ack.latest_checkpoint_id:
                self._offer(_CkptRef(ack.latest_checkpoint_id, ack.latest_global_version, ack.latest_checkpoint_uri))

            if self.config.state_persist_every and self.runs % self.config.state_persist_every == 0:
                await self._persist_state()

            # 6a. Pull the newest checkpoint we know of and start the next local run from it.
            base = await self._load_latest()
            theta_base = {k: v.detach().clone() for k, v in self.task.model.state_dict().items()}
            steps_since_base = samples_since_base = 0

    def _run_steps(self, n: int) -> tuple[float, int]:
        total_samples = 0
        loss = 0.0
        t0 = time.perf_counter()
        for i in range(n):
            if self._stopping.is_set():
                break
            loss, ns = self.task.train_step()
            loss = float(loss)
            total_samples += int(ns)
            self.inner_step = i + 1
            self.loss = loss
            metrics.inner_steps_total.labels(**self._labels).inc()
        dt = max(time.perf_counter() - t0, 1e-9)
        self.samples_per_sec = total_samples / dt
        metrics.samples_per_sec.labels(**self._labels).set(self.samples_per_sec)
        metrics.local_loss.labels(**self._labels).set(loss)
        return loss, total_samples

    # ───────────── checkpoints / state ─────────────

    async def _maybe_restore(self) -> None:
        if self._restored or not self.node_state_id or self.task.state_handler is None:
            return
        try:
            blob = await asyncio.to_thread(read_blob, self.blob, self.node_state_id)
            self.task.state_handler.restore(blob)
            self.session.last_state_id = self.node_state_id
            log.info("restored worker state from %s", self.node_state_id)
        except Exception:
            log.exception("failed to restore worker state %s; starting fresh", self.node_state_id)
        self._restored = True

    async def _load_latest(self) -> _CkptRef:
        """Load model params from the latest global checkpoint (never from our own old weights)."""
        t0 = time.perf_counter()
        while self.latest is None:
            self.state = pb.NODE_STATE_IDLE
            self.latest_changed.clear()
            await self.latest_changed.wait()
        blocked = time.perf_counter() - t0
        if blocked > 0.01:
            metrics.time_blocked_seconds.labels(**self._labels).inc(blocked)
        ref = self.latest
        data = await asyncio.to_thread(read_blob, self.blob, ref.uri)
        ckpt = decode_checkpoint(data)
        self.task.model.load_state_dict(ckpt.params)
        self.base_version = ref.version
        self.inner_step = 0
        return ref

    async def _persist_state(self) -> None:
        if self.task.state_handler is None:
            return
        try:
            blob = self.task.state_handler.save()
            key = f"nodes/{self.node_id}/state-{self.runs:08d}.safetensors"
            uri = await asyncio.to_thread(self.blob.put, key, blob)
            self.session.send_state_update(uri)
        except Exception:
            log.exception("failed to persist worker state")

    # ───────────── push ─────────────

    def _channel(self) -> grpc.aio.Channel | None:
        if not self.aggregator_addr:
            return None
        if self._agg_channel is None or self._agg_channel_addr != self.aggregator_addr:
            if self._agg_channel is not None:
                asyncio.get_running_loop().create_task(self._agg_channel.close())
            self._agg_channel = channel(self.aggregator_addr)
            self._agg_channel_addr = self.aggregator_addr
        return self._agg_channel

    async def _push(
        self, base: _CkptRef, theta_base: StateDict, steps: int, samples: int, loss: float
    ) -> pb.PushDeltaAck | None:
        ch = self._channel()
        if ch is None:
            return None
        local = self.task.model.state_dict()
        delta = {k: (local[k].detach().cpu() - theta_base[k].cpu()) for k in theta_base}
        data = await asyncio.to_thread(dumps, delta)
        header = pb.DeltaHeader(
            delta_id=str(uuid.uuid4()),
            worker_id=self.node_id,
            base_checkpoint_id=base.checkpoint_id,
            base_global_version=base.version,
            inner_steps=steps,
            num_samples=samples,
            local_loss=loss,
            total_bytes=len(data),
            sha256=sha256_hex(data),
        )
        parts = list(chunk(data))

        async def chunks():
            for i, part in enumerate(parts):
                yield pb.DeltaChunk(header=header if i == 0 else None, seq=i, data=part, last=i == len(parts) - 1)

        t0 = time.perf_counter()
        try:
            ack = await pb_grpc.AggregatorStub(ch).PushDelta(chunks(), timeout=self.config.push_timeout_s)
        except grpc.aio.AioRpcError as e:
            log.info("PushDelta to %s failed: %s", self.aggregator_addr, e.code().name)
            return None
        metrics.push_duration_seconds.labels(**self._labels).observe(time.perf_counter() - t0)
        metrics.push_bytes.labels(**self._labels).inc(len(data))
        self.last_push_inner_steps = steps
        return ack
