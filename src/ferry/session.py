"""Client side of ``Coordinator.Session``: register, heartbeat, reconnect with backoff (DESIGN §4.1)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import Awaitable, Callable

import grpc

from ferry import __version__
from ferry.rpc import channel
from ferry.v1 import ferry_pb2 as pb
from ferry.v1 import ferry_pb2_grpc as pb_grpc

log = logging.getLogger(__name__)

_STOP = object()


class SessionClient:
    """Keeps one long-lived Session open. All writes go through a single outbox so the stream has one writer."""

    def __init__(
        self,
        addr: str,
        node_type: int,
        on_message: Callable[[pb.CoordinatorMsg], Awaitable[None]],
        heartbeat: Callable[[], pb.Heartbeat],
        *,
        node_id: str = "",
        advertise_addr: str = "",
        heartbeat_interval_s: float = 5.0,
        backoff_s: float = 1.0,
        backoff_max_s: float = 30.0,
    ) -> None:
        self.addr = addr
        self.node_type = node_type
        self.node_id = node_id
        self.advertise_addr = advertise_addr
        self.on_message = on_message
        self.heartbeat = heartbeat
        self.heartbeat_interval_s = heartbeat_interval_s
        self.backoff_s = backoff_s
        self.backoff_max_s = backoff_max_s
        self.registered = asyncio.Event()  # set while a session is registered
        self.ever_registered = asyncio.Event()
        self.last_state_id: str = ""
        self.connects = 0
        self._outbox: asyncio.Queue | None = None
        self._stopping = False
        self._call = None

    def send(self, msg: pb.NodeMsg) -> bool:
        """Queue ``msg`` on the current session. Returns False if disconnected (message dropped)."""
        if self._outbox is None or not self.registered.is_set():
            return False
        self._outbox.put_nowait(msg)
        return True

    def send_state_update(self, node_state_id: str) -> None:
        self.last_state_id = node_state_id
        self.send(pb.NodeMsg(state=pb.StateUpdate(node_state_id=node_state_id)))

    def reconnect(self) -> None:
        """Drop the current session; ``run`` re-registers with the same node_id."""
        if self._call is not None:
            self._call.cancel()

    async def stop(self) -> None:
        self._stopping = True
        if self._outbox is not None:
            self._outbox.put_nowait(_STOP)
        if self._call is not None:
            self._call.cancel()

    async def run(self) -> None:
        delay = self.backoff_s
        while not self._stopping:
            try:
                await self._run_once()
                delay = self.backoff_s
            except asyncio.CancelledError:
                raise
            except grpc.aio.AioRpcError as e:
                log.info("session to %s lost: %s", self.addr, e.code().name)
            except Exception:
                log.exception("session error")
            finally:
                self.registered.clear()
                self._outbox = None
                self._call = None
            if self._stopping:
                break
            await asyncio.sleep(delay * (0.5 + random.random()))
            delay = min(delay * 2, self.backoff_max_s)

    async def _run_once(self) -> None:
        outbox: asyncio.Queue = asyncio.Queue()
        self._outbox = outbox

        async def requests():
            yield pb.NodeMsg(
                register=pb.Register(
                    type=self.node_type, node_id=self.node_id, advertise_addr=self.advertise_addr, version=__version__
                )
            )
            while True:
                msg = await outbox.get()
                if msg is _STOP:
                    return
                yield msg

        async with channel(self.addr) as ch:
            stub = pb_grpc.CoordinatorStub(ch)
            call = stub.Session(requests())
            self._call = call
            hb_task = None
            try:
                async for msg in self._iterate(call):
                    if msg.WhichOneof("msg") == "register_ack":
                        self.node_id = msg.register_ack.node_id
                        self.connects += 1
                        self.registered.set()
                        self.ever_registered.set()
                        if self.last_state_id:
                            outbox.put_nowait(pb.NodeMsg(state=pb.StateUpdate(node_state_id=self.last_state_id)))
                        hb_task = asyncio.create_task(self._heartbeats(outbox))
                    await self.on_message(msg)
                    if msg.WhichOneof("msg") == "shutdown":
                        log.warning("coordinator requested shutdown: %s", msg.shutdown.reason)
            finally:
                if hb_task:
                    hb_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await hb_task

    @staticmethod
    async def _iterate(call):
        """Iterate a call; a local ``call.cancel()`` (reconnect/stop) ends iteration instead of cancelling us."""
        try:
            async for msg in call:
                yield msg
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise

    async def _heartbeats(self, outbox: asyncio.Queue) -> None:
        while True:
            try:
                outbox.put_nowait(pb.NodeMsg(heartbeat=self.heartbeat()))
            except Exception:
                log.exception("failed to build heartbeat")
            await asyncio.sleep(self.heartbeat_interval_s)
