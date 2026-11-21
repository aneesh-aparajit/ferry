"""In-process cluster tests: coordinator + aggregator + workers over real gRPC on localhost."""

import asyncio
import socket
import time

import grpc
import pytest
import torch

from examples.linreg import eval_loss, make_model, make_task
from ferry.aggregator import AggregatorNode
from ferry.blobstore import read_blob
from ferry.config import AggregatorConfig, CoordinatorConfig, WorkerConfig
from ferry.coordinator import Coordinator
from ferry.metadata import InMemoryMetadataStore
from ferry.serialization import decode_checkpoint
from ferry.types import Delta, DeltaMeta
from ferry.v1 import ferry_pb2 as pb
from ferry.v1 import ferry_pb2_grpc as pb_grpc
from ferry.worker import Worker

HB = 0.2


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def wait_for(pred, timeout=20.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not met in time")


@pytest.fixture
def ports():
    return free_port(), free_port()


async def start_coordinator(port, store=None, **kw):
    c = Coordinator(
        CoordinatorConfig(listen_addr=f"127.0.0.1:{port}", heartbeat_interval_s=HB, **kw),
        store=store or InMemoryMetadataStore(),
    )
    await c.start()
    return c


def agg_config(cport, aport, tmp_path, **kw):
    base = dict(
        coordinator_addr=f"127.0.0.1:{cport}",
        advertise_addr=f"127.0.0.1:{aport}",
        blob_store=f"file://{tmp_path}",
        heartbeat_interval_s=HB,
        commit_retry_s=0.2,
        aggregation_policy="buffer_k(k=2, max_wait_s=0.5, max_staleness=8)",
    )
    base.update(kw)
    return AggregatorConfig(**base)


def worker_config(cport, tmp_path, h, **kw):
    return WorkerConfig(
        coordinator_addr=f"127.0.0.1:{cport}",
        h=h,
        blob_store=f"file://{tmp_path}",
        heartbeat_interval_s=HB,
        reconnect_backoff_s=0.1,
        reconnect_backoff_max_s=0.5,
        push_timeout_s=5,
        **kw,
    )


async def test_end_to_end_training_heterogeneous_workers(ports, tmp_path):
    cport, aport = ports
    coord = await start_coordinator(cport)
    # FedAvg: Nesterov momentum overshoots on this tiny convex toy over only a handful of outer steps.
    agg = AggregatorNode(agg_config(cport, aport, tmp_path, outer_optimizer="sgd(lr=1.0)"), make_model)
    await agg.start()
    assert agg.committed_version == 0
    initial = eval_loss(agg.params)

    w1 = Worker(worker_config(cport, tmp_path, h=10), make_task(seed=1))
    w2 = Worker(worker_config(cport, tmp_path, h=25), make_task(seed=2))
    await asyncio.wait_for(asyncio.gather(w1.run(max_runs=6), w2.run(max_runs=4)), 60)
    await wait_for(lambda: not agg.buffer and not agg._pending)

    assert w1.pushes_ok + w1.pushes_rejected == 6 and w2.pushes_ok + w2.pushes_rejected == 4
    assert agg.committed_version >= 3
    final = eval_loss(agg.params)
    assert final < initial * 0.5, (initial, final)

    store = coord.store
    st = await store.get_aggregator_state()
    assert st.global_version == agg.committed_version
    ckpt = decode_checkpoint(read_blob(agg.blob, st.latest_checkpoint_uri))
    assert ckpt.global_version == st.global_version

    nodes = {n.node_id: n for n in await store.list_nodes()}
    assert nodes[w1.node_id].reported_config["h"] == 10
    assert nodes[w2.node_id].reported_config["h"] == 25
    assert nodes[w1.node_id].node_state_id  # persisted resumable state
    deltas = await store.list_deltas()
    assert len(deltas) >= 8 and {d.status for d in deltas} <= {"applied", "rejected", "accepted"}

    await agg.stop()
    await coord.stop()


async def test_worker_keeps_training_when_aggregator_down(ports, tmp_path):
    """DESIGN §7.1 step 6 / §8.2: failed push → keep θ_local and base, push an accumulated delta later."""
    cport, aport = ports
    coord = await start_coordinator(cport)
    cfg = agg_config(
        cport,
        aport,
        tmp_path,
        node_id="agg-1",
        aggregation_policy="buffer_k(k=1, max_wait_s=0.1, max_staleness=8)",
        outer_optimizer="sgd",
    )
    a1 = AggregatorNode(cfg, make_model)
    await a1.start()
    await a1.stop()  # aggregator goes away; coordinator still advertises it

    task = make_task(seed=3)
    inner = task.train_step

    def slow_step():
        time.sleep(0.01)
        return inner()

    task.train_step = slow_step
    h = 5
    w = Worker(worker_config(cport, tmp_path, h=h), task)
    wt = asyncio.create_task(w.run())
    await wait_for(lambda: w.pushes_failed >= 2)

    a2 = AggregatorNode(cfg, make_model)  # same node_id, restores from v0
    await a2.start()
    assert a2.epoch == 2 and a2.committed_version == 0
    await wait_for(lambda: w.pushes_ok >= 1)
    assert w.last_push_inner_steps >= 3 * h and w.last_push_inner_steps % h == 0
    await wait_for(lambda: a2.committed_version >= 1)

    await w.stop()
    await wt
    await a2.stop()
    await coord.stop()


async def test_worker_resumes_inner_state(ports, tmp_path):
    cport, aport = ports
    coord = await start_coordinator(cport)
    agg = AggregatorNode(agg_config(cport, aport, tmp_path), make_model)
    await agg.start()

    t1 = make_task(seed=4)
    w1 = Worker(worker_config(cport, tmp_path, h=7, node_id="w-fixed"), t1)
    await w1.run(max_runs=1)
    node = await coord.store.get_node("w-fixed")
    assert node.node_state_id

    t2 = make_task(seed=4)
    w2 = Worker(worker_config(cport, tmp_path, h=7, node_id="w-fixed"), t2)
    w2.config.h = 0  # don't train: just register, restore, load
    run = asyncio.create_task(w2.run(max_runs=1))
    await asyncio.wait_for(run, 20)
    s1 = t1.optimizer.state_dict()["state"][0]
    s2 = t2.optimizer.state_dict()["state"][0]
    assert float(s2["step"]) == float(s1["step"]) == 7
    assert torch.equal(s1["exp_avg"], s2["exp_avg"])
    # Params come from the global checkpoint, never from the worker's own old weights.
    for k, v in t2.model.state_dict().items():
        assert torch.equal(v, agg.params[k]) or agg.committed_version > 0

    await agg.stop()
    await coord.stop()


async def test_describe_timeout_closes_session(ports):
    cport, _ = ports
    coord = await start_coordinator(cport, describe_timeout_s=0.3)
    async with grpc.aio.insecure_channel(f"127.0.0.1:{cport}") as ch:
        stub = pb_grpc.CoordinatorStub(ch)

        async def reqs():
            yield pb.NodeMsg(register=pb.Register(type=pb.NODE_TYPE_WORKER))
            await asyncio.sleep(10)

        kinds = [m.WhichOneof("msg") async for m in stub.Session(reqs())]
    assert kinds[0] == "register_ack" and "describe" in kinds and kinds[-1] == "shutdown"
    await coord.stop()


async def test_first_message_must_be_register(ports):
    cport, _ = ports
    coord = await start_coordinator(cport)
    async with grpc.aio.insecure_channel(f"127.0.0.1:{cport}") as ch:
        stub = pb_grpc.CoordinatorStub(ch)

        async def reqs():
            yield pb.NodeMsg(heartbeat=pb.Heartbeat())

        with pytest.raises(grpc.aio.AioRpcError) as e:
            async for _ in stub.Session(reqs()):
                pass
        assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    await coord.stop()


async def test_aggregator_buffers_commits_while_coordinator_down(ports, tmp_path):
    """DESIGN §8.2 'Coordinator down': aggregator keeps checkpoints persisted and commits when it returns."""
    cport, aport = ports
    store = InMemoryMetadataStore()
    c1 = await start_coordinator(cport, store=store)
    agg = AggregatorNode(agg_config(cport, aport, tmp_path, outer_optimizer="sgd"), make_model)
    await agg.start()
    await c1.stop()

    keys = list(agg.params)
    agg.buffer.append(
        Delta(DeltaMeta("d1", "w", "c", 0, 10, 10), {k: torch.ones_like(agg.params[k]) for k in keys}, received_at=0)
    )
    await agg.apply_now()
    assert agg.version == 1 and agg.committed_version == 0 and len(agg._pending) == 1

    c2 = await start_coordinator(cport, store=store)
    await wait_for(lambda: agg.committed_version == 1)
    assert (await store.get_aggregator_state()).global_version == 1

    await agg.stop()
    await c2.stop()


async def test_new_aggregator_replaces_old(ports, tmp_path):
    cport, aport = ports
    coord = await start_coordinator(cport)
    a1 = AggregatorNode(agg_config(cport, aport, tmp_path), make_model)
    await a1.start()
    a2 = AggregatorNode(agg_config(cport, free_port(), tmp_path), make_model)
    await a2.start()
    await wait_for(lambda: a1._stopped.is_set())
    assert "replaced" in (a1.shutdown_reason or "")
    assert a2.epoch == a1.epoch + 1 and a2.committed_version == 0
    await a2.stop()
    await coord.stop()
