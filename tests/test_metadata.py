"""MetadataStore contract tests. Postgres runs when FERRY_TEST_POSTGRES_DSN is set."""

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest

from ferry.metadata import CommitRequest, DeltaRecord, InMemoryMetadataStore

PG_DSN = os.environ.get("FERRY_TEST_POSTGRES_DSN")


async def _reset_pg(dsn):
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute("DROP TABLE IF EXISTS deltas, checkpoints, aggregator_state, nodes CASCADE")


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.skipif(not PG_DSN, reason="no PG"))])
async def store(request):
    if request.param == "memory":
        s = InMemoryMetadataStore()
    else:
        from ferry.metadata.postgres import PostgresMetadataStore

        await _reset_pg(PG_DSN)
        s = PostgresMetadataStore(PG_DSN)
    await s.init()
    yield s
    await s.close()
    if request.param == "postgres":
        await _reset_pg(PG_DSN)


def _req(agg, cid, parent, epoch=1, initial=False, deltas=()):
    return CommitRequest(agg, epoch, cid, f"file:///x/{cid}", parent, list(deltas), [], initial)


async def test_nodes_and_leases(store):
    n = await store.upsert_node("w1", "worker", None, "JOINING")
    assert n.node_id == "w1" and n.node_state_id is None
    await store.set_node_state_id("w1", "file:///state/1")
    await store.set_reported_config("w1", {"h": 10})
    resumed = await store.upsert_node("w1", "worker", None, "JOINING")
    assert resumed.node_state_id == "file:///state/1" and resumed.reported_config == {"h": 10}
    with pytest.raises(ValueError):
        await store.upsert_node("w1", "aggregator", "x:1", "JOINING")

    now = datetime.now(UTC)
    await store.record_heartbeat("w1", "TRAINING", {"loss": 1.0}, now - timedelta(seconds=1))
    assert await store.expire_leases(now, dead_after_s=15) == [("w1", "SUSPECT")]
    assert await store.expire_leases(now, dead_after_s=15) == []
    assert await store.expire_leases(now + timedelta(seconds=20), dead_after_s=15) == [("w1", "DEAD")]


async def test_commit_is_conditional(store):
    await store.upsert_node("a1", "aggregator", "h:1", "JOINING")
    await store.upsert_node("a2", "aggregator", "h:2", "JOINING")
    assert await store.register_aggregator("a1") == 1
    sub = store.subscribe_checkpoints()
    nxt = asyncio.ensure_future(anext(sub))
    await asyncio.sleep(0.2)  # let LISTEN start on Postgres

    r = await store.commit_checkpoint(_req("a1", "c0", 0, initial=True))
    assert r.accepted and r.global_version == 0
    assert await asyncio.wait_for(nxt, 5) == "c0"
    assert not (await store.commit_checkpoint(_req("a1", "c0b", 0, initial=True))).accepted

    store_deltas = [DeltaRecord("d1", "w1", 0, 10, 100, "accepted")]
    req = _req("a1", "c1", 0, deltas=["d1"])
    req.delta_records = store_deltas
    r = await store.commit_checkpoint(req)
    assert r.accepted and r.global_version == 1
    assert [(d.delta_id, d.status) for d in await store.list_deltas()] == [("d1", "applied")]

    # Same parent twice → second is rejected (prevents two aggregators committing on the same version).
    r = await store.commit_checkpoint(_req("a1", "c1b", 0))
    assert not r.accepted and r.reason.startswith("version_conflict")

    # Non-active aggregator is rejected.
    r = await store.commit_checkpoint(_req("a2", "c2x", 1))
    assert not r.accepted and r.reason == "not_active_aggregator"

    # Epoch fencing (DESIGN §8.3) only when enabled.
    assert await store.register_aggregator("a1") == 2
    r = await store.commit_checkpoint(_req("a1", "c2", 1, epoch=1), enforce_epoch=True)
    assert not r.accepted and r.reason == "stale_epoch"
    r = await store.commit_checkpoint(_req("a1", "c2", 1, epoch=1), enforce_epoch=False)
    assert r.accepted and r.global_version == 2

    st = await store.get_aggregator_state()
    assert (st.latest_checkpoint_id, st.global_version, st.active_aggregator_addr) == ("c2", 2, "h:1")
    assert st.latest_checkpoint_uri == "file:///x/c2"
    assert [c.checkpoint_id for c in await store.list_checkpoints()] == ["c2", "c1", "c0"]
    pruned = await store.prune_checkpoints(keep=2)
    assert [c.checkpoint_id for c in pruned] == ["c0"]
    await sub.aclose()
