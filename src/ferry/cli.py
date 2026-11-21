"""``ferry coordinator | aggregator | worker`` entry points."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

from ferry.config import AggregatorConfig, CoordinatorConfig, WorkerConfig


def load_object(ref: str) -> Callable[..., Any]:
    """Resolve ``package.module:attr``. The current directory is importable (for ``examples.*``)."""
    if ":" not in ref:
        raise SystemExit(f"expected module:attr, got {ref!r}")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    mod, attr = ref.split(":", 1)
    return getattr(importlib.import_module(mod), attr)


def _kv(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        out[k] = v
    return out


async def _run_coordinator(a: argparse.Namespace) -> None:
    from ferry.coordinator import Coordinator

    dsn = a.postgres_dsn or os.environ.get("FERRY_POSTGRES_DSN", "")
    if not dsn:
        raise SystemExit("--postgres-dsn is required (or pass --postgres-dsn memory:// for an in-memory store)")
    c = Coordinator(
        CoordinatorConfig(
            listen_addr=a.listen,
            postgres_dsn=dsn,
            heartbeat_interval_s=a.heartbeat_interval,
            describe_timeout_s=a.describe_timeout,
            checkpoint_retention=a.checkpoint_retention,
            gc_blob_store=a.gc_blob_store,
            enforce_epoch_fencing=a.enforce_epoch_fencing,
            metrics_port=a.metrics_port,
        )
    )
    await c.start()
    try:
        await c.wait()
    finally:
        await c.stop()


async def _run_aggregator(a: argparse.Namespace) -> None:
    from ferry.aggregator import AggregatorNode

    node = AggregatorNode(
        AggregatorConfig(
            coordinator_addr=a.coordinator,
            advertise_addr=a.advertise,
            listen_addr=a.listen or "",
            blob_store=a.blob_store,
            aggregator=a.aggregator,
            outer_optimizer=a.outer_optimizer,
            aggregation_policy=a.policy,
            node_id=a.node_id or "",
            heartbeat_interval_s=a.heartbeat_interval,
            metrics_port=a.metrics_port,
        ),
        load_object(a.model) if a.model else None,
    )
    await node.start()
    try:
        await node.wait()
    finally:
        await node.stop()


async def _run_worker(a: argparse.Namespace) -> None:
    from ferry.worker import Worker

    task = load_object(a.task)(**_kv(a.task_arg))
    w = Worker(
        WorkerConfig(
            coordinator_addr=a.coordinator,
            node_id=a.node_id or "",
            h=a.h,
            inner_lr=a.inner_lr,
            batch_size=a.batch_size,
            device=a.device,
            blob_store=a.blob_store,
            state_persist_every=a.state_persist_every,
            heartbeat_interval_s=a.heartbeat_interval,
            extra=_kv(a.extra),
            metrics_port=a.metrics_port,
        ),
        task,
    )
    await w.run(max_runs=a.max_runs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ferry", description=__doc__)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("coordinator", help="run the control plane")
    c.add_argument("--listen", default="0.0.0.0:7070")
    c.add_argument("--postgres-dsn", default="")
    c.add_argument("--heartbeat-interval", type=float, default=5.0)
    c.add_argument("--describe-timeout", type=float, default=10.0)
    c.add_argument("--checkpoint-retention", type=int, default=20)
    c.add_argument("--gc-blob-store", default=None, help="blob store to delete GC'd checkpoints from")
    c.add_argument("--enforce-epoch-fencing", action="store_true")
    c.add_argument("--metrics-port", type=int, default=None)
    c.set_defaults(fn=_run_coordinator)

    g = sub.add_parser("aggregator", help="run the aggregator")
    g.add_argument("--coordinator", required=True)
    g.add_argument("--advertise", required=True, help="host:port workers dial")
    g.add_argument("--listen", default=None)
    g.add_argument("--model", default=None, help="module:factory returning an nn.Module or state dict (for v0)")
    g.add_argument("--blob-store", default="file://./ferry-data")
    g.add_argument("--aggregator", default="staleness_weighted_mean")
    g.add_argument("--outer-optimizer", default="nesterov(lr=0.7, momentum=0.9)")
    g.add_argument("--policy", default="buffer_k(k=4, max_wait_s=30, max_staleness=8)")
    g.add_argument("--node-id", default=None)
    g.add_argument("--heartbeat-interval", type=float, default=5.0)
    g.add_argument("--metrics-port", type=int, default=None)
    g.set_defaults(fn=_run_aggregator)

    w = sub.add_parser("worker", help="run a training worker")
    w.add_argument("--coordinator", required=True)
    w.add_argument("--task", required=True, help="module:factory returning a ferry.worker.TrainTask")
    w.add_argument("--task-arg", action="append", default=[], help="key=value passed to the task factory")
    w.add_argument("--node-id", default=None)
    w.add_argument("--h", type=int, default=500)
    w.add_argument("--inner-lr", type=float, default=0.0)
    w.add_argument("--batch-size", type=int, default=0)
    w.add_argument("--device", default="cpu")
    w.add_argument("--blob-store", default="file://./ferry-data")
    w.add_argument("--state-persist-every", type=int, default=1)
    w.add_argument("--heartbeat-interval", type=float, default=5.0)
    w.add_argument("--extra", action="append", default=[], help="key=value reported in DescribeResponse.extra")
    w.add_argument("--max-runs", type=int, default=None)
    w.add_argument("--metrics-port", type=int, default=None)
    w.set_defaults(fn=_run_worker)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(args.fn(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
