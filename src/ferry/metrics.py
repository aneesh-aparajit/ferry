"""Prometheus metrics (DESIGN §10). Defined once at module level so several components can share a process."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Worker
inner_steps_total = Counter("ferry_worker_inner_steps_total", "Inner optimizer steps", ["node_id"])
samples_per_sec = Gauge("ferry_worker_samples_per_sec", "Training throughput", ["node_id"])
local_loss = Gauge("ferry_worker_local_loss", "Latest local loss", ["node_id"])
push_duration_seconds = Histogram("ferry_worker_push_duration_seconds", "PushDelta duration", ["node_id"])
push_bytes = Counter("ferry_worker_push_bytes", "Delta bytes pushed", ["node_id"])
push_failures_total = Counter("ferry_worker_push_failures_total", "Failed pushes", ["node_id"])
time_blocked_seconds = Counter(
    "ferry_worker_time_blocked_seconds", "Time a worker spent unable to train (should stay ~0)", ["node_id"]
)

# Aggregator
deltas_received_total = Counter("ferry_aggregator_deltas_received_total", "Deltas received", ["status"])
delta_staleness = Histogram(
    "ferry_aggregator_delta_staleness", "Staleness of received deltas", buckets=(0, 1, 2, 3, 4, 6, 8, 12, 16, 32)
)
outer_steps_total = Counter("ferry_aggregator_outer_steps_total", "Outer optimizer steps")
checkpoint_write_seconds = Histogram("ferry_aggregator_checkpoint_write_seconds", "Checkpoint blob write time")
global_version = Gauge("ferry_aggregator_global_version", "Latest committed global version")

# Coordinator
nodes = Gauge("ferry_coordinator_nodes", "Nodes by type and state", ["type", "state"])
lease_expirations_total = Counter("ferry_coordinator_lease_expirations_total", "Lease expirations", ["state"])
commit_rejections_total = Counter("ferry_coordinator_commit_rejections_total", "Rejected checkpoint commits")


def maybe_start_server(port: int | None) -> None:
    if port:
        start_http_server(port)
