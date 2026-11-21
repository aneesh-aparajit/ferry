"""Component configuration with the defaults from DESIGN §9."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CoordinatorConfig:
    listen_addr: str = "0.0.0.0:7070"
    postgres_dsn: str = ""  # required in production; "memory://" uses the in-memory store
    heartbeat_interval_s: float = 5.0
    describe_timeout_s: float = 10.0
    checkpoint_retention: int = 20
    # Optional: blob store the coordinator may delete GC'd checkpoints from. Without it, nothing is deleted.
    gc_blob_store: str | None = None
    # DESIGN §8.3: split-brain fencing. Off in v1; when on, commits must carry the current epoch.
    enforce_epoch_fencing: bool = False
    metrics_port: int | None = None

    @property
    def lease_ttl_s(self) -> float:
        return 3 * self.heartbeat_interval_s


@dataclass
class AggregatorConfig:
    coordinator_addr: str = ""
    advertise_addr: str = ""
    listen_addr: str = ""  # defaults to advertise_addr's port on 0.0.0.0
    blob_store: str = "file://./ferry-data"
    aggregator: str = "staleness_weighted_mean"
    outer_optimizer: str = "nesterov(lr=0.7, momentum=0.9)"
    aggregation_policy: str = "buffer_k(k=4, max_wait_s=30, max_staleness=8)"
    node_id: str = ""
    heartbeat_interval_s: float = 5.0
    commit_retry_s: float = 2.0
    metrics_port: int | None = None

    def resolved_listen_addr(self) -> str:
        if self.listen_addr:
            return self.listen_addr
        port = self.advertise_addr.rsplit(":", 1)[-1]
        return f"0.0.0.0:{port}"


@dataclass
class WorkerConfig:
    coordinator_addr: str = ""
    node_id: str = ""
    h: int = 500
    inner_lr: float = 0.0  # informational: reported to the coordinator
    batch_size: int = 0  # informational
    device: str = "cpu"
    blob_store: str = "file://./ferry-data"
    state_persist_every: int = 1  # in local runs
    heartbeat_interval_s: float = 5.0
    push_timeout_s: float = 60.0
    reconnect_backoff_s: float = 1.0
    reconnect_backoff_max_s: float = 30.0
    extra: dict[str, str] = field(default_factory=dict)
    metrics_port: int | None = None
