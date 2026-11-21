"""gRPC channel/server helpers and proto ↔ Python conversions."""

from __future__ import annotations

import grpc

from ferry.v1 import ferry_pb2 as pb

MAX_MESSAGE_BYTES = 4 * 1024 * 1024

_KEEPALIVE = [
    ("grpc.keepalive_time_ms", 20_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
]
CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
    *_KEEPALIVE,
]
SERVER_OPTIONS = [
    *CHANNEL_OPTIONS,
    ("grpc.http2.min_ping_interval_without_data_ms", 10_000),
]


def channel(addr: str) -> grpc.aio.Channel:
    # TODO(v1+): TLS credentials. DESIGN §1 scopes v1 security to transport TLS, configured by the operator.
    return grpc.aio.insecure_channel(addr, options=CHANNEL_OPTIONS)


def server() -> grpc.aio.Server:
    return grpc.aio.server(options=SERVER_OPTIONS)


def state_name(state: int) -> str:
    """``NODE_STATE_TRAINING`` → ``TRAINING``."""
    return pb.NodeState.Name(state).removeprefix("NODE_STATE_")


def node_type_name(t: int) -> str:
    return {pb.NODE_TYPE_WORKER: "worker", pb.NODE_TYPE_AGGREGATOR: "aggregator"}.get(t, "")
