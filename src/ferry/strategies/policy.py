"""``AggregationPolicy``: when buffered deltas are applied and which are accepted (DESIGN §6, §7.3)."""

from __future__ import annotations

from typing import Protocol

from ferry.strategies.spec import build
from ferry.types import Delta, DeltaMeta


class AggregationPolicy(Protocol):
    def accept(self, header: DeltaMeta, current_version: int) -> tuple[bool, str]: ...

    def should_apply(self, buffer: list[Delta], now: float) -> bool: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


class BufferK:
    """Apply once ``k`` deltas are buffered, or ``max_wait_s`` after the oldest buffered delta arrived.

    Rejects deltas whose staleness exceeds ``max_staleness``. ``k=1`` is supported but recommended only with a
    momentum-free outer optimizer (FedAvg).
    """

    def __init__(self, k: int = 4, max_wait_s: float = 30.0, max_staleness: int = 8) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.max_wait_s = max_wait_s
        self.max_staleness = max_staleness

    def accept(self, header: DeltaMeta, current_version: int) -> tuple[bool, str]:
        if header.inner_steps <= 0:
            return False, "invalid_inner_steps"
        if header.base_global_version > current_version:
            return False, "unknown_base_version"
        staleness = current_version - header.base_global_version
        if staleness > self.max_staleness:
            return False, "too_stale"
        return True, ""

    def should_apply(self, buffer: list[Delta], now: float) -> bool:
        if not buffer:
            return False
        if len(buffer) >= self.k:
            return True
        oldest = min(d.received_at for d in buffer)
        return now - oldest >= self.max_wait_s

    def state_dict(self) -> dict:
        return {"k": self.k, "max_wait_s": self.max_wait_s, "max_staleness": self.max_staleness}

    def load_state_dict(self, state: dict) -> None:
        # Policy config comes from the aggregator's own config; nothing stateful to restore in BufferK.
        return None


REGISTRY = {"buffer_k": BufferK}


def build_policy(spec: str) -> AggregationPolicy:
    return build(spec, REGISTRY)
