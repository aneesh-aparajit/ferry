"""Core value types shared by the aggregator, worker and strategies."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

StateDict = dict[str, torch.Tensor]
OuterGradient = dict[str, torch.Tensor]


@dataclass(frozen=True)
class DeltaMeta:
    """Plain-Python mirror of ``DeltaHeader`` (DESIGN §4)."""

    delta_id: str
    worker_id: str
    base_checkpoint_id: str
    base_global_version: int
    inner_steps: int
    num_samples: int
    local_loss: float = 0.0
    total_bytes: int = 0
    sha256: str = ""

    @classmethod
    def from_proto(cls, h) -> DeltaMeta:
        return cls(
            delta_id=h.delta_id,
            worker_id=h.worker_id,
            base_checkpoint_id=h.base_checkpoint_id,
            base_global_version=h.base_global_version,
            inner_steps=h.inner_steps,
            num_samples=h.num_samples,
            local_loss=h.local_loss,
            total_bytes=h.total_bytes,
            sha256=h.sha256,
        )


@dataclass
class Delta:
    """A worker delta: ``θ_local − θ_base`` plus its header metadata."""

    meta: DeltaMeta
    tensors: StateDict
    received_at: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def staleness_base(self) -> int:
        return self.meta.base_global_version
