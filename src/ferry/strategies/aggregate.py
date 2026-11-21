"""``Aggregator`` strategies: combine a batch of deltas into one outer gradient (DESIGN §6, §7.3).

Every strategy returns the **outer gradient** ``−combined_delta`` (DiLoCo convention).
"""

from __future__ import annotations

from typing import Protocol

import torch

from ferry.strategies.spec import build
from ferry.types import Delta, OuterGradient, StateDict


class Aggregator(Protocol):
    """Combines a batch of deltas into one outer gradient."""

    def aggregate(self, deltas: list[Delta], current_version: int) -> OuterGradient: ...


def _keys(deltas: list[Delta]) -> list[str]:
    if not deltas:
        raise ValueError("cannot aggregate an empty batch")
    keys = list(deltas[0].tensors)
    for d in deltas[1:]:
        if set(d.tensors) != set(keys):
            raise ValueError(f"delta {d.meta.delta_id} has mismatched tensor keys")
    return keys


def _step_scales(deltas: list[Delta], normalize: bool) -> list[float]:
    """Rescale each delta to the batch's mean step count so fast workers don't act as a larger LR."""
    if not normalize:
        return [1.0] * len(deltas)
    steps = [max(1, d.meta.inner_steps) for d in deltas]
    ref = sum(steps) / len(steps)
    return [ref / s for s in steps]


def _weighted(deltas: list[Delta], weights: list[float], normalize_steps: bool) -> OuterGradient:
    keys = _keys(deltas)
    total = sum(weights)
    if total <= 0:
        raise ValueError("aggregation weights sum to zero")
    scales = _step_scales(deltas, normalize_steps)
    out: StateDict = {}
    for k in keys:
        acc = None
        for d, w, s in zip(deltas, weights, scales, strict=True):
            t = d.tensors[k].to(torch.float32) * (w * s / total)
            acc = t if acc is None else acc + t
        out[k] = -acc.to(deltas[0].tensors[k].dtype)
    return out


class Mean:
    """Unweighted mean, no step normalization."""

    def aggregate(self, deltas: list[Delta], current_version: int) -> OuterGradient:
        return _weighted(deltas, [1.0] * len(deltas), normalize_steps=False)


class SampleWeightedMean:
    """Weight by ``num_samples``; normalize by ``inner_steps``."""

    def __init__(self, normalize_steps: bool = True) -> None:
        self.normalize_steps = normalize_steps

    def aggregate(self, deltas: list[Delta], current_version: int) -> OuterGradient:
        w = [float(max(1, d.meta.num_samples)) for d in deltas]
        return _weighted(deltas, w, self.normalize_steps)


class StalenessWeightedMean:
    """Default. Weight = ``num_samples × 1/(1 + staleness)``; normalize by ``inner_steps``."""

    def __init__(self, normalize_steps: bool = True, sample_weighted: bool = True) -> None:
        self.normalize_steps = normalize_steps
        self.sample_weighted = sample_weighted

    def aggregate(self, deltas: list[Delta], current_version: int) -> OuterGradient:
        w = []
        for d in deltas:
            staleness = max(0, current_version - d.meta.base_global_version)
            base = float(max(1, d.meta.num_samples)) if self.sample_weighted else 1.0
            w.append(base / (1.0 + staleness))
        return _weighted(deltas, w, self.normalize_steps)


class TrimmedMean:
    """Coordinate-wise trimmed mean (robust to outlier workers). Step-normalized."""

    def __init__(self, trim_ratio: float = 0.1, normalize_steps: bool = True) -> None:
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        self.trim_ratio = trim_ratio
        self.normalize_steps = normalize_steps

    def aggregate(self, deltas: list[Delta], current_version: int) -> OuterGradient:
        keys = _keys(deltas)
        scales = _step_scales(deltas, self.normalize_steps)
        n = len(deltas)
        k = int(n * self.trim_ratio)
        out: StateDict = {}
        for key in keys:
            stacked = torch.stack([d.tensors[key].to(torch.float32) * s for d, s in zip(deltas, scales, strict=True)])
            if k > 0:
                stacked, _ = torch.sort(stacked, dim=0)
                stacked = stacked[k : n - k]
            out[key] = -stacked.mean(dim=0).to(deltas[0].tensors[key].dtype)
        return out


REGISTRY = {
    "mean": Mean,
    "sample_weighted_mean": SampleWeightedMean,
    "staleness_weighted_mean": StalenessWeightedMean,
    "trimmed_mean": TrimmedMean,
}


def build_aggregator(spec: str) -> Aggregator:
    return build(spec, REGISTRY)
