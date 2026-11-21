"""``OuterOptimizer`` implementations (DESIGN §6).

- ``sgd(lr=1.0)``                              == FedAvg
- ``nesterov(lr=0.7, momentum=0.9)``           == DiLoCo
- ``adam(lr=..., betas=(0.9, 0.99))``          == FedAdam
"""

from __future__ import annotations

from typing import Protocol

import torch

from ferry.strategies.spec import build
from ferry.types import OuterGradient, StateDict


class OuterOptimizer(Protocol):
    """Applies an outer gradient to the global parameters. Stateful; state is checkpointed."""

    def step(self, params: StateDict, outer_grad: OuterGradient) -> StateDict: ...

    def state_dict(self) -> StateDict: ...

    def load_state_dict(self, state: StateDict) -> None: ...


def _floatlike(t: torch.Tensor) -> bool:
    return t.is_floating_point() or t.is_complex()


class SGDOuter:
    """SGD with optional (Nesterov) momentum, matching ``torch.optim.SGD`` update rules."""

    def __init__(self, lr: float = 1.0, momentum: float = 0.0, nesterov: bool = False) -> None:
        if nesterov and momentum <= 0:
            raise ValueError("nesterov requires momentum > 0")
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self._buf: StateDict = {}

    def step(self, params: StateDict, outer_grad: OuterGradient) -> StateDict:
        new: StateDict = {}
        for name, p in params.items():
            g = outer_grad.get(name)
            if g is None or not _floatlike(p):
                new[name] = p  # e.g. integer buffers (BatchNorm num_batches_tracked)
                continue
            g = g.to(p.dtype)
            if self.momentum:
                buf = self._buf.get(name)
                buf = g.clone() if buf is None else buf.mul(self.momentum).add(g)
                self._buf[name] = buf
                g = g.add(buf, alpha=self.momentum) if self.nesterov else buf
            new[name] = p - self.lr * g
        return new

    def state_dict(self) -> StateDict:
        return {f"momentum/{k}": v for k, v in self._buf.items()}

    def load_state_dict(self, state: StateDict) -> None:
        self._buf = {k[len("momentum/") :]: v for k, v in state.items() if k.startswith("momentum/")}


class AdamOuter:
    """Adam outer optimizer (FedAdam, Reddi et al. 2021)."""

    def __init__(self, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.99), eps: float = 1e-8) -> None:
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self._m: StateDict = {}
        self._v: StateDict = {}
        self._t = 0

    def step(self, params: StateDict, outer_grad: OuterGradient) -> StateDict:
        self._t += 1
        new: StateDict = {}
        bc1 = 1 - self.b1**self._t
        bc2 = 1 - self.b2**self._t
        for name, p in params.items():
            g = outer_grad.get(name)
            if g is None or not _floatlike(p):
                new[name] = p
                continue
            g = g.to(p.dtype)
            m = self._m.get(name, torch.zeros_like(p)).mul(self.b1).add(g, alpha=1 - self.b1)
            v = self._v.get(name, torch.zeros_like(p)).mul(self.b2).addcmul(g, g, value=1 - self.b2)
            self._m[name], self._v[name] = m, v
            new[name] = p - self.lr * (m / bc1) / ((v / bc2).sqrt() + self.eps)
        return new

    def state_dict(self) -> StateDict:
        out: StateDict = {"step": torch.tensor(self._t, dtype=torch.int64)}
        out.update({f"m/{k}": v for k, v in self._m.items()})
        out.update({f"v/{k}": v for k, v in self._v.items()})
        return out

    def load_state_dict(self, state: StateDict) -> None:
        self._t = int(state["step"].item()) if "step" in state else 0
        self._m = {k[2:]: v for k, v in state.items() if k.startswith("m/")}
        self._v = {k[2:]: v for k, v in state.items() if k.startswith("v/")}


REGISTRY = {
    "sgd": SGDOuter,
    "fedavg": lambda: SGDOuter(lr=1.0),
    "nesterov": lambda lr=0.7, momentum=0.9: SGDOuter(lr=lr, momentum=momentum, nesterov=True),
    "adam": lambda lr=1e-3, betas=(0.9, 0.99), eps=1e-8: AdamOuter(lr=lr, betas=tuple(betas), eps=eps),
}


def build_outer_optimizer(spec: str) -> OuterOptimizer:
    return build(spec, REGISTRY)
