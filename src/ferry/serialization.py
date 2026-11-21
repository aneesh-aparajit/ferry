"""Tensor (de)serialization. Safetensors only: pickle is never used on any network or blob path (DESIGN §4.2)."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from ferry.types import StateDict

CHUNK_SIZE = 1 << 20  # 1 MB max chunk payload
CHECKPOINT_FORMAT = "ferry-ckpt-v1"

_PARAMS = "params/"
_OUTER = "outer/"


class IntegrityError(ValueError):
    """A reassembled payload did not match its declared size or checksum."""


def _prepare(tensors: StateDict) -> StateDict:
    return {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}


def dumps(tensors: StateDict, metadata: dict[str, str] | None = None) -> bytes:
    """Serialize a flat ``{name: tensor}`` dict to safetensors bytes."""
    return st_save(_prepare(tensors), metadata=metadata)


def loads(data: bytes) -> StateDict:
    """Parse safetensors bytes into a flat ``{name: tensor}`` dict."""
    return st_load(data)


def read_metadata(data: bytes) -> dict[str, str]:
    """Return the ``__metadata__`` section of a safetensors blob without loading tensors."""
    if len(data) < 8:
        raise IntegrityError("payload too small to be safetensors")
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + n])
    return dict(header.get("__metadata__") or {})


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk(data: bytes, size: int = CHUNK_SIZE) -> Iterator[bytes]:
    """Split ``data`` into slices of at most ``size`` bytes (always yields at least one slice)."""
    if not data:
        yield b""
        return
    for i in range(0, len(data), size):
        yield data[i : i + size]


@dataclass
class Reassembler:
    """Reassembles ``seq``-ordered chunks and verifies ``total_bytes`` and ``sha256``."""

    total_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        self._parts: list[bytes] = []
        self._next_seq = 0
        self._size = 0

    def add(self, seq: int, data: bytes) -> None:
        if seq != self._next_seq:
            raise IntegrityError(f"out-of-order chunk: expected seq {self._next_seq}, got {seq}")
        self._size += len(data)
        if self.total_bytes and self._size > self.total_bytes:
            raise IntegrityError("payload exceeds declared total_bytes")
        self._parts.append(data)
        self._next_seq += 1

    def finish(self) -> bytes:
        data = b"".join(self._parts)
        if len(data) != self.total_bytes:
            raise IntegrityError(f"size mismatch: declared {self.total_bytes}, got {len(data)}")
        if self.sha256 and sha256_hex(data) != self.sha256:
            raise IntegrityError("sha256 mismatch")
        return data


def reassemble(chunks: Iterable[tuple[int, bytes]], total_bytes: int, sha256: str) -> bytes:
    r = Reassembler(total_bytes, sha256)
    for seq, data in chunks:
        r.add(seq, data)
    return r.finish()


# ───────────────────────── Checkpoints ─────────────────────────


@dataclass
class Checkpoint:
    params: StateDict
    outer_state: StateDict
    global_version: int
    policy_state: dict


def encode_checkpoint(ckpt: Checkpoint) -> bytes:
    """Checkpoint blob: ``{params, outer_optimizer_state, global_version, policy_state}`` (DESIGN §7.2)."""
    tensors: StateDict = {}
    for k, v in ckpt.params.items():
        tensors[_PARAMS + k] = v
    for k, v in ckpt.outer_state.items():
        tensors[_OUTER + k] = v
    meta = {
        "format": CHECKPOINT_FORMAT,
        "global_version": str(ckpt.global_version),
        "policy_state": json.dumps(ckpt.policy_state),
    }
    return dumps(tensors, meta)


def decode_checkpoint(data: bytes) -> Checkpoint:
    meta = read_metadata(data)
    if meta.get("format") != CHECKPOINT_FORMAT:
        raise IntegrityError(f"not a Ferry checkpoint (format={meta.get('format')!r})")
    tensors = loads(data)
    params = {k[len(_PARAMS) :]: v for k, v in tensors.items() if k.startswith(_PARAMS)}
    outer = {k[len(_OUTER) :]: v for k, v in tensors.items() if k.startswith(_OUTER)}
    return Checkpoint(
        params=params,
        outer_state=outer,
        global_version=int(meta.get("global_version", "0")),
        policy_state=json.loads(meta.get("policy_state", "{}")),
    )


# ───────────────────────── torch.optim state ─────────────────────────


def encode_optimizer_state(state: dict, extra: dict | None = None) -> bytes:
    """Serialize a ``torch.optim.Optimizer.state_dict()`` without pickle.

    Tensor entries become safetensors tensors; scalars and ``param_groups`` go into JSON metadata.
    """
    tensors: StateDict = {}
    scalars: dict[str, dict] = {}
    for pid, pstate in state.get("state", {}).items():
        for key, val in pstate.items():
            if isinstance(val, torch.Tensor):
                tensors[f"state/{pid}/{key}"] = val
            else:
                scalars.setdefault(str(pid), {})[key] = val
    meta = {
        "format": "ferry-optim-v1",
        "param_groups": json.dumps(state.get("param_groups", [])),
        "scalars": json.dumps(scalars),
        "extra": json.dumps(extra or {}),
    }
    return dumps(tensors, meta)


def decode_optimizer_state(data: bytes) -> tuple[dict, dict]:
    """Inverse of :func:`encode_optimizer_state`. Returns ``(optimizer_state_dict, extra)``."""
    meta = read_metadata(data)
    tensors = loads(data)
    state: dict[int, dict] = {}
    for name, t in tensors.items():
        _, pid, key = name.split("/", 2)
        state.setdefault(int(pid), {})[key] = t
    for pid, vals in json.loads(meta.get("scalars", "{}")).items():
        state.setdefault(int(pid), {}).update(vals)
    return {"state": state, "param_groups": json.loads(meta.get("param_groups", "[]"))}, json.loads(
        meta.get("extra", "{}")
    )
