import pytest
import torch

from ferry.blobstore import LocalFSBlobStore, open_blob_store, read_blob
from ferry.serialization import (
    Checkpoint,
    IntegrityError,
    chunk,
    decode_checkpoint,
    decode_optimizer_state,
    dumps,
    encode_checkpoint,
    encode_optimizer_state,
    reassemble,
    sha256_hex,
)
from ferry.strategies import (
    BufferK,
    Mean,
    SampleWeightedMean,
    SGDOuter,
    StalenessWeightedMean,
    TrimmedMean,
    build_outer_optimizer,
    build_policy,
    parse_spec,
)
from ferry.strategies.outer import AdamOuter
from ferry.types import Delta, DeltaMeta


def _delta(value, steps=10, samples=100, base=0, did="d"):
    return Delta(DeltaMeta(did, "w", "c", base, steps, samples), {"w": torch.full((3,), float(value))})


# ───────── serialization ─────────


def test_chunk_roundtrip_and_integrity():
    data = dumps({"a": torch.randn(600, 600)})  # > 1 MB → several chunks
    parts = list(chunk(data))
    assert len(parts) > 1 and all(len(p) <= 1 << 20 for p in parts)
    assert reassemble(enumerate(parts), len(data), sha256_hex(data)) == data
    with pytest.raises(IntegrityError):
        reassemble(enumerate(parts), len(data), "0" * 64)
    with pytest.raises(IntegrityError):
        reassemble([(1, parts[1])], len(data), "")


def test_checkpoint_roundtrip():
    ck = Checkpoint({"w": torch.ones(2)}, {"momentum/w": torch.zeros(2)}, 7, {"k": 4})
    out = decode_checkpoint(encode_checkpoint(ck))
    assert out.global_version == 7 and out.policy_state == {"k": 4}
    assert torch.equal(out.params["w"], ck.params["w"]) and "momentum/w" in out.outer_state
    with pytest.raises(IntegrityError):
        decode_checkpoint(dumps({"x": torch.ones(1)}))


def test_optimizer_state_roundtrip_without_pickle():
    m = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=0.1)
    m(torch.randn(3, 4)).sum().backward()
    opt.step()
    blob = encode_optimizer_state(opt.state_dict(), {"cursor": 5})
    m2 = torch.nn.Linear(4, 2)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=0.5)
    state, extra = decode_optimizer_state(blob)
    opt2.load_state_dict(state)
    assert extra == {"cursor": 5}
    assert opt2.param_groups[0]["lr"] == 0.1
    a, b = opt.state_dict()["state"][0], opt2.state_dict()["state"][0]
    assert torch.equal(a["exp_avg"], b["exp_avg"]) and float(a["step"]) == float(b["step"])


def test_localfs_blobstore(tmp_path):
    s = open_blob_store(f"file://{tmp_path}")
    assert isinstance(s, LocalFSBlobStore)
    uri = s.put("checkpoints/a.bin", b"hello")
    assert read_blob(s, uri) == b"hello"
    assert not list(tmp_path.glob("checkpoints/.tmp-*"))
    with pytest.raises(ValueError):
        s.put("../escape", b"x")
    s.delete(uri)
    assert not (tmp_path / "checkpoints/a.bin").exists()


# ───────── strategies ─────────


def test_parse_spec():
    assert parse_spec("nesterov(lr=0.7, momentum=0.9)") == ("nesterov", {"lr": 0.7, "momentum": 0.9})
    assert parse_spec("mean") == ("mean", {})
    with pytest.raises(ValueError):
        parse_spec("__import__('os').system('x')")


def test_outer_grad_is_negative_delta():
    g = Mean().aggregate([_delta(1), _delta(3)], 0)
    assert torch.allclose(g["w"], torch.full((3,), -2.0))


def test_step_normalization_and_sample_weighting():
    # Same per-step movement (0.1/step), different H: normalized deltas are equal.
    g = SampleWeightedMean().aggregate([_delta(1.0, steps=10), _delta(3.0, steps=30)], 0)
    assert torch.allclose(g["w"], torch.full((3,), -2.0))
    g = SampleWeightedMean(normalize_steps=False).aggregate([_delta(0, samples=300), _delta(4, samples=100)], 0)
    assert torch.allclose(g["w"], torch.full((3,), -1.0))


def test_staleness_weighting():
    g = StalenessWeightedMean().aggregate([_delta(0, base=5), _delta(3, base=4)], current_version=5)
    # weights 1 and 1/2 → (0*1 + 3*0.5)/1.5 = 1
    assert torch.allclose(g["w"], torch.full((3,), -1.0))


def test_trimmed_mean_drops_outliers():
    ds = [_delta(v) for v in (1, 1, 1, 1, 1000)]
    g = TrimmedMean(trim_ratio=0.2).aggregate(ds, 0)
    assert torch.allclose(g["w"], torch.full((3,), -1.0))


def test_fedavg_applies_mean_delta():
    opt = build_outer_optimizer("sgd(lr=1.0)")
    p = {"w": torch.zeros(3), "n": torch.tensor(3)}
    out = opt.step(p, {"w": torch.full((3,), -2.0), "n": torch.tensor(1)})
    assert torch.allclose(out["w"], torch.full((3,), 2.0)) and out["n"].item() == 3


def test_nesterov_matches_torch_sgd():
    p0 = torch.randn(5)
    grads = [torch.randn(5) for _ in range(3)]
    ours = build_outer_optimizer("nesterov(lr=0.7, momentum=0.9)")
    params = {"w": p0.clone()}
    ref = torch.nn.Parameter(p0.clone())
    topt = torch.optim.SGD([ref], lr=0.7, momentum=0.9, nesterov=True)
    for g in grads:
        params = ours.step(params, {"w": g})
        ref.grad = g.clone()
        topt.step()
    assert torch.allclose(params["w"], ref.detach(), atol=1e-6)
    # state survives a save/restore
    fresh = SGDOuter(lr=0.7, momentum=0.9, nesterov=True)
    fresh.load_state_dict(ours.state_dict())
    assert torch.equal(fresh.state_dict()["momentum/w"], ours.state_dict()["momentum/w"])


def test_adam_state_roundtrip():
    a = AdamOuter(lr=0.1)
    a.step({"w": torch.zeros(2)}, {"w": torch.ones(2)})
    b = AdamOuter(lr=0.1)
    b.load_state_dict(a.state_dict())
    assert b._t == 1 and torch.equal(b._m["w"], a._m["w"])


def test_buffer_k_policy():
    pol = build_policy("buffer_k(k=2, max_wait_s=5, max_staleness=3)")
    assert isinstance(pol, BufferK)
    meta = lambda base: DeltaMeta("d", "w", "c", base, 10, 10)  # noqa: E731
    assert pol.accept(meta(7), 10) == (True, "")
    assert pol.accept(meta(6), 10) == (False, "too_stale")
    assert pol.accept(meta(11), 10)[0] is False
    d = _delta(1)
    d.received_at = 100.0
    assert not pol.should_apply([], 100.0)
    assert not pol.should_apply([d], 101.0)
    assert pol.should_apply([d], 105.0)
    assert pol.should_apply([d, d], 100.0)
