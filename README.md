# ferry

A control plane for remote, low-communication training workloads.

Ferry coordinates a fleet of training workers (GPU nodes) and an aggregator (CPU node) that train a shared model while
communicating rarely. Workers run many local optimizer steps, ship a compact *delta* of their weights, and pick up the
latest global model, like a ferry moving cargo between islands on a schedule instead of building a bridge.

Inspired by [DiLoCo](https://arxiv.org/abs/2311.08105) and federated optimization (FedAvg / FedOpt), with an
operational model borrowed from Kubernetes: one control plane every component dials into, a metadata store as the
source of truth, and nodes that report their own status.

## Components

- **Coordinator:** membership, heartbeats, leases and the checkpoint registry. Backed by Postgres. Never touches tensors.
- **Aggregator:** receives worker deltas, applies the outer optimizer step, writes checkpoints to blob storage.
- **Worker:** runs `H` inner steps on local data, pushes a delta, pulls the latest checkpoint.
- **Blob store:** pluggable (filesystem, S3, ...) storage for checkpoint and delta payloads.

## Core ideas

- **One address.** Everything dials the coordinator, so workers can sit behind NAT and firewalls.
- **Pointers, not payloads.** The control plane stores checkpoint IDs; weights flow directly between workers, the
  aggregator and blob storage.
- **Persist, then publish.** A checkpoint ID is registered only after its blob is fully written.
- **Async and heterogeneous.** Workers can differ in hardware and local step count; nobody waits on stragglers.
- **Never waste GPU time.** If the aggregator is down, workers keep training and retry the push later.
- **Recovery.** A node that reconnects gets the latest checkpoint and its own last state, and resumes.

## Wire format

gRPC, with a bidirectional `Session` stream for control messages and separate streaming RPCs for bulk tensors.
Tensors are serialized as [safetensors](https://github.com/huggingface/safetensors) in ~1 MB chunks, never pickle.

## Quickstart

```bash
uv sync
# 1. control plane (use a Postgres DSN in production; memory:// keeps state in-process)
uv run ferry coordinator --postgres-dsn memory://
# 2. aggregator: initializes version 0 from the model factory if no checkpoint exists
uv run ferry aggregator --coordinator localhost:7070 --advertise localhost:7071 --model examples.linreg:make_model
# 3. workers, each with its own H
uv run ferry worker --coordinator localhost:7070 --task examples.linreg:make_task --task-arg seed=1 --h 50
uv run ferry worker --coordinator localhost:7070 --task examples.linreg:make_task --task-arg seed=2 --h 120
```

A worker task is any `module:factory` returning a `ferry.worker.TrainTask(model, train_step, optimizer=...)`.
See [examples/linreg.py](examples/linreg.py).

## Layout

| Path                          | What                                                                    |
|-------------------------------|-------------------------------------------------------------------------|
| `proto/ferry/v1/ferry.proto`  | Wire protocol (regenerate stubs with `scripts/gen_proto.sh`)            |
| `src/ferry/coordinator/`      | Sessions, describe handshake, leases, `CommitCheckpoint`, fan-out       |
| `src/ferry/aggregator/`       | `PushDelta`, buffering, outer step, persist-then-publish commits        |
| `src/ferry/worker/`           | Inner loop, delta push with accumulate-on-failure, state persistence    |
| `src/ferry/strategies/`       | `Aggregator`, `OuterOptimizer`, `AggregationPolicy` implementations     |
| `src/ferry/metadata/`         | `MetadataStore`: Postgres (`schema.sql`) and in-memory                  |
| `src/ferry/blobstore.py`      | `BlobStore`: local FS (atomic rename) and S3 (`ferry[s3]`)              |
| `src/ferry/serialization.py`  | safetensors encode/decode, 1 MB chunking, sha256 checks (no pickle)     |

## Tests

```bash
uv run pytest                                   # unit + in-process cluster tests
FERRY_TEST_POSTGRES_DSN=postgresql://... uv run pytest tests/test_metadata.py   # Postgres store
```

## Status

v1 implemented per [adr/DESIGN.md](adr/DESIGN.md). Targets small models first, with a single-node coordinator. See
[adr/README.md](adr/README.md) for the overview.
