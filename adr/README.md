# Ferry

Ferry is a **control plane for remote, low-communication training workloads**.

It coordinates a fleet of training workers (GPU nodes) and an aggregator (CPU node) that train a shared model while
communicating rarely. Workers train locally for many steps, ship a compact *delta* of their weights, and pick up the
latest global model, much like a ferry moving cargo between islands on a schedule instead of building a bridge.

Ferry is inspired
by [DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105)
and by federated optimization (FedAvg / FedOpt), and borrows its operational model from Kubernetes: one control plane
that every component dials into, a metadata store as the source of truth, and nodes that report their own status.

> **Status:** early design / v1 in progress. Targets small models (MNIST, CIFAR) first. See [DESIGN.md](./DESIGN.md)
> for the full spec.

## What Ferry does

- **Membership and health.** Workers and the aggregator register with the coordinator, hold a long-lived session, and
  send heartbeats. The coordinator tracks each node's state and lease.
- **Checkpoint registry.** The aggregator persists global weights (and outer optimizer state) to blob storage. The
  coordinator records only the checkpoint **ID**. It never touches tensors.
- **Recovery.** A node that disconnects and re-registers is handed the latest checkpoint ID and its own last persisted
  state, so it can resume instead of starting over.
- **Asynchronous aggregation.** Workers can have different hardware and different local step counts (`H`). Nobody waits
  on the slowest worker: each worker pushes its delta when it is done, gets the latest global checkpoint, and keeps
  training.
- **Pluggable strategy.** Aggregation (how deltas are combined) and the outer optimizer (how the combined delta is
  applied) are interfaces. FedAvg, DiLoCo (Nesterov outer step) and FedAdam-style variants are configurations of the
  same two pieces.

## Components

| Component       | Runs on         | Role                                                                                          |
|-----------------|-----------------|-----------------------------------------------------------------------------------------------|
| **Coordinator** | 1 node          | Control plane. Membership, health, leases, checkpoint registry. Backed by Postgres.           |
| **Aggregator**  | CPU node        | Receives worker deltas, applies the outer optimizer step, writes checkpoints to blob storage. |
| **Worker**      | GPU/CPU node(s) | Runs `H` inner optimizer steps on local data, pushes a delta, pulls the latest checkpoint.    |
| **Blob store**  | FS / S3 / etc.  | Holds checkpoint and delta payloads (safetensors). Pluggable.                                 |
| **Postgres**    | 1 node          | Coordinator metadata: nodes, sessions, checkpoint pointers, worker state.                     |

```
                 ┌──────────────────────────┐
                 │       Coordinator        │◄──── Postgres
                 │ (membership, health,     │
                 │  checkpoint registry)    │
                 └──────▲────────────▲──────┘
          Session (bidi)│            │Session (bidi)
                        │            │
   ┌────────────┐       │            │       ┌────────────┐
   │  Worker 1  │───────┘            └───────│ Aggregator │
   │  Worker 2  │──── PushDelta ────────────►│            │
   │    ...     │                            └─────┬──────┘
   └─────┬──────┘                                  │ write checkpoint
         │            fetch checkpoint             ▼
         └──────────────────────────────────► Blob store
```

## Core ideas

- **One address.** Every component dials the coordinator. Workers never need to be reachable from the outside, so they
  work behind NAT and firewalls. The aggregator's address is discovered from the coordinator.
- **Pointers, not payloads.** The control plane stores checkpoint IDs. Weights flow directly between workers, the
  aggregator and blob storage.
- **Persist, then publish.** A checkpoint ID is registered with the coordinator only after its blob is fully written.
- **Never waste GPU time.** If the aggregator is unavailable, workers keep training from their current base and retry
  the push later, with the larger delta.
- **Config lives with the worker.** No central job spec in v1. Each worker owns its config (`H`, learning rate, batch
  size...) and reports it to the coordinator when it connects.

## Wire format

- **Transport:** gRPC. A bidirectional `Session` stream carries the control plane (small messages). Bulk tensors use
  separate streaming RPCs so large uploads never delay heartbeats.
- **Tensors:** [safetensors](https://github.com/huggingface/safetensors), chunked into roughly 1 MB gRPC messages. No
  pickle (`torch.save`) on the wire.

## Known limitations (v1)

- The coordinator is a single node. If it is down, nodes cannot register or report health.
- Exactly one aggregator is active at a time. Split-brain protection (fencing) is designed for but not enforced yet.
- No central job spec or scheduler.
- Targets small models. Large-model concerns (sharding, streaming fragments, quantized deltas) are future work.

## Further reading

- Douillard et al., *DiLoCo: Distributed Low-Communication Training of Language Models* (2023)
- Liu et al., *Asynchronous Local-SGD Training for Language Modeling* (2024)
- Douillard et al., *Streaming DiLoCo with overlapping communication* (2025)
- Reddi et al., *Adaptive Federated Optimization* (FedOpt, 2021)
- Li et al., *Scaling Distributed Machine Learning with the Parameter Server* (2014)