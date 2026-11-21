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

## Status

Early design, v1 in progress. Targets small models (MNIST, CIFAR) first, with a single-node coordinator. See
[adr/README.md](adr/README.md) for the overview and [adr/DESIGN.md](adr/DESIGN.md) for the full spec.
