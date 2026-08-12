# Architecture

## Invariants

KVBridge treats a cache transfer as a typed, revision-pinned state conversion. Five invariants are non-negotiable:

1. Token positions align because tokenizers are fingerprint-equal.
2. Cache tensors have canonical `[batch, kv_heads, tokens, head_dim]` layout.
3. Keys are mapped in content space and rotated using factors produced by the receiving model.
4. An artifact is valid for one ordered source/target revision pair only.
5. Any failed validation is observable and falls back to full target prefill.

## Components

### Cache plane

`KVCache` owns immutable layer tuples and optional `RotaryFactors`. It validates all K/V layer shapes, floating-point types, and factor geometry at construction. RoPE inversion is an orthogonal inverse; both half-split and interleaved layouts are supported.

### Calibration plane

`CalibrationPair` aligns source and target caches from identical token sequences. SafeTensors shards make the dataset out-of-core and avoid executable pickle payloads. A shard factory can be reopened for every target-layer block.

### Fitting plane

Stage 1 fits single-source, same-head probes for every source/target layer pair and averages K/V R² across heads. Stage 2 concatenates every KV head from the selected source layers and solves one multi-output ridge system per target layer for K and V. Reshaping that solution yields independent target-head projections while sharing the expensive covariance computation.

`RidgeAccumulator` stores count, running means, and centered `XᵀX`, `XᵀY`, and `YᵀY` moments. Batchwise Chan updates avoid subtracting large raw moments, while a parallel Chan correction keeps distributed all-reduce fixed-size and mergeable.

### Artifact plane

Weights and biases are stored in SafeTensors. The JSON manifest stores schema version, timestamps, paper provenance, fit configuration, source selection, diagnostics, model signatures, and a SHA-256 digest of the tensor file. Writes use temporary files followed by atomic replacement. Loading verifies the digest before tensor access.

### Serving plane

The mapper strips source RoPE, gathers top-k layer features, projects K and V, reshapes the output into target cache layout, and applies target RoPE. `GuardedTransferEngine` wraps that operation with numerical, magnitude, latency, and application-defined quality gates. Its fallback boundary intentionally catches mapper/backend exceptions so a handoff failure cannot crash a serving request.

## Memory model

For source feature width `d_s = k H_s D_s` and target output width `d_t = H_t D_t`, one target layer retains two sufficient-statistic sets (K and V):

`2 × (d_s² + d_s d_t + d_s + d_t + 1) × accumulator_bytes`.

Peak memory grows with the configured target-layer block, not total target depth. Total I/O passes are approximately `ceil(L_t / selection_block) + ceil(L_t / fit_block)`. This is the central modest-hardware tradeoff.

## Trust boundaries

- Model/tokenizer revisions are external inputs and must be pinned.
- Calibration text may be sensitive; shards should inherit dataset access controls.
- SafeTensors removes pickle code execution but does not establish provenance; artifact distribution should add repository-level signatures/attestations.
- The Hugging Face adapter is a compatibility boundary because cache APIs change across Transformers versions. Pin and test that dependency in each deployment image.
