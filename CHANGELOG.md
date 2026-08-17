# Changelog

## Unreleased

- Add an exact vectorized layer-selection accumulator with covariance sharing.
- Add contract-bound, atomic selection-block and target-layer fit checkpoints.
- Add pre-sampled calibration-pair support to avoid double-striding TPU caches.
- Add a sequential-residency PyTorch/XLA SPMD FSDPv2 runner for TPU v5e/v6e.
- Add a sequential TPU held-out evaluator and tamper-evident evidence validator.
- Add pinned 128K and 256K Qwen3 14B→32B TPU configs and a Kaggle/TRC runbook.
- Expose total sampled-host-cache memory in the scale planner.
- Published hash-bound Tesla-T4 Qwen3 0.6B→1.7B T2 evidence, including all raw evaluation rows and deterministic confidence intervals; both preregistered quality gates rejected the pair.
- Added a lightweight publication validator that verifies provenance and recomputes aggregates without pretending the omitted calibration shards or mapper weights were re-verified.
- Added exact symmetric diagonal equilibration and CPU-FP64 recovery for ill-conditioned ridge solves, plus finite-statistic checks and regression tests.

## 0.2.0 - 2026-08-12

- Added GPU-resident, memory-bounded ridge accumulation with deterministic token striding.
- Added attention-output cosine evaluation and fail-closed short-suffix logit-KL probes.
- Added BF16 artifact storage with manifest dtype/byte validation and measured precision evidence.
- Added one-time mapper device residency to avoid per-request artifact transfers.
- Added pinned Qwen3 0.6B to 1.7B and 1.7B to 4B T2 configurations.
- Added dry-run-safe real-model capture, fit, and evaluation jobs with immutable provenance.
- Added atomic evidence commits, per-shard SHA-256 records, and end-to-end evidence validation.
- Added fail-closed stage resume and validated recovery from post-write evaluator teardown failures.
- Added deterministic request-ID shadow sampling plus pre-map batch/token resource gates.
- Added deterministic bootstrap intervals and paired-difference uncertainty helpers.
- Added numerically stable centered ridge statistics and calibration-contract reuse for fit ablations.
- Rejected non-finite tensors, metrics, and non-standard JSON numeric constants fail closed.
- Added recoverable integrity indexing for exact-config legacy calibration captures.
- Preserved legacy manifest backups byte-for-byte and widened NumPy 2 compatibility.
- Pinned the tested Transformers major version and aligned the runtime package version with v0.2.0.
- Modernized GitHub Actions to Node 24-based checkout/setup actions.

## 0.1.0 - 2026-08-08

- Paper-faithful per-head K/V ridge mapper with top-k source layers.
- Exact captured-factor RoPE removal and reapplication.
- Memory-bounded, mergeable, distributed-ready sufficient statistics.
- SafeTensors calibration shards and tamper-evident mapper artifacts.
- Model/tokenizer compatibility gates and guarded fallback runtime.
- Experimental Hugging Face `DynamicCache` handoff path.
- Resource planner, CPU scale suite, CI, and research protocol.
