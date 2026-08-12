# Changelog

## 0.2.0 - 2026-08-12

- Added GPU-resident, memory-bounded ridge accumulation with deterministic token striding.
- Added attention-output cosine evaluation and fail-closed short-suffix logit-KL probes.
- Added BF16 artifact storage with manifest dtype/byte validation and measured precision evidence.
- Added one-time mapper device residency to avoid per-request artifact transfers.
- Added pinned Qwen3 0.6B to 1.7B and 1.7B to 4B T2 configurations.
- Added dry-run-safe real-model capture, fit, and evaluation jobs with immutable provenance.
- Modernized GitHub Actions to Node 24-based checkout/setup actions.

## 0.1.0 - 2026-08-08

- Paper-faithful per-head K/V ridge mapper with top-k source layers.
- Exact captured-factor RoPE removal and reapplication.
- Memory-bounded, mergeable, distributed-ready sufficient statistics.
- SafeTensors calibration shards and tamper-evident mapper artifacts.
- Model/tokenizer compatibility gates and guarded fallback runtime.
- Experimental Hugging Face `DynamicCache` handoff path.
- Resource planner, CPU scale suite, CI, and research protocol.
