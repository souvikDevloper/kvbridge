# Research and systems collaboration

KVBridge welcomes narrowly scoped collaborations that add independently verifiable evidence. Production-grade engineering and model-pair quality are separate claims: a contribution may strengthen either one, but every result must state which claim it supports.

## High-value contribution tracks

1. **Single-GPU replication:** repeat the pinned Qwen3 0.6B→1.7B run on a second T4, P100, L4, or A10 and publish the complete evidence chain.
2. **Headline pair:** run Qwen3 1.7B→4B on a verified 24 GB or larger accelerator, keeping its calibration, mapper, and results separate from the smoke pair.
3. **Paper-scale reproduction:** provide an eight-H100-class environment for the pinned Qwen3 14B→32B configuration and downstream benchmark protocol.
4. **Numerics:** compare FP32 and FP64 centered accumulation, condition numbers, source-layer stability, and Cholesky versus least-squares fallback.
5. **Serving integration:** add a revision-pinned vLLM or another serving-backend adapter with shadow-prefill fallback and failure-injection tests.
6. **Quality research:** study longer contexts, domain shift, calibration scale, artifact compression, and probe false-accept/false-reject behavior.

## Evidence package

Every measured claim should include:

- immutable source/target and dataset revisions;
- tokenizer and model fingerprints;
- executable config and code commit;
- capture, mapper, and result manifests with SHA-256 links;
- Python, package, driver, CUDA, and exact accelerator identity;
- raw per-sequence metrics, sample count, confidence level, bootstrap resamples, and seed;
- peak memory, warmup policy, and unaggregated latency when performance is claimed;
- an explicit pass/reject decision against preregistered pair-specific gates.

Do not extrapolate a small-model result to the 14B→32B pair. Rejected pairs and numerical failures are publishable findings when their provenance and failure mode are preserved.

## How to propose a run

Open a GitHub issue with the model pair, available hardware, intended evidence tier, estimated runtime, and which artifacts you can publish. Do not upload proprietary calibration text or raw logits. Start with a dry-run resource plan, then agree on the immutable config before consuming accelerator time.

Code contributions should follow [CONTRIBUTING.md](CONTRIBUTING.md); coordinated experiments should follow [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).
