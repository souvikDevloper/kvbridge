# Multi-lab experiment protocol

This protocol separates results executable on modest hardware from experiments requiring a partner lab. Every result row must record model commit hashes, tokenizer fingerprints, code commit, container digest, CUDA/driver versions, GPU topology, seed, prompt construction, and raw result artifact.

## Research questions

1. Does memory-bounded block fitting reproduce the monolithic closed-form solution within numerical tolerance?
2. How does accumulator precision (FP32 vs FP64) affect layer selection, cache reconstruction, attention-output cosine, and downstream retention?
3. How does calibration size and domain affect long-context transfer beyond 1,024 tokens?
4. Can mapper weights be quantized or low-rank-compressed without moving error into attention-sensitive subspaces?
5. Which online probes reliably trigger fallback before downstream quality collapses?

## T0: deterministic software validation

- Run `ruff check`, all unit tests, package build, demo, planner, and artifact corruption tests.
- Require exact synthetic source-layer recovery and holdout R² > 0.999.
- Require visible fallback for rejected quality probes, NaN/Inf, magnitude violation, and latency violation.
- Repeat on Python 3.10 and 3.12, Linux and Windows where available.

## T1: deterministic synthetic scale validation

- Run the planted-affine model-family sweep at several depths, head widths, and token counts.
- Require exact source-layer recovery and holdout cache R² > 0.999.
- Measure warm/cold map latency and FP32/BF16 artifact size separately.
- Label all results synthetic; do not infer real-model downstream retention.

## T2: small-model integration

Choose a same-family pair that fits the available accelerator and satisfies the shared-tokenizer/matched-KV/full-attention gates. Capture 50-200 general-domain calibration sequences. Verify:

- source/target cache geometry and RoPE round-trip;
- target logits from its own cache versus a reconstructed identity/control cache;
- mapped-cache perplexity on held-out text;
- attention-output cosine by layer/head;
- greedy and sampled continuation smoke tests;
- Transformers cache compatibility for pinned dependency versions.

No small-model result should be extrapolated numerically to Qwen3 14B→32B.

Use the pinned 0.6B to 1.7B smoke configuration first, followed by 1.7B to 4B where memory permits. Record attention-output cosine, short-suffix logit KL, next-token agreement, target-prefill/transfer latency, peak VRAM, and every per-sequence value.

## T3: Qwen3 14B→32B reproduction

Use `configs/qwen3_14b_to_32b.paper.json`: 500 FineWeb-Edu sequences, 1,024 tokens, stride 4, ridge λ=0.01, top-k=8, BF16 forward, FP32 covariance. Match the source/target model commits in the config or record an intentional revision update.

Evaluate target standalone and transfer on ARC-Challenge, HellaSwag, WinoGrande, MMLU 5-shot, GSM8K 8-shot chain-of-thought, prefix-conditioned WikiText-2 perplexity, and CoQA multi-turn handoff. Report raw accuracy, simple retention, and floor-normalized retention. Never choose k on a benchmark later presented as held out.

Latency uses 50 warmups and 30 timed trials for each context length from 64 to 32,768 tokens. Record cache movement and mapper execution together; target re-prefill excludes the LM head to match the source paper.

## T4: new investigations

Run a factorial ablation over:

- accumulator precision: FP32, FP64;
- calibration sequences: 50, 100, 200, 500;
- domains: FineWeb-Edu, Wikipedia, code, one partner-specific domain;
- mapper storage: FP32, BF16, per-channel INT8, rank-truncated SVD;
- context lengths: 1K, 4K, 8K, 16K, 32K;
- fit block sizes: 1, 2, 4, 8 target layers.

Primary quality endpoints are downstream retention and attention-output cosine. R² is secondary. Primary systems endpoints are peak host memory, peak VRAM, artifact size, calibration wall time, mapper latency, prefill latency, and fallback rate.

## Statistical reporting

- Publish per-task sample counts and confidence intervals for accuracy deltas.
- Use paired bootstrap intervals where target and transfer score the same examples.
- Publish the bootstrap confidence level, resample count, and seed; KVBridge records all three.
- Report median, p95, and p99 latency with trial counts and warmup policy.
- Keep raw JSON/CSV outputs and immutable manifests alongside tables.
- Reuse calibration shards across fit ablations only when their calibration-contract hash matches; record capture and fit code revisions separately.
- Distinguish exploratory ablations from preregistered confirmatory runs.

## Ship gate

A model pair can enter canary traffic only if it passes task-specific retention floors, attention-output diagnostics, artifact integrity, revision gates, long-context checks, and rollback drills. Production acceptance is pair-specific; matched KV alone is not sufficient.
