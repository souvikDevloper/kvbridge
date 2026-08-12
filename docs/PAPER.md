# KVBridge: Memory-Bounded and Failure-Safe Productionization of Closed-Form Cross-Model KV-Cache Transfer

**Souvik**<br>
Independent Researcher<br>
Technical Report v0.2 - 12 August 2026

## Abstract

Cross-model KV-cache transfer can eliminate redundant target-model prefill when a serving system switches between differently sized members of an LLM family. Heo et al. recently showed that this relationship contains substantial linear structure and proposed a closed-form per-head ridge mapper with target-specific source-layer selection and position-free key mapping. Their strongest same-family pairs retained 73-98% of target standalone accuracy while mapping was 2.7-25 times faster than target re-prefill. Turning the method into a deployable component, however, introduces requirements beyond the estimator: bounded host memory, out-of-core calibration, model and tokenizer identity, exact rotary-position handling, artifact provenance, cache API integration, observability, and safe fallback.

We present KVBridge, an independent production-oriented implementation and systems extension. KVBridge computes cancellation-resistant mergeable centered statistics, fits target layers in configurable blocks, captures RoPE factors from the models instead of reconstructing scaling rules, persists data and weights without pickle, and treats rejection as a normal serving outcome. On three CPU-only synthetic model families, it exactly recovered all planted source-layer relationships and achieved holdout cache R² between 0.99999873 and 0.99999895. Median mapping latency ranged from 0.339 ms for a 64-token micro case to 2.844 ms for a 256-token medium case. These results validate structural correctness, not real-model downstream quality. A deterministic resource analysis for Qwen3 14B to 32B reproduces a 1.074-billion-parameter, 4.0005-GiB FP32 mapper while bounding one-layer fit statistics to approximately 0.56 GiB through repeated shard passes. We release a staged protocol for small-model integration and multi-lab Qwen3 reproduction, explicitly reserving paper-scale quality and GPU speed claims for future partnership runs.

**Keywords:** LLM serving, KV cache, prefill, representation alignment, ridge regression, systems reproducibility

## 1. Introduction

Long-running agent sessions accumulate context while production routers change models for quality, cost, specialization, or load. A receiver normally processes the entire transcript again to construct its own attention cache. The cost increases with context length and target-model size. Prefix caches solve only the same-model, same-prefix case.

Heo et al. formulate a representation-conversion alternative: map the source model's keys and values into the target model's cache space, then decode without target prefill [1]. For matched-KV pairs - equal KV-head count and per-head dimension - their per-head ridge maps combine multiple source layers, remove RoPE from keys before regression, and restore the target rotation afterward. The estimator is gradient-free, but its published configurations still produce 1.01-3.36 billion mapper parameters and 4-12 GB artifacts. Calibration and evaluation used an eight-H100 node [1].

This report asks a systems question: what must surround the closed-form estimator so that a small engineering team can implement, inspect, test, and later validate it with a partner lab without weakening production standards or overstating evidence?

Our contributions are:

1. A paper-faithful PyTorch implementation of top-k cross-layer, cross-head, per-target-head ridge mapping for K and V.
2. A memory-bounded fit that retains mergeable sufficient statistics, reopens out-of-core calibration shards, and processes configurable target-layer blocks.
3. An exact RoPE boundary that consumes cosine and sine factors emitted by each model, avoiding incomplete reimplementations of Llama-3, YaRN, or dynamic scaling.
4. A typed artifact and serving contract with revision/tokenizer fingerprints, SafeTensors, integrity verification, atomic writes, numerical gates, short-suffix logit-KL probes, structured events, and full-prefill fallback.
5. A resource planner, CPU validation suite, and staged multi-lab protocol that make executed evidence and future experiments visibly distinct.

## 2. Background and related work

For target layer l and KV head h, the source paper predicts target cache content from a concatenation of all source KV heads in selected source layers:

`K_hat[l,h] = X_K[l] W_K[l,h] + b_K[l,h]`

`V_hat[l,h] = X_V[l] W_V[l,h] + b_V[l,h]`.

The closed-form centered ridge solution is

`W = (X^T X + lambda I)^(-1) X^T Y`,

with `b = mean(Y) - mean(X) W`. For keys, source RoPE is inverted before the map and target RoPE is applied after it. The source study finds that multi-layer aggregation contributes more than a single source layer and that attention-output cosine predicts pair-level retention better than raw reconstruction R² [1]. This distinction controls our evaluation claims: R² is useful for implementation tests, but never sufficient for traffic admission.

Other cross-model work spans different constraints. LatentAlign learns per-model adapters into a shared cache space [2]. Cache-to-Cache trains neural fusers and gates to transfer semantics between LLMs [3]. IAM maps cross-scale attention patterns rather than KV values and reports prefill and cache savings [4]. DroidSpeak selectively recomputes layers while sharing caches between architecturally identical fine-tuned variants [5]. These systems establish the broader opportunity; KVBridge focuses narrowly on productionizing the closed-form, within-family ridge regime from [1].

## 3. System design

### 3.1 Typed cache and compatibility contract

Every layer uses `[batch, kv_heads, tokens, head_dim]`. Construction checks equal K/V shapes, consistent layer geometry, floating-point types, and matching rotary factors. A `ModelSignature` pins model ID, revision, architecture, attention type, tokenizer hash, layer count, KV heads, and head dimension.

Version 0.1 fails closed unless source and target have the same tokenizer fingerprint, matched KV geometry, and dense full attention. The mathematics can express mismatched dimensions, but the source paper did not evaluate that regime. Supporting it by default would turn an open research question into an undocumented production assumption.

### 3.2 Exact content-space keys

RoPE applies an orthogonal, position-dependent rotation to keys. Instead of recomputing frequencies from configuration fields, the Hugging Face adapter calls the model's own rotary module and captures its cosine and sine tensors. Source keys are inverse-rotated, mapped in content space, and rotated with target factors. This design preserves model-specific scaling behavior and makes the artifact independent of the calibration positions within the target model's supported context.

### 3.3 Memory-bounded sufficient statistics

Naively retaining all calibration tokens is unnecessary. For each ridge system, KVBridge accumulates:

`n, mean(X), mean(Y), M2(X), C(X,Y), and M2(Y)`.

Batchwise and distributed merges apply the Chan correction term, so covariance is not recovered by subtracting two large raw moments after accumulation. These statistics directly provide centered covariance, slope, bias, residual error, and R². They merge across shards or processes and support `torch.distributed` all-reduce. Calibration files use SafeTensors and a factory abstraction so each pass reopens only one aligned cache pair at a time.

Let `d_s = k H_s D_s` and `d_t = H_t D_t`. One target layer retains separate K and V statistics with approximate storage

`2 (d_s^2 + d_s d_t + d_s + d_t + 1) q`,

where q is the accumulator byte width. Processing B target layers together multiplies this working set by B, not total target depth. Selection and final fitting have independent block sizes. Lower memory therefore costs more deterministic passes over shards; the planner exposes both sides before a run.

### 3.4 Artifact plane

Weights and biases are stored in SafeTensors, avoiding executable pickle payloads. A JSON manifest records schema version, creation time, estimator configuration, source selections, fit diagnostics, source/target signatures and fingerprints, paper provenance, and the tensor-file SHA-256. Saving writes temporary files and atomically replaces the final paths. Loading verifies the digest before exposing tensors.

Checksums detect modification when the manifest is trusted; they do not authenticate a publisher. Signed release attestations remain a deployment responsibility.

### 3.5 Handoff and failure semantics

For first-token correctness, the Hugging Face handoff retains the final prompt token. The source prefills the preceding prefix; KVBridge maps that prefix; the target consumes the held-back token against the mapped cache and produces the first target logits. Later tokens use the target's normal dynamic cache.

The guarded runtime validates finiteness, magnitude, latency budget, and an application-defined acceptance probe. A concrete short-suffix logit-KL policy supports sampled shadow comparison against full target prefill without placing raw logits in telemetry. Any rejection or backend exception calls full target prefill and emits an event containing status, reason, model pair, token count, and elapsed time. Silent degradation is explicitly disallowed.

## 4. Evaluation methodology

### 4.1 Evidence tiers

We separate four tiers:

- T0: deterministic unit, corruption, planner, package, and fallback tests on any CPU.
- T1: synthetic model-family validation on a CPU, testing planted relationships and shape/RoPE semantics.
- T2: small same-family real-model integration on an available GPU.
- T3: full Qwen3 14B to 32B reproduction and new ablations on partner-lab accelerators.

This report contains T0-T1 measurements and T3 resource estimates. It contains no T2-T3 quality or GPU latency result.

### 4.2 Synthetic construction

Each case creates random source keys and values, chooses one predictive source layer per target layer, and generates target content with independent affine K/V maps plus Gaussian noise with standard deviation 0.001. Source and target use different RoPE bases. Calibration and evaluation sequences are disjoint. The fit uses λ=0.01 and float64 accumulation. Correctness requires exact recovery of every planted source layer and holdout cache R² above 0.999.

Latency measurements use 10 warmups and 100 timed mappings. They run on Windows, Python 3.10.11, CPU-only PyTorch 2.8.0, eight PyTorch threads, and a 12-logical-core Intel processor. Timings describe this reference environment only.

### 4.3 Artifact-precision evidence

The FP32/BF16 experiment saves and reloads the same fitted mapper, then evaluates unseen-cache R² and attention-output cosine under identical grouped-query attention. BF16 reduced the SafeTensors file from 201,608 to 101,760 bytes (50.47% of FP32). Mean attention-output cosine changed from 0.99999907 to 0.99999659, a delta of -0.00000247, on the synthetic case. This supports compact artifact transport in the tested linear setting; it is not evidence that BF16 preserves real-model task accuracy.

### 4.4 Software tests

The 50-test suite covers split-half and interleaved RoPE round trips, invalid cache shapes, missing factors, affine ridge recovery, centered accumulator merging, deterministic tokenizer fingerprints, compatibility rejection, source selection, unseen-sequence mapping, token-strided fitting, CPU/CUDA accumulation rejection, attention-output metrics, logit-KL policy, BF16 artifacts, device residency, SafeTensors round trips, tamper detection, strict JSON evidence validation, calibration-contract migration, re-iterable calibration shards, resource formulas, runtime acceptance, and visible fallback.

### 4.4 Reproducibility controls

The repository records the local environment beside raw JSON and CSV outputs, uses deterministic synthetic seeds, separates warmup from timed iterations, and checks generated artifacts back into a schema-versioned evidence path. The Qwen3 planner is driven by the same JSON configuration intended for capture, preventing documentation-only resource numbers from drifting away from executable settings.

Release verification combines static lint, a two-version CI matrix, unit and adversarial-path tests, a no-download command-line demo, a resource-plan assertion, and Python package construction. Real-model jobs are dry-run by default and require an explicit execution flag after resource inspection.

## 5. Results

### 5.1 Structural recovery and CPU latency

| Case | Geometry | Parameters | Fit (s) | Median map (ms) | p95 (ms) | Holdout R² | Selection |
|---|---|---:|---:|---:|---:|---:|---|
| Micro | 3→2 layers; 2×4; 64 tokens | 288 | 0.012 | 0.339 | 0.376 | 0.99999873 | exact |
| Small | 5→4 layers; 2×8; 128 tokens | 2,176 | 0.043 | 0.781 | 1.033 | 0.99999895 | exact |
| Medium | 8→6 layers; 4×16; 256 tokens | 49,920 | 0.280 | 2.844 | 3.276 | 0.99999892 | exact |

Every planted source layer was recovered, and all holdout scores exceeded the preregistered structural threshold. Latency rose with target depth, token count, and feature width as expected. Because the synthetic target is exactly affine, these near-perfect scores test implementation fidelity; they do not imply that a real model pair will retain equivalent downstream accuracy.

### 5.2 Qwen3 resource plan

For Qwen3 14B to 32B, `H_s = H_t = 8`, `D_s = D_t = 128`, `L_s = 40`, `L_t = 64`, and `k = 8`. KVBridge computes 1,073,872,896 parameters including biases and 4.0005 GiB at FP32. This agrees with the approximately 1.07-billion/4-GB source-paper configuration [1].

With FP32 sufficient statistics, a one-target-layer fit block requires an estimated 0.5626 GiB; an eight-layer selection block requires 0.6299 GiB. One BF16 1,024-token aligned source/target cache pair is approximately 0.4063 GiB. The selected blocks require 72 sequential calibration passes (8 selection plus 64 fit). At 25-50 GB/s host-to-device bandwidth, loading the artifact is estimated at 172-86 ms, close to the source paper's computed range. These are formula-based planning numbers, not measured accelerator results.

### 5.3 Failure-path evidence

The tests alter one byte in a mapper tensor file and verify that loading fails before mapping. A separate guarded-runtime test forces the application quality probe to reject an otherwise valid cache; the request takes the full-prefill fallback and emits a fallback event. These are small tests, but they establish the control-flow property production depends on: rejection is explicit, testable, and non-fatal.

## 6. Proposed further investigation

The original study identifies two hard matched-KV pairs and argues that residual placement relative to attention-sensitive subspaces matters more than global error magnitude [1]. Our next experiments therefore prioritize attention-aware and systems-aware questions.

### 6.1 Precision and block equivalence

Compare monolithic and blockwise fits under FP32 and FP64 accumulation. Measure selected-layer stability, weight difference, R², attention-output cosine, downstream retention, peak RAM, and calibration time. Sufficient-statistic blocking should be mathematically equivalent for a fixed reduction order, but floating-point order may affect ill-conditioned systems.

### 6.2 Artifact compression

Evaluate BF16, per-channel INT8, and truncated-SVD mapper storage. The objective is not minimum reconstruction error alone. Compression should be accepted only when attention-output cosine and task retention remain within pair-specific bounds. A fourfold artifact reduction would materially reduce a multi-model router's host-memory and PCIe costs, but no such gain is claimed before measurement.

### 6.3 Online quality probes

Shadow traffic can compare mapped-cache and full-prefill target logits on sampled requests. Candidate gates include short suffix logit KL, attention-output cosine on selected layers, entropy drift, and canary-task scores. We propose evaluating false-accept and false-reject rates rather than choosing a threshold from reconstruction R².

### 6.4 Distribution and context shift

Cross domain (web, Wikipedia, code, partner-specific text) with context lengths from 1K to 32K. The key question is whether content-space mapping transfers outside the calibration position and register, and whether failure concentrates by task or layer.

### 6.5 Multi-lab reproduction

A partner run should publish model commits, tokenizer hashes, container digest, CUDA stack, topology, raw configurations, benchmark sample counts, paired confidence intervals, and unaggregated latency. K selection must be separated from held-out reporting. The repository's experiment protocol specifies the full matrix.

## 7. Limitations and threats to validity

The local evaluation uses intentionally linear synthetic relationships and cannot measure emergent model behavior, attention sensitivity, downstream accuracy, generation stability, or GPU speed. CPU latencies are single-host measurements without process isolation and should not be compared to the source paper's H100 results. The Hugging Face adapter targets Qwen/Llama-style model-level rotary embeddings and DynamicCache; dependency changes require new integration tests.

The memory planner omits framework allocator fragmentation, model residency, dataset pipeline buffers, communication workspaces, and operating-system pressure. It is a preflight lower-level estimate, not a capacity guarantee. SHA-256 provides integrity but not authentic provenance. Finally, matched KV and shared tokenization are safety gates, not proofs of transferability.

## 8. Broader impact

Avoiding redundant prefill could reduce latency, energy, and accelerator demand in model-routing systems. The same mechanism can also make a fluent target continue from a corrupted internal state. For this reason, KVBridge couples performance work with identity checks, explicit evidence tiers, shadow evaluation, and fallback. Calibration caches may encode sensitive input and should receive the same access, encryption, and deletion controls as the source dataset.

## 9. Conclusion

KVBridge demonstrates that production completeness and modest local resources are compatible when evidence boundaries are explicit. The project implements the closed-form transfer method, bounds its calibration memory, packages artifacts safely, exposes resource costs, and makes failure observable. CPU experiments validate the numerical and control-flow core, while the Qwen3 configuration and multi-lab protocol preserve a credible path to full-model evaluation. The remaining question is empirical rather than rhetorical: which model pairs and compressed artifact regimes meet attention-aware quality gates at serving scale? That question is ready for a transparent partner-lab run.

## References

[1] T. Heo et al. "Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse." arXiv:2608.03893, 2026. https://arxiv.org/abs/2608.03893

[2] L. M. Dery et al. "Latent Space Communication via K-V Cache Alignment." arXiv:2601.06123, 2026. https://arxiv.org/abs/2601.06123

[3] T. Fu et al. "Cache-to-Cache: Direct Semantic Communication Between Large Language Models." ICLR 2026. https://openreview.net/forum?id=LeatkxrBCi

[4] Y. Zhao, Z. Li, and H. Zhao. "IAM: Efficient Inference through Attention Mapping between Different-scale LLMs." ACL 2025. https://aclanthology.org/2025.acl-long.959/

[5] Y. Liu et al. "DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving." arXiv:2411.02820, 2024; NSDI 2026. https://arxiv.org/abs/2411.02820

[6] J. Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." Neurocomputing, 2024. https://arxiv.org/abs/2104.09864

[7] A. Yang et al. "Qwen3 Technical Report." arXiv:2505.09388, 2025. https://arxiv.org/abs/2505.09388

## Appendix A. Reproduction commands

```text
python -m pip install -e ".[dev]"
ruff check src tests experiments
pytest
python -m kvbridge demo --output artifacts/demo
python experiments/run_local_scale.py --repeats 100 --warmup 10
python -m kvbridge plan configs/qwen3_14b_to_32b.paper.json
python -m build
```

## Appendix B. Partnership run package

The repository includes a revision-pinned Qwen3 configuration, a dry-run-by-default capture script, out-of-core shard format, resource preflight, experiment protocol, threat model, and production checklist. A partner should add an approved compute environment, model/dataset credentials, immutable object storage, benchmark orchestration, and signed artifact attestation.
