# Qwen3 0.6B→1.7B T2 smoke evidence

**Outcome: valid execution, quality gates rejected.** This is a real-model single-GPU integration result, not a paper-scale reproduction and not a production-acceptance claim.

## Run contract

- UTC execution date: 2026-08-17
- Code revision: `e69e13922f4a4e7d13110d4b3fd6c1056e1d15a4`
- GPU: Tesla T4, 15 GiB; CUDA runtime 12.8; PyTorch 2.11.0
- Pair: revision-pinned `Qwen/Qwen3-0.6B` → `Qwen/Qwen3-1.7B`
- Calibration: 16 FineWeb-Edu sequences × 512 tokens, stride 4; 2,048 observations
- Evaluation: eight held-out sequences × 256 tokens
- Fit: top-k 1, λ=0.01, content-space keys, CUDA FP32 accumulation
- Artifact: 58,777,600 parameters, BF16 storage, 0.1095 GiB planned
- Fit time: 128.68 s; peak fitting allocation: 542,900,224 bytes

## Result

| Metric | Estimate | 95% bootstrap interval where defined |
|---|---:|---:|
| Cache R² mean | 0.234721 | [0.143677, 0.313710] |
| Attention-output cosine mean | 0.656811 | [0.633500, 0.674282] |
| Attention-output cosine minimum | 0.293665 | — |
| Logit KL mean | 1.614515 | [0.858162, 2.613657] |
| Logit KL p95 | 4.590321 | — |
| Next-token agreement | 0.25 | [0.00, 0.625] |
| Transfer median / p95 | 46.090 / 54.234 ms | — |
| Target prefix-prefill median | 87.291 ms | — |
| Per-case prefill/transfer ratio median | 3.074× | — |

The attention minimum was below the preregistered 0.90 floor and KL p95 was above the 0.20 ceiling. Both gates and the combined gate are `false`. The correct serving decision is fallback, even though mapping was faster in this run.

## Published files and hashes

| File | SHA-256 |
|---|---|
| `capture_manifest.json` | `d50af9e5d1ca68989c8121875f35cb247ec470a0f7650d3bd85d479e258115da` |
| `mapper_manifest.json` | `0ebbbb025c1647b9b9fce7491437952ac87b43e7296b11d757c03cf612ef0401` |
| `fit_run.json` | `3f6997587bd65891773a0e164585df622e532f487148c692e34923c85681d1a4` |
| `result.json` | `41f9586a0fd4b9602b03dde4be0afb5187ac6fff57f4740a980faaffd3946e9d` |

`result.json` contains every per-sequence metric and per-layer attention cosine. The lightweight validator verifies the config/capture/fit/mapper/result hash chain and recomputes all aggregates, deterministic bootstrap intervals, and gate decisions:

```bash
python experiments/validate_published_evidence.py \
  configs/qwen3_0.6b_to_1.7b.t2-smoke.json \
  results/t2/qwen3-0.6b-to-1.7b-smoke
```

The repository intentionally omits the 1.887 GB calibration shards and 112 MB mapper weights. Their content hashes remain in the manifests, and the full strict validator passed in the originating Colab runtime before export. The lightweight validator reports `calibration_shards_verified: false` and `mapper_weights_verified: false` so this storage boundary cannot be confused with full artifact re-verification.
