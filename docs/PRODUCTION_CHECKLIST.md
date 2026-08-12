# Production checklist

## Before calibration

- [ ] Pin source and target checkpoint commits.
- [ ] Compute and compare tokenizer fingerprints.
- [ ] Confirm dense full attention and matched KV geometry.
- [ ] Run `kvbridge plan` and set block sizes within host RAM limits.
- [ ] Record dataset license, access controls, and retention policy.
- [ ] Pin PyTorch, Transformers, CUDA, driver, and attention backend.
- [ ] Pin the calibration dataset revision and hash the executable config.
- [ ] Confirm token stride is identical in the planner and fitter.

## Before artifact promotion

- [ ] Run reconstruction, attention-output, task, and long-context evaluation.
- [ ] Verify held-out benchmark selection discipline.
- [ ] Load the artifact from a clean process and verify its checksum.
- [ ] Archive raw configs/results with code and container digests.
- [ ] Verify artifact storage dtype/byte count and decompress once at process startup.
- [ ] Record attention-output cosine and short-suffix logit KL on held-out prompts.
- [ ] Exercise corruption, incompatibility, timeout, and quality-gate fallbacks.
- [ ] Define pair-specific SLOs and rollback thresholds.

## Before serving

- [ ] Start in shadow mode against full target prefill.
- [ ] Calibrate the logit-KL threshold on shadow traffic; do not adopt a smoke default blindly.
- [ ] Emit structured acceptance/fallback events and dashboards.
- [ ] Bound input length, batch size, cache magnitude, and transfer latency.
- [ ] Canary by model pair and artifact version.
- [ ] Keep full re-prefill available as a tested fallback.
- [ ] Recalibrate on any source, target, tokenizer, or RoPE revision.
