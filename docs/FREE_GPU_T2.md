# Free-GPU T2 runbook

This runbook produces a real-model integration result for Qwen3 0.6B to 1.7B. It is a T2 systems/quality smoke test, not a reproduction of the paper's 14B to 32B result and not evidence about H100 performance.

## Provider choice

Kaggle is the first target because its official notebook documentation provides free Tesla P100 access with a weekly quota that is commonly about 30 hours. A P100 does not provide native BF16 execution, so the pinned smoke config loads model weights in FP16 and decompresses the BF16 mapper artifact to resident FP32 tensors for mapping.

Free H100/H200 availability is opportunistic rather than guaranteed. Lightning AI and Modal offer starter credits that may cover short accelerator runs, but instance availability, identity verification, and current credit policy are account-dependent. Record the actual device from the result manifest; never relabel a P100/T4/L4 run as an H100 run.

## Kaggle notebook steps

1. Create a notebook with Internet enabled and select a GPU accelerator.
2. Run the following in a fresh session:

```bash
!git clone https://github.com/souvikDevloper/kvbridge.git
%cd kvbridge
!bash scripts/kaggle_t2_smoke.sh
```

3. Download and retain these outputs before the session expires:

- `data/calibration/capture_manifest.json`
- `artifacts/qwen3-0.6b-to-1.7b/manifest.json`
- `artifacts/qwen3-0.6b-to-1.7b/fit_run.json`
- `results/qwen3-0.6b-to-1.7b.t2.json`

The driver is fail-fast and all real-model scripts are dry-run by default outside it. Each executed result records the code commit, config and artifact hashes, pinned model/dataset revisions, CUDA/PyTorch versions, actual GPU name, memory, and unaggregated per-sequence metrics.

## Acceptance boundary

The config preregisters an attention-output cosine floor of 0.90 and a p95 one-token logit KL ceiling of 0.20. These are initial smoke gates, not universal production SLOs. A failed gate is a valid research result and must remain visible.

Before any serving claim, expand evaluation to downstream tasks, longer contexts, multiple domains, more seeds, confidence intervals, and shadow-traffic false-accept/false-reject analysis. Promote a model pair only after pair-specific thresholds and rollback drills pass.

## Larger single-GPU follow-up

`configs/qwen3_1.7b_to_4b.t2-headline.json` is the next single-GPU experiment. Prefer a verified 24 GB or larger accelerator. Run the same three scripts explicitly rather than changing the smoke result in place, and write to a distinct calibration, artifact, and result path.
