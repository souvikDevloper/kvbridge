# Lightning H100/H200 pilot runbook

The first high-memory pilot is Qwen3 8B→14B, not Qwen3.5 27B. The chosen pair stays inside the dense, matched-KV Qwen3 regime implemented and studied by KVBridge; Qwen3.5 27B is a later cross-generation/architecture generalization experiment. The two BF16 checkpoints total about 46 GB on disk, versus roughly 95 GB for Qwen3 14B→32B, making the pilot compatible with Lightning's free-tier 50 GB persistent-storage limit and a single 80 GB H100.

Lightning's current public pricing grants free accounts 15 monthly credits, lists single H100/H200 access, and bills compute per second. Availability and account eligibility are not guaranteed. Inspect the quoted machine price and remaining credits before starting, enable auto-stop, and stop the Studio immediately after exporting evidence. The free Studio session restarts every four hours, so do dependency setup on CPU first and start the GPU only for capture/fit/evaluation.

## Preflight on the free CPU Studio

```bash
git clone https://github.com/souvikDevloper/kvbridge.git
cd kvbridge
python -m pip install -e ".[hf]"
python -m kvbridge plan configs/qwen3_8b_to_14b.t2-h100-pilot.json
```

The plan should report 8,192 observations, 335,626,240 mapper parameters, approximately 0.625 GiB BF16 artifact storage, and 50 calibration-data passes. Do not start if the actual GPU, free disk, credit balance, or four-hour window cannot cover the run.

## GPU execution

Select one H100 80 GB or H200 141 GB, record the provider/instance identifier, then run:

```bash
cd kvbridge
nvidia-smi
bash scripts/lightning_t2_8b_to_14b.sh
```

The driver is resumable only at completed stage boundaries. Download `capture_manifest.json`, `manifest.json`, `fit_run.json`, and the result JSON as soon as the run completes. A nonzero final exit is expected when valid evidence fails a preregistered quality gate; validate and export the files before stopping the machine.

## Why not 27B first?

Model size alone does not make stronger evidence. A 27B Qwen3.5 run changes the model generation and may change attention/cache semantics, confounding numerical hardening with architectural generalization. The evidence ladder is:

1. Qwen3 8B→14B high-memory pilot.
2. Qwen3 14B→32B paper-aligned reproduction on H200 or partner hardware.
3. Qwen3.5 27B generalization only after its cache geometry and full-attention compatibility pass preflight.

Official references: [Lightning pricing](https://lightning.ai/pricing), [Lightning billing](https://lightning.ai/docs/overview/faq/billing), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), and [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B).
