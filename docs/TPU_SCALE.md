# TPU v5e/v6e scale runbook

KVBridge has one PyTorch/XLA SPMD path for both Kaggle TPU v5e-8 and Cloud TPU
v6e-8. Kaggle is the free compatibility and reduced-scale validation lane.
An approved [TPU Research Cloud](https://sites.research.google/trc/) allocation
is the no-card route to a full v6e experiment; TRC approval and a particular
TPU generation are not guaranteed.

## Hardware decision

| Runtime | Accelerator memory | KVBridge role |
|---|---:|---|
| Kaggle TPU v5e-8 | 128 GB total high-speed memory | Free SPMD/compiler validation; attempt 128K only if quota and session time allow |
| TPU v6e-8 | 256 GB HBM, 1.44 TB host RAM | Recommended 14B→32B 128K run and 256K calibration ablation |

The Kaggle notebook UI exposed TPU v5e-8 with a 20-hour weekly quota on
2026-08-17. Treat accelerator availability and quotas as provider state that can
change, and inspect the notebook UI before every run. Google documents v6e-8 as
one eight-chip, single-VM slice optimized for inference.

Official references: [Kaggle TPU guide](https://www.kaggle.com/docs/tpu),
[Cloud TPU v6e](https://cloud.google.com/tpu/docs/v6e),
[PyTorch/XLA SPMD](https://docs.pytorch.org/xla/master/spmd.html), and
[TRC FAQ](https://sites.research.google/trc/faq/).

## What the scale path changes

The legacy capture path co-loads both models and writes roughly 406 GiB of
unsampled caches for the 500-sequence paper configuration. The TPU path instead:

1. loads only Qwen3-14B and shards each parameter across the SPMD mesh;
2. captures content-space K/V and applies stride four before gathering to host;
3. releases the source model and its temporary download cache;
4. repeats capture with Qwen3-32B;
5. fits vectorized centered sufficient statistics with one target-layer block
   resident at a time; and
6. atomically checkpoints every selection block and fitted target layer.

The 128K config retains about 50.8 GiB of sampled caches in host memory. The
256K ablation retains about 101.6 GiB. Model weights and sampled caches are not
persisted as raw public evidence; the revision-bound mapper, checkpoint
manifest, and `fit_run.json` are. A restarted job must recapture host caches,
but completed fit blocks are verified and skipped.

## Kaggle v5e-8

Create a notebook, enable Internet, select **TPU v5e-8**, then run:

```bash
git clone https://github.com/souvikDevloper/kvbridge.git
cd kvbridge
python experiments/run_tpu_scale.py \
  configs/qwen3_14b_to_32b.tpu-128k.json
```

That command is a no-download dry run. Confirm it reports 128,000 observations,
about 50.8 GiB of sampled host caches, eight required XLA devices, and a 2.0
GiB BF16 mapper. Then execute:

```bash
bash scripts/tpu_scale.sh
```

Do not label an interrupted, fit-only, or quality-rejected run a reproduction.
Kaggle is industry-valid evidence for numerical correctness on the declared TPU
v5e-8 runtime. It is not NVIDIA/H100 latency evidence.

## TRC v6e-8

Request a single-host v6e-8 allocation in the TRC application and explain that
the project is open research with public code, revision-pinned artifacts, and a
multi-lab replication plan. On the provisioned TPU VM, use the same command:

```bash
export KVBRIDGE_TPU_CONFIG=configs/qwen3_14b_to_32b.tpu-128k.json
export KVBRIDGE_TPU_OUTPUT_DIR=runs/qwen3-14b-to-32b-v6e-128k
bash scripts/tpu_scale.sh
```

Only after the 128K run fits and passes evaluation should the exploratory 256K
ablation be launched:

```bash
export KVBRIDGE_TPU_CONFIG=configs/qwen3_14b_to_32b.tpu-256k.json
export KVBRIDGE_TPU_OUTPUT_DIR=runs/qwen3-14b-to-32b-v6e-256k
bash scripts/tpu_scale.sh
```

Use a persistent compile cache and checkpoint directory on the TPU VM or a
project-controlled bucket. TRC provides TPU quota, but associated VM, storage,
network, or bucket charges may still apply; verify the project billing policy
before provisioning anything outside the award.

## Evidence boundary

`run_tpu_scale.py` produces a mapper fit and labels its manifest
`fit-complete-evaluation-pending`. `scripts/tpu_scale.sh` then runs the held-out
TPU diagnostics unless `KVBRIDGE_TPU_EVALUATE=0`. Validate either stage without
model downloads:

```bash
python experiments/validate_tpu_evidence.py \
  configs/qwen3_14b_to_32b.tpu-128k.json \
  --run-dir runs/qwen3-14b-to-32b-tpu-128k \
  --result runs/qwen3-14b-to-32b-tpu-128k/evaluation.json
```

Those diagnostics are intentionally not a complete paper result. Publication
also requires downstream-retention, long-context, and hardware-local latency
results under `docs/EXPERIMENT_PROTOCOL.md`. Every table must name the exact TPU
generation; v5e timings cannot be presented as v6e, H100, or H200 timings.
