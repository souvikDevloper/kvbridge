# Threat model

## Assets

- model and tokenizer identity;
- calibration data and derived caches;
- mapper weights/manifests;
- request continuity and generated output;
- latency and fallback telemetry.

## Risks and mitigations

| Risk | Failure | Mitigation |
|---|---|---|
| Wrong checkpoint | Plausible but invalid cache | Revisioned model signature and fingerprint |
| Tokenizer drift | Token/position misalignment | Canonical vocabulary/special-token SHA-256 gate |
| RoPE mismatch | Context-length-dependent corruption | Capture factors from each model's own rotary module |
| Artifact tampering | Altered weights | SafeTensors plus manifest SHA-256 verification |
| Malicious pickle | Code execution during load | No pickle-based artifact or shard format |
| Numerical explosion | NaN/Inf or extreme activations | Finite and magnitude gates with fallback |
| Silent quality regression | Fluent but wrong continuation | Shadow oracle, application probe, visible fallback events |
| Resource exhaustion | Fit or map OOM | Preflight planner, target-layer blocks, explicit limits |
| Dependency drift | Cache API/layout changes | Version pin, integration matrix, revision canary |
| Calibration leakage | Sensitive text/state exposure | Encrypted storage, least privilege, retention policy |

## Non-goals

The checksum is not a publisher signature, and KVBridge does not sandbox model code loaded with `trust_remote_code`. Production distribution should add signed release attestations and avoid remote code unless independently reviewed.
