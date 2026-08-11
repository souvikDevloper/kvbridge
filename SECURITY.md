# Security policy

Please report vulnerabilities privately to the repository owner before public disclosure. Include affected version, reproduction steps, impact, and a minimal proof of concept.

KVBridge artifacts and calibration shards use SafeTensors and never call `torch.load`. SHA-256 detects accidental or malicious modification when the manifest is trusted; it is not a substitute for signed release provenance. Treat model repositories, custom modeling code, calibration caches, and artifact storage as supply-chain boundaries.

Only the latest minor release receives security fixes during the pre-1.0 phase.
