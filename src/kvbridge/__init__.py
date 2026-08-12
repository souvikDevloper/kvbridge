"""KVBridge: closed-form cross-model KV-cache transfer."""

from kvbridge.cache import KVCache, RotaryFactors
from kvbridge.config import FitConfig, ModelSignature
from kvbridge.fit import CalibrationPair, fit_mapper
from kvbridge.huggingface import HFCapture
from kvbridge.mapper import CrossModelKVMapper, TransferReport
from kvbridge.metrics import AttentionCosineReport, attention_output_cosine, logit_kl_divergence
from kvbridge.probes import LogitKLPolicy, QualityProbeResult, ShadowLogitKLProbe
from kvbridge.runtime import GuardedResult, GuardedTransferEngine, GuardPolicy, ShadowSamplingPolicy

__all__ = [
    "CalibrationPair",
    "AttentionCosineReport",
    "CrossModelKVMapper",
    "FitConfig",
    "HFCapture",
    "GuardPolicy",
    "GuardedResult",
    "GuardedTransferEngine",
    "KVCache",
    "ModelSignature",
    "LogitKLPolicy",
    "QualityProbeResult",
    "RotaryFactors",
    "TransferReport",
    "ShadowLogitKLProbe",
    "ShadowSamplingPolicy",
    "attention_output_cosine",
    "fit_mapper",
    "logit_kl_divergence",
]

__version__ = "0.2.0"
