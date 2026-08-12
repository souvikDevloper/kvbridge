import torch

from kvbridge.cache import KVCache
from kvbridge.metrics import attention_output, attention_output_cosine, logit_kl_divergence
from kvbridge.probes import LogitKLPolicy


def test_attention_output_supports_grouped_query_heads() -> None:
    query = torch.randn(1, 4, 3, 8)
    key = torch.randn(1, 2, 3, 8)
    value = torch.randn(1, 2, 3, 8)

    output = attention_output(query, key, value, causal=True)

    assert output.shape == query.shape
    assert output.isfinite().all()


def test_attention_output_cosine_is_one_for_identical_caches() -> None:
    keys = [torch.randn(1, 2, 5, 4) for _ in range(2)]
    values = [torch.randn(1, 2, 5, 4) for _ in range(2)]
    cache = KVCache(keys, values)
    queries = [torch.randn(1, 4, 5, 4) for _ in range(2)]

    report = attention_output_cosine(queries, cache, cache, causal=True)

    assert report.mean > 0.999999
    assert report.minimum > 0.999999


def test_logit_kl_policy_accepts_identity_and_rejects_drift() -> None:
    reference = torch.tensor([[[4.0, 1.0, -2.0], [1.0, 2.0, 3.0]]])
    drifted = torch.tensor([[[-2.0, 1.0, 4.0], [3.0, 2.0, 1.0]]])
    policy = LogitKLPolicy(max_kl=0.01, max_suffix_tokens=2)

    accepted = policy.evaluate(reference, reference)
    rejected = policy.evaluate(drifted, reference)

    assert logit_kl_divergence(reference, reference) == 0.0
    assert accepted.accepted
    assert not rejected.accepted
    assert rejected.value > rejected.threshold
