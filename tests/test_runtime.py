from kvbridge.probes import QualityProbeResult
from kvbridge.runtime import GuardedTransferEngine, GuardPolicy, ShadowSamplingPolicy
from kvbridge.synthetic import fit_demo, make_problem


def test_guard_accepts_valid_transfer() -> None:
    problem = make_problem(calibration_pairs=4, tokens=12)
    events = []
    engine = GuardedTransferEngine(
        fit_demo(problem),
        event_sink=events.append,
    )

    result = engine.run(
        problem.evaluation.source,
        target_rotary=problem.evaluation.target.rotary,
        accept=lambda _: True,
        on_accept=lambda _: "mapped",
        fallback=lambda: "prefilled",
    )

    assert result.status == "accepted"
    assert result.value == "mapped"
    assert events[0]["status"] == "accepted"


def test_guard_falls_back_visibly() -> None:
    problem = make_problem(calibration_pairs=4, tokens=12)
    events = []
    engine = GuardedTransferEngine(fit_demo(problem), event_sink=events.append)

    result = engine.run(
        problem.evaluation.source,
        target_rotary=problem.evaluation.target.rotary,
        accept=lambda _: False,
        on_accept=lambda _: "mapped",
        fallback=lambda: "prefilled",
    )

    assert result.status == "fallback"
    assert result.value == "prefilled"
    assert "quality gate" in result.reason
    assert events[0]["status"] == "fallback"


def test_guard_emits_probe_metrics_and_falls_back() -> None:
    problem = make_problem(calibration_pairs=4, tokens=12)
    events = []
    engine = GuardedTransferEngine(
        fit_demo(problem),
        event_sink=events.append,
    )

    result = engine.run(
        problem.evaluation.source,
        target_rotary=problem.evaluation.target.rotary,
        accept=None,
        on_accept=lambda _: "mapped",
        fallback=lambda: "prefilled",
        quality_probe=lambda _: QualityProbeResult(
            name="short_suffix_logit_kl",
            accepted=False,
            value=0.2,
            threshold=0.1,
        ),
    )

    assert result.status == "fallback"
    assert result.quality_probe is not None
    assert events[0]["shadow_selected"] is True
    assert events[0]["quality_probe"] == "short_suffix_logit_kl"
    assert events[0]["quality_value"] == 0.2


def test_guard_skips_unsampled_shadow_probe() -> None:
    problem = make_problem(calibration_pairs=4, tokens=12)
    events = []
    engine = GuardedTransferEngine(
        fit_demo(problem),
        event_sink=events.append,
        shadow_sampling=ShadowSamplingPolicy(rate=0.0),
    )

    result = engine.run(
        problem.evaluation.source,
        target_rotary=problem.evaluation.target.rotary,
        accept=None,
        on_accept=lambda _: "mapped",
        fallback=lambda: "prefilled",
        quality_probe=lambda _: (_ for _ in ()).throw(AssertionError("should not run")),
        request_id="request-7",
    )

    assert result.status == "accepted"
    assert result.quality_probe is None
    assert events[0]["shadow_selected"] is False


def test_guard_falls_back_before_mapping_on_resource_bound() -> None:
    problem = make_problem(calibration_pairs=4, tokens=12)
    events = []
    engine = GuardedTransferEngine(
        fit_demo(problem),
        policy=GuardPolicy(max_tokens=4),
        event_sink=events.append,
    )

    result = engine.run(
        problem.evaluation.source,
        target_rotary=problem.evaluation.target.rotary,
        accept=None,
        on_accept=lambda _: "mapped",
        fallback=lambda: "prefilled",
    )

    assert result.status == "fallback"
    assert result.transfer_report is None
    assert "token bound" in result.reason


def test_shadow_sampling_is_deterministic_for_request_id() -> None:
    policy = ShadowSamplingPolicy(rate=0.5, salt="test")

    assert policy.select("request-42") is policy.select("request-42")
