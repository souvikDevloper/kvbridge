from kvbridge.runtime import GuardedTransferEngine
from kvbridge.synthetic import fit_demo, make_problem


def test_guard_accepts_valid_transfer() -> None:
    problem = make_problem(calibration_pairs=4, tokens=12)
    events = []
    engine = GuardedTransferEngine(fit_demo(problem), event_sink=events.append)

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
