from pathlib import Path

from kvbridge.planning import ExperimentConfig, build_scale_plan


def test_qwen_plan_matches_paper_mapper_footprint() -> None:
    config = ExperimentConfig.load(
        Path(__file__).parents[1] / "configs" / "qwen3_14b_to_32b.paper.json"
    )
    plan = build_scale_plan(config)

    assert plan.mapper_parameters == 1_073_872_896
    assert 4.0 <= plan.mapper_gib < 4.01
    assert plan.observations == 128_000
    assert plan.fit_block_working_set_gib < 0.6
