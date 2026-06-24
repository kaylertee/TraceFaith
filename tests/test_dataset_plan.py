from __future__ import annotations

import json
from pathlib import Path

from aria.dataset_plan import create_dataset_plan, mark_planned_run_completed, select_planned_run
from aria.schemas import Example
from experiments.run_tracefaith_retargeted_support import _explicit_batch_counts_provided


def test_create_dataset_plan_writes_balanced_manifest(tmp_path: Path) -> None:
    examples = [
        *_examples("scienceqa", 3),
        *_examples("ai2d", 3),
        *_examples("mmmu_pro", 3),
    ]
    path = tmp_path / "plan.jsonl"

    planned = create_dataset_plan(examples, path, per_dataset_limit=2)

    assert path.exists()
    assert [example.dataset for example in planned] == ["scienceqa", "scienceqa", "ai2d", "ai2d", "mmmu_pro", "mmmu_pro"]
    assert planned[0].metadata["tracefaith_plan_index"] == 0
    assert planned[-1].metadata["tracefaith_plan_dataset_index"] == 1


def test_select_planned_run_resumes_active_run(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.jsonl"
    state_path = tmp_path / "state.json"
    create_dataset_plan([*_examples("scienceqa", 3), *_examples("ai2d", 3), *_examples("mmmu_pro", 3)], plan_path, 3)

    first_examples, first_output, first_config = select_planned_run(
        plan_path=plan_path,
        state_path=state_path,
        output_dir=tmp_path / "run1",
        scienceqa_count=1,
        ai2d_count=1,
        mmmu_pro_count=1,
        default_output_root=tmp_path,
    )
    resumed_examples, resumed_output, resumed_config = select_planned_run(
        plan_path=plan_path,
        state_path=state_path,
        output_dir=None,
        scienceqa_count=2,
        ai2d_count=2,
        mmmu_pro_count=2,
        default_output_root=tmp_path,
    )

    assert [example.example_id for example in resumed_examples] == [example.example_id for example in first_examples]
    assert resumed_output == first_output
    assert first_config["dataset_plan_mode"] == "next_pending"
    assert resumed_config["dataset_plan_mode"] == "resume_active_run"


def test_planned_run_marks_completed_and_selects_next_pending(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.jsonl"
    state_path = tmp_path / "state.json"
    create_dataset_plan([*_examples("scienceqa", 3), *_examples("ai2d", 3), *_examples("mmmu_pro", 3)], plan_path, 3)

    first_examples, first_output, _ = select_planned_run(
        plan_path=plan_path,
        state_path=state_path,
        output_dir=tmp_path / "run1",
        scienceqa_count=1,
        ai2d_count=1,
        mmmu_pro_count=1,
        default_output_root=tmp_path,
    )
    mark_planned_run_completed(state_path, first_output, first_examples)
    second_examples, second_output, _ = select_planned_run(
        plan_path=plan_path,
        state_path=state_path,
        output_dir=tmp_path / "run2",
        scienceqa_count=1,
        ai2d_count=1,
        mmmu_pro_count=1,
        default_output_root=tmp_path,
    )

    assert second_output == tmp_path / "run2"
    assert [example.example_id for example in second_examples] == ["scienceqa-1", "ai2d-1", "mmmu_pro-1"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert sorted(state["completed_example_ids"]) == ["ai2d-0", "mmmu_pro-0", "scienceqa-0"]


def test_explicit_batch_counts_detects_any_dataset_count() -> None:
    class Args:
        scienceqa_count = None
        ai2d_count = None
        mmmu_pro_count = None

    args = Args()
    assert not _explicit_batch_counts_provided(args)
    args.ai2d_count = 10
    assert _explicit_batch_counts_provided(args)


def _examples(dataset: str, count: int) -> list[Example]:
    return [
        Example(
            example_id=f"{dataset}-{index}",
            dataset=dataset,
            question=f"{dataset} question {index}",
            choices={"A": "yes", "B": "no"},
            correct_answer="A",
            image_path=f"/tmp/{dataset}-{index}.png",
            metadata={"source_index": index},
        )
        for index in range(count)
    ]
