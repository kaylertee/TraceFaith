import json
from pathlib import Path

from aria import loaders
from aria.loaders import load_ai2d, load_mixed_scienceqa_ai2d, load_mixed_tracefaith_datasets, load_mmmu_pro
from aria.schemas import Example


def test_load_ai2d_local_jsonl_normalizes_raw_rows(tmp_path: Path) -> None:
    local_path = tmp_path / "ai2d.jsonl"
    local_path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Which organism is at the start of the food chain?",
                "options": ["grass", "frog", "snake", "hawk"],
                "answer": 0,
                "image_path": "/tmp/ai2d-q1.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples, split = load_ai2d(max_examples=1, split="localtest", local_jsonl_path=local_path)

    assert split == "localtest"
    assert examples[0].dataset == "ai2d"
    assert examples[0].example_id == "ai2d-localtest-q1"
    assert examples[0].choices == {"A": "grass", "B": "frog", "C": "snake", "D": "hawk"}
    assert examples[0].correct_answer == "A"
    assert examples[0].metadata["expected_modality_profile"] == "image_led"


def test_load_mmmu_pro_local_jsonl_normalizes_raw_rows(tmp_path: Path) -> None:
    local_path = tmp_path / "mmmu_pro.jsonl"
    local_path.write_text(
        json.dumps(
            {
                "id": "m1",
                "question": "Which graph supports the conclusion?",
                "options": ["Graph A", "Graph B", "Graph C", "Graph D"],
                "answer": "C",
                "image_path": "/tmp/mmmu-pro-m1.png",
                "subject": "Physics",
                "discipline": "Science",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples, split = load_mmmu_pro(max_examples=1, split="localtest", local_jsonl_path=local_path)

    assert split == "localtest"
    assert examples[0].dataset == "mmmu_pro"
    assert examples[0].example_id == "mmmu_pro-localtest-m1"
    assert examples[0].choices["C"] == "Graph C"
    assert examples[0].correct_answer == "C"
    assert examples[0].metadata["discipline"] == "Science"


def test_load_mmmu_pro_local_jsonl_filters_non_science_subjects(tmp_path: Path) -> None:
    local_path = tmp_path / "mmmu_pro.jsonl"
    rows = [
        {
            "id": "history",
            "question": "Which timeline event is shown?",
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "image_path": "/tmp/history.png",
            "subject": "History",
        },
        {
            "id": "physics",
            "question": "Which circuit is closed?",
            "options": ["A", "B", "C", "D"],
            "answer": "B",
            "image_path": "/tmp/physics.png",
            "subject": "Physics",
        },
    ]
    local_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    examples, _ = load_mmmu_pro(max_examples=10, split="localtest", local_jsonl_path=local_path)

    assert [example.example_id for example in examples] == ["mmmu_pro-localtest-physics"]


def test_load_mixed_scienceqa_ai2d_returns_stable_counts(monkeypatch) -> None:
    def fake_scienceqa(**kwargs):
        return [
            Example(
                example_id=f"scienceqa-{index}",
                dataset="scienceqa",
                question="ScienceQA?",
                choices={"A": "yes", "B": "no"},
                correct_answer="A",
                image_path=f"/tmp/scienceqa-{index}.png",
            )
            for index in range(kwargs["max_examples"])
        ]

    def fake_ai2d(**kwargs):
        return (
            [
                Example(
                    example_id=f"ai2d-{index}",
                    dataset="ai2d",
                    question="AI2D?",
                    choices={"A": "yes", "B": "no"},
                    correct_answer="A",
                    image_path=f"/tmp/ai2d-{index}.png",
                )
                for index in range(kwargs["max_examples"])
            ],
            "test",
        )

    monkeypatch.setattr(loaders, "load_scienceqa", fake_scienceqa)
    monkeypatch.setattr(loaders, "load_ai2d", fake_ai2d)

    examples, config = load_mixed_scienceqa_ai2d(scienceqa_count=10, ai2d_count=10)

    assert len(examples) == 20
    assert [example.dataset for example in examples[:10]] == ["scienceqa"] * 10
    assert [example.dataset for example in examples[10:]] == ["ai2d"] * 10
    assert config["scienceqa_count_loaded"] == 10
    assert config["ai2d_count_loaded"] == 10
    assert config["ai2d_split"] == "test"


def test_load_mixed_tracefaith_datasets_supports_zero_counts(monkeypatch) -> None:
    def fake_mmmu_pro(**kwargs):
        return (
            [
                Example(
                    example_id=f"mmmu-pro-{index}",
                    dataset="mmmu_pro",
                    question="MMMU-Pro?",
                    choices={"A": "yes", "B": "no"},
                    correct_answer="A",
                    image_path=f"/tmp/mmmu-pro-{index}.png",
                )
                for index in range(kwargs["max_examples"])
            ],
            "validation",
        )

    monkeypatch.setattr(loaders, "load_mmmu_pro", fake_mmmu_pro)

    examples, config = load_mixed_tracefaith_datasets(scienceqa_count=0, ai2d_count=0, mmmu_pro_count=2)

    assert len(examples) == 2
    assert [example.dataset for example in examples] == ["mmmu_pro", "mmmu_pro"]
    assert config["scienceqa_count_loaded"] == 0
    assert config["ai2d_count_loaded"] == 0
    assert config["mmmu_pro_count_loaded"] == 2
