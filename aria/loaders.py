from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Any

from aria.io import read_jsonl
from aria.schemas import Example

MMMU_PRO_SCIENCE_SUBJECTS = {
    "Agriculture",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Diagnostics_and_Laboratory_Medicine",
    "Electronics",
    "Energy_and_Power",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Pharmacy",
    "Physics",
    "Public_Health",
}


def load_local_jsonl(path: Path, max_examples: int | None = None) -> list[Example]:
    examples = read_jsonl(path, Example)
    return examples[:max_examples] if max_examples else examples


def load_scienceqa(
    max_examples: int = 100,
    split: str = "validation",
    image_dir: Path | None = None,
    require_image: bool = True,
    require_explanation: bool = False,
    diagram_only: bool = False,
) -> list[Example]:
    """Load ScienceQA examples and materialize images for vision-language models."""
    from datasets import load_dataset

    image_dir = image_dir or Path("data") / "processed" / "scienceqa_images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("derek-thomas/ScienceQA", split=split)
    candidates: list[Example] = []
    for index, row in enumerate(dataset):
        image = row.get("image")
        if require_image and image is None:
            continue
        if require_explanation and not row.get("solution"):
            continue
        if diagram_only and str(row.get("task") or "").lower() != "diagram":
            continue
        choices_raw = row.get("choices") or []
        if not choices_raw:
            continue
        answer_index = row.get("answer")
        if answer_index is None:
            continue
        example_id = str(row.get("id") or f"scienceqa-{split}-{index}")
        image_path = _save_scienceqa_image(image, image_dir, example_id) if image is not None else None
        choices = {chr(ord("A") + i): choice for i, choice in enumerate(choices_raw)}
        correct_answer = chr(ord("A") + int(answer_index)) if answer_index is not None else None
        candidates.append(
            Example(
                example_id=example_id,
                dataset="scienceqa",
                question=row.get("question") or "",
                choices=choices,
                correct_answer=correct_answer,
                context=row.get("hint"),
                lecture=row.get("lecture"),
                explanation=row.get("solution"),
                image_path=str(image_path) if image_path else None,
                image_metadata={
                    "has_image": image is not None,
                    "width": getattr(image, "width", None) if image is not None else None,
                    "height": getattr(image, "height", None) if image is not None else None,
                    "mode": getattr(image, "mode", None) if image is not None else None,
                },
                metadata={
                    "source_index": index,
                    "task": row.get("task"),
                    "grade": row.get("grade"),
                    "subject": row.get("subject"),
                    "topic": row.get("topic"),
                    "category": row.get("category"),
                    "skill": row.get("skill"),
                },
            )
        )
    candidates.sort(key=lambda example: (str(example.metadata.get("task") or "").lower() != "diagram", example.metadata.get("source_index", 0)))
    return candidates[:max_examples]


def load_ai2d(
    max_examples: int = 100,
    split: str | None = None,
    image_dir: Path | None = None,
    local_jsonl_path: Path | None = None,
) -> tuple[list[Example], str]:
    """Load AI2D diagram QA examples from local JSONL or Hugging Face."""
    image_dir = image_dir or Path("data") / "processed" / "ai2d_images" / (split or "auto")
    image_dir.mkdir(parents=True, exist_ok=True)
    if local_jsonl_path is not None and local_jsonl_path.exists():
        return _load_ai2d_local_jsonl(local_jsonl_path, max_examples=max_examples, split=split or "local"), split or "local"

    from datasets import get_dataset_split_names, load_dataset

    dataset_name = "lmms-lab/ai2d"
    available_splits = list(get_dataset_split_names(dataset_name))
    chosen_split = _choose_split(available_splits, preferred=split)
    dataset = load_dataset(dataset_name, split=chosen_split)
    examples: list[Example] = []
    for index, row in enumerate(dataset):
        example = _ai2d_example_from_row(dict(row), index=index, split=chosen_split, image_dir=image_dir)
        if example is None:
            continue
        examples.append(example)
        if len(examples) >= max_examples:
            break
    return examples, chosen_split


def load_mmmu_pro(
    max_examples: int = 100,
    split: str | None = None,
    image_dir: Path | None = None,
    local_jsonl_path: Path | None = None,
    science_subjects_only: bool = True,
) -> tuple[list[Example], str]:
    """Load MMMU-Pro examples from local JSONL or Hugging Face."""
    image_dir = image_dir or Path("data") / "processed" / "mmmu_pro_images" / (split or "auto")
    image_dir.mkdir(parents=True, exist_ok=True)
    if local_jsonl_path is not None and local_jsonl_path.exists():
        return _load_mmmu_pro_local_jsonl(
            local_jsonl_path,
            max_examples=max_examples,
            split=split or "local",
            science_subjects_only=science_subjects_only,
        ), split or "local"

    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    dataset_name = "MMMU/MMMU_Pro"
    config_names = list(get_dataset_config_names(dataset_name))
    chosen_config = "standard (4 options)" if "standard (4 options)" in config_names else (config_names[0] if config_names else None)
    split_names = list(get_dataset_split_names(dataset_name, chosen_config)) if chosen_config else list(get_dataset_split_names(dataset_name))
    chosen_split = _choose_split(split_names, preferred=split)
    dataset = load_dataset(dataset_name, chosen_config, split=chosen_split) if chosen_config else load_dataset(dataset_name, split=chosen_split)
    examples: list[Example] = []
    for index, row in enumerate(dataset):
        if science_subjects_only and str(row.get("subject") or "") not in MMMU_PRO_SCIENCE_SUBJECTS:
            continue
        example = _mmmu_pro_example_from_row(dict(row), index=index, split=chosen_split, image_dir=image_dir, config_name=chosen_config)
        if example is None:
            continue
        examples.append(example)
        if len(examples) >= max_examples:
            break
    resolved_split = f"{chosen_config}:{chosen_split}" if chosen_config else chosen_split
    return examples, resolved_split


def _load_ai2d_local_jsonl(path: Path, max_examples: int, split: str) -> list[Example]:
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                example = Example.model_validate(payload)
                examples.append(_ensure_dataset_metadata(example, dataset="ai2d", split=split))
            except Exception:
                example = _ai2d_example_from_row(payload, index=index, split=split, image_dir=path.parent)
                if example is not None:
                    examples.append(example)
            if len(examples) >= max_examples:
                break
    return examples


def _load_mmmu_pro_local_jsonl(
    path: Path,
    max_examples: int,
    split: str,
    science_subjects_only: bool = True,
) -> list[Example]:
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                example = Example.model_validate(payload)
                if science_subjects_only and str(example.metadata.get("subject") or "") not in MMMU_PRO_SCIENCE_SUBJECTS:
                    continue
                examples.append(_ensure_dataset_metadata(example, dataset="mmmu_pro", split=split))
            except Exception:
                if science_subjects_only and str(payload.get("subject") or payload.get("subfield") or payload.get("topic") or "") not in MMMU_PRO_SCIENCE_SUBJECTS:
                    continue
                example = _mmmu_pro_example_from_row(payload, index=index, split=split, image_dir=path.parent, config_name="local")
                if example is not None:
                    examples.append(example)
            if len(examples) >= max_examples:
                break
    return examples


def _ai2d_example_from_row(row_dict: dict[str, Any], index: int, split: str, image_dir: Path) -> Example | None:
    question = _first_present(row_dict, ["question", "query", "prompt"])
    choices_raw = _first_present(row_dict, ["choices", "options", "answer_choices"])
    answer_raw = _first_present(row_dict, ["answer", "label", "correct_answer", "gt_answer", "gt"])
    image = _first_present(row_dict, ["image", "decoded_image"])
    image_path_raw = _first_present(row_dict, ["image_path", "image_file", "image_filename"])
    if not question or not choices_raw or answer_raw is None:
        return None
    choices = _normalize_choices(choices_raw)
    if not choices:
        return None
    correct_answer = _normalize_answer_label(answer_raw, choices)
    if correct_answer is None:
        return None
    source_id = str(_first_present(row_dict, ["id", "question_id", "qid", "abc_id"]) or f"ai2d-{split}-{index}")
    if image is not None and hasattr(image, "save"):
        image_path = _save_image(image, image_dir, source_id)
        image_metadata = {
            "has_image": True,
            "width": getattr(image, "width", None),
            "height": getattr(image, "height", None),
            "mode": getattr(image, "mode", None),
        }
    elif image_path_raw:
        image_path = Path(str(image_path_raw))
        image_metadata = {"has_image": True}
    else:
        return None
    return Example(
        example_id=f"ai2d-{split}-{source_id}",
        dataset="ai2d",
        question=str(question),
        choices=choices,
        correct_answer=correct_answer,
        image_path=str(image_path),
        image_metadata=image_metadata,
        metadata={
            "source_index": index,
            "source_id": source_id,
            "split": split,
            "task": "diagram",
            "expected_modality_profile": "image_led",
        },
    )


def _mmmu_pro_example_from_row(
    row_dict: dict[str, Any],
    index: int,
    split: str,
    image_dir: Path,
    config_name: str | None,
) -> Example | None:
    question = _first_present(row_dict, ["question", "query", "prompt", "input"])
    choices_raw = _first_present(row_dict, ["choices", "options", "answer_choices", "all_choices"])
    answer_raw = _first_present(row_dict, ["answer", "label", "correct_answer", "gt_answer", "gt"])
    image = _first_present(row_dict, ["image", "decoded_image", "image_1"])
    image_path_raw = _first_present(row_dict, ["image_path", "image_file", "image_filename", "image_url"])
    if not question or not choices_raw or answer_raw is None:
        return None
    choices = _normalize_choices(choices_raw)
    if not choices:
        return None
    correct_answer = _normalize_answer_label(answer_raw, choices)
    if correct_answer is None:
        return None
    source_id = str(_first_present(row_dict, ["id", "question_id", "qid", "sample_id"]) or f"mmmu-pro-{split}-{index}")
    if image is not None and hasattr(image, "save"):
        image_path = _save_image(image, image_dir, source_id)
        image_metadata = {
            "has_image": True,
            "width": getattr(image, "width", None),
            "height": getattr(image, "height", None),
            "mode": getattr(image, "mode", None),
        }
    elif image_path_raw:
        image_path = Path(str(image_path_raw))
        image_metadata = {"has_image": True}
    else:
        return None
    return Example(
        example_id=f"mmmu_pro-{split}-{source_id}",
        dataset="mmmu_pro",
        question=str(question),
        choices=choices,
        correct_answer=correct_answer,
        context=_first_present(row_dict, ["context", "hint", "lecture"]),
        explanation=_first_present(row_dict, ["explanation", "solution", "rationale"]),
        image_path=str(image_path),
        image_metadata=image_metadata,
        metadata={
            "source_index": index,
            "source_id": source_id,
            "split": split,
            "config": config_name,
            "subject": _first_present(row_dict, ["subject", "subfield", "topic"]),
            "discipline": _first_present(row_dict, ["discipline", "category"]),
            "expected_modality_profile": "joint_image_text",
        },
    )


def load_mixed_scienceqa_ai2d(
    scienceqa_count: int = 10,
    ai2d_count: int = 10,
    scienceqa_split: str = "validation",
    ai2d_split: str | None = None,
    scienceqa_image_dir: Path | None = None,
    ai2d_image_dir: Path | None = None,
    ai2d_local_jsonl_path: Path | None = None,
    require_explanation: bool = True,
    diagram_only: bool = False,
) -> tuple[list[Example], dict[str, Any]]:
    scienceqa_examples = load_scienceqa(
        max_examples=scienceqa_count,
        split=scienceqa_split,
        image_dir=scienceqa_image_dir,
        require_image=True,
        require_explanation=require_explanation,
        diagram_only=diagram_only,
    )
    ai2d_examples, chosen_ai2d_split = load_ai2d(
        max_examples=ai2d_count,
        split=ai2d_split,
        image_dir=ai2d_image_dir,
        local_jsonl_path=ai2d_local_jsonl_path,
    )
    examples = [*scienceqa_examples, *ai2d_examples]
    return examples, {
        "scienceqa_count_requested": scienceqa_count,
        "scienceqa_count_loaded": len(scienceqa_examples),
        "scienceqa_split": scienceqa_split,
        "ai2d_count_requested": ai2d_count,
        "ai2d_count_loaded": len(ai2d_examples),
        "ai2d_split": chosen_ai2d_split,
        "ai2d_source": str(ai2d_local_jsonl_path) if ai2d_local_jsonl_path else "lmms-lab/ai2d",
    }


def load_mixed_tracefaith_datasets(
    scienceqa_count: int = 10,
    ai2d_count: int = 10,
    mmmu_pro_count: int = 0,
    scienceqa_split: str = "validation",
    ai2d_split: str | None = None,
    mmmu_pro_split: str | None = None,
    scienceqa_image_dir: Path | None = None,
    ai2d_image_dir: Path | None = None,
    mmmu_pro_image_dir: Path | None = None,
    ai2d_local_jsonl_path: Path | None = None,
    mmmu_pro_local_jsonl_path: Path | None = None,
    mmmu_pro_science_subjects_only: bool = True,
    require_explanation: bool = True,
    diagram_only: bool = False,
) -> tuple[list[Example], dict[str, Any]]:
    scienceqa_examples = load_scienceqa(
        max_examples=scienceqa_count,
        split=scienceqa_split,
        image_dir=scienceqa_image_dir,
        require_image=True,
        require_explanation=require_explanation,
        diagram_only=diagram_only,
    ) if scienceqa_count else []
    ai2d_examples, chosen_ai2d_split = load_ai2d(
        max_examples=ai2d_count,
        split=ai2d_split,
        image_dir=ai2d_image_dir,
        local_jsonl_path=ai2d_local_jsonl_path,
    ) if ai2d_count else ([], ai2d_split or "not_requested")
    mmmu_pro_examples, chosen_mmmu_pro_split = load_mmmu_pro(
        max_examples=mmmu_pro_count,
        split=mmmu_pro_split,
        image_dir=mmmu_pro_image_dir,
        local_jsonl_path=mmmu_pro_local_jsonl_path,
        science_subjects_only=mmmu_pro_science_subjects_only,
    ) if mmmu_pro_count else ([], mmmu_pro_split or "not_requested")
    examples = [*scienceqa_examples, *ai2d_examples, *mmmu_pro_examples]
    return examples, {
        "scienceqa_count_requested": scienceqa_count,
        "scienceqa_count_loaded": len(scienceqa_examples),
        "scienceqa_split": scienceqa_split,
        "ai2d_count_requested": ai2d_count,
        "ai2d_count_loaded": len(ai2d_examples),
        "ai2d_split": chosen_ai2d_split,
        "ai2d_source": str(ai2d_local_jsonl_path) if ai2d_local_jsonl_path else "lmms-lab/ai2d",
        "mmmu_pro_count_requested": mmmu_pro_count,
        "mmmu_pro_count_loaded": len(mmmu_pro_examples),
        "mmmu_pro_split": chosen_mmmu_pro_split,
        "mmmu_pro_source": str(mmmu_pro_local_jsonl_path) if mmmu_pro_local_jsonl_path else "MMMU/MMMU_Pro",
        "mmmu_pro_science_subjects_only": mmmu_pro_science_subjects_only,
        "mmmu_pro_allowed_subjects": sorted(MMMU_PRO_SCIENCE_SUBJECTS),
    }


def _save_scienceqa_image(image: object, image_dir: Path, example_id: str) -> Path:
    return _save_image(image, image_dir, example_id)


def _save_image(image: object, image_dir: Path, example_id: str) -> Path:
    safe_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in example_id)
    path = image_dir / f"{safe_id}.png"
    if path.exists():
        return path
    image.save(path)
    return path


def _choose_split(available_splits: list[str], preferred: str | None = None) -> str:
    if preferred:
        if preferred not in available_splits:
            raise ValueError(f"Requested AI2D split {preferred!r} not found; available splits: {available_splits}")
        return preferred
    for candidate in ["test", "validation", "val", "train"]:
        if candidate in available_splits:
            return candidate
    if not available_splits:
        raise ValueError("AI2D dataset has no available splits")
    return available_splits[0]


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _normalize_choices(choices_raw: Any) -> dict[str, str]:
    if isinstance(choices_raw, str):
        stripped = choices_raw.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return _normalize_choices(ast.literal_eval(stripped))
            except (SyntaxError, ValueError):
                return {}
    if isinstance(choices_raw, dict):
        return {str(key).strip().upper(): str(value) for key, value in choices_raw.items() if str(value).strip()}
    if isinstance(choices_raw, list):
        return {chr(ord("A") + index): str(value) for index, value in enumerate(choices_raw) if str(value).strip()}
    return {}


def _normalize_answer_label(answer_raw: Any, choices: dict[str, str]) -> str | None:
    if isinstance(answer_raw, int):
        labels = sorted(choices)
        return labels[answer_raw] if 0 <= answer_raw < len(labels) else None
    answer = str(answer_raw).strip()
    upper = answer.upper()
    if upper in choices:
        return upper
    for label, text in choices.items():
        if answer == text or answer.lower() == text.lower():
            return label
    if upper.isdigit():
        return _normalize_answer_label(int(upper), choices)
    return None


def _ensure_dataset_metadata(example: Example, dataset: str, split: str) -> Example:
    payload = example.model_dump()
    payload["dataset"] = dataset
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("split", split)
    metadata.setdefault("expected_modality_profile", "image_led")
    payload["metadata"] = metadata
    return Example.model_validate(payload)
