from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aria.io import read_jsonl, write_json_atomic, write_jsonl
from aria.schemas import Example


DEFAULT_PLAN_DIR = Path("data") / "tracefaith_dataset_plan"
DEFAULT_PLAN_PATH = DEFAULT_PLAN_DIR / "tracefaith_500x3_plan.jsonl"
DEFAULT_STATE_PATH = DEFAULT_PLAN_DIR / "tracefaith_500x3_progress.json"


def create_dataset_plan(
    examples: list[Example],
    output_path: Path,
    per_dataset_limit: int = 500,
) -> list[Example]:
    """Persist a stable balanced dataset plan without changing example contents."""
    selected: list[Example] = []
    for dataset in ["scienceqa", "ai2d", "mmmu_pro"]:
        dataset_examples = [example for example in examples if example.dataset == dataset]
        if len(dataset_examples) < per_dataset_limit:
            raise ValueError(
                f"Requested {per_dataset_limit} {dataset} examples for the plan, found {len(dataset_examples)}"
            )
        for dataset_index, example in enumerate(dataset_examples[:per_dataset_limit]):
            metadata = dict(example.metadata)
            metadata.update(
                {
                    "tracefaith_plan_dataset_index": dataset_index,
                    "tracefaith_plan_dataset_limit": per_dataset_limit,
                }
            )
            selected.append(example.model_copy(update={"metadata": metadata}, deep=True))
    planned = []
    for plan_index, example in enumerate(selected):
        metadata = dict(example.metadata)
        metadata["tracefaith_plan_index"] = plan_index
        planned.append(example.model_copy(update={"metadata": metadata}, deep=True))
    write_jsonl(output_path, planned)
    return planned


def load_dataset_plan(path: Path) -> list[Example]:
    return read_jsonl(path, Example)


def select_planned_run(
    plan_path: Path,
    state_path: Path,
    output_dir: Path | None,
    scienceqa_count: int,
    ai2d_count: int,
    mmmu_pro_count: int,
    default_output_root: Path,
) -> tuple[list[Example], Path, dict[str, Any]]:
    """Select the current active planned run or the next pending examples."""
    plan = load_dataset_plan(plan_path)
    state = _read_state(state_path)
    active = state.get("active_run")
    if isinstance(active, dict) and active.get("status") == "running":
        active_output_dir = Path(str(active.get("output_dir") or ""))
        active_ids = [str(item) for item in active.get("selected_example_ids", [])]
        if active_ids and active_output_dir:
            active_examples = _examples_by_ids(plan, active_ids)
            return active_examples, active_output_dir, {
                "dataset_plan_mode": "resume_active_run",
                "dataset_plan_path": str(plan_path),
                "dataset_plan_state_path": str(state_path),
                "dataset_plan_active_run_id": active.get("run_id"),
            }

    completed_ids = set(_completed_ids_from_state(state))
    selected = [
        *_next_pending_by_dataset(plan, completed_ids, "scienceqa", scienceqa_count),
        *_next_pending_by_dataset(plan, completed_ids, "ai2d", ai2d_count),
        *_next_pending_by_dataset(plan, completed_ids, "mmmu_pro", mmmu_pro_count),
    ]
    if not selected:
        raise ValueError(f"No pending examples remain in dataset plan {plan_path}")
    resolved_output_dir = output_dir or _default_planned_output_dir(default_output_root)
    run_id = resolved_output_dir.name
    state["active_run"] = {
        "run_id": run_id,
        "status": "running",
        "output_dir": str(resolved_output_dir),
        "selected_example_ids": [example.example_id for example in selected],
        "selected_counts": _counts_by_dataset(selected),
        "started_at": _now_iso(),
    }
    state.setdefault("runs", []).append(state["active_run"])
    write_json_atomic(state_path, state)
    return selected, resolved_output_dir, {
        "dataset_plan_mode": "next_pending",
        "dataset_plan_path": str(plan_path),
        "dataset_plan_state_path": str(state_path),
        "dataset_plan_active_run_id": run_id,
        "dataset_plan_completed_before_run": len(completed_ids),
    }


def mark_planned_run_completed(
    state_path: Path,
    output_dir: Path,
    examples: list[Example],
) -> None:
    state = _read_state(state_path)
    selected_ids = [example.example_id for example in examples]
    completed = set(_completed_ids_from_state(state))
    completed.update(selected_ids)
    state["completed_example_ids"] = sorted(completed)
    state["last_completed_run"] = {
        "output_dir": str(output_dir),
        "completed_example_ids": selected_ids,
        "completed_counts": _counts_by_dataset(examples),
        "completed_at": _now_iso(),
    }
    active = state.get("active_run")
    if isinstance(active, dict) and Path(str(active.get("output_dir"))) == output_dir:
        active["status"] = "completed"
        active["completed_at"] = state["last_completed_run"]["completed_at"]
        state["active_run"] = None
    for run in state.get("runs", []):
        if isinstance(run, dict) and Path(str(run.get("output_dir"))) == output_dir:
            run["status"] = "completed"
            run["completed_at"] = state["last_completed_run"]["completed_at"]
    write_json_atomic(state_path, state)


def _examples_by_ids(plan: list[Example], example_ids: list[str]) -> list[Example]:
    by_id = {example.example_id: example for example in plan}
    missing = [example_id for example_id in example_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"Dataset plan no longer contains active-run examples: {missing[:5]}")
    return [by_id[example_id] for example_id in example_ids]


def _next_pending_by_dataset(
    plan: list[Example],
    completed_ids: set[str],
    dataset: str,
    count: int,
) -> list[Example]:
    pending = [
        example
        for example in plan
        if example.dataset == dataset and example.example_id not in completed_ids
    ]
    if len(pending) < count:
        raise ValueError(f"Requested {count} pending {dataset} examples, found {len(pending)}")
    return pending[:count]


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"completed_example_ids": [], "runs": [], "active_run": None}
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_ids_from_state(state: dict[str, Any]) -> list[str]:
    return [str(item) for item in state.get("completed_example_ids", [])]


def _counts_by_dataset(examples: list[Example]) -> dict[str, int]:
    return {
        "scienceqa": sum(example.dataset == "scienceqa" for example in examples),
        "ai2d": sum(example.dataset == "ai2d" for example in examples),
        "mmmu_pro": sum(example.dataset == "mmmu_pro" for example in examples),
    }


def _default_planned_output_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"tracefaith_planned_batch_{stamp}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
