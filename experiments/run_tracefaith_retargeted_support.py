from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aria.hf_models import (
    DEFAULT_INTERVENTION_MODEL_ID,
    DEFAULT_TARGET_MODEL_ID,
    HFGenerationConfig,
    InterventionModel,
    TargetRerunModel,
)
from aria.dataset_plan import (
    DEFAULT_PLAN_PATH,
    DEFAULT_STATE_PATH,
    mark_planned_run_completed,
    select_planned_run,
)
from aria.loaders import load_mixed_tracefaith_datasets
from aria.pipeline import run_faithfulness_evaluation
from aria.schemas import Example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TraceFaith intervention-model behavioral experiment.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--scienceqa-count", type=int, default=None)
    parser.add_argument("--ai2d-count", type=int, default=None)
    parser.add_argument("--mmmu-pro-count", type=int, default=None)
    parser.add_argument("--planned-run", action="store_true")
    parser.add_argument("--dataset-plan-jsonl", type=Path, default=ROOT / DEFAULT_PLAN_PATH)
    parser.add_argument("--dataset-plan-state", type=Path, default=ROOT / DEFAULT_STATE_PATH)
    parser.add_argument("--planned-default-per-dataset", type=int, default=50)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL_ID)
    parser.add_argument("--intervention-model", default=DEFAULT_INTERVENTION_MODEL_ID)
    parser.add_argument("--scienceqa-split", default="validation")
    parser.add_argument("--ai2d-split", default=None)
    parser.add_argument("--mmmu-pro-split", default=None)
    parser.add_argument("--ai2d-local-jsonl", type=Path, default=None)
    parser.add_argument("--mmmu-pro-local-jsonl", type=Path, default=None)
    parser.add_argument("--mmmu-pro-include-non-science", action="store_true")
    parser.add_argument(
        "--scienceqa-image-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "scienceqa_images",
    )
    parser.add_argument(
        "--ai2d-image-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "ai2d_images",
    )
    parser.add_argument(
        "--mmmu-pro-image-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "mmmu_pro_images",
    )
    parser.add_argument("--require-explanation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scienceqa-diagram-only", action="store_true")
    parser.add_argument("--skip-target-rerun", action="store_true")
    parser.add_argument("--evaluation-chunk-size", type=int, default=1)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=20260613)
    parser.add_argument("--selection-pool-multiplier", type=int, default=8)
    parser.add_argument("--exclude-selected-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--target-quantization", choices=["auto", "prequantized", "bnb4", "none"], default="bnb4")
    parser.add_argument("--intervention-quantization", choices=["auto", "prequantized", "bnb4", "none"], default="bnb4")
    parser.add_argument("--allow-cpu-offload", action="store_true")
    parser.add_argument("--min-visual-tokens", type=int, default=256)
    parser.add_argument("--max-visual-tokens", type=int, default=768)
    parser.add_argument("--trace-max-new-tokens", type=int, default=512)
    parser.add_argument("--intervention-max-new-tokens", type=int, default=512)
    parser.add_argument("--target-max-new-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auto_continue_planned = args.planned_run and not _explicit_batch_counts_provided(args)
    if auto_continue_planned and args.output_dir is not None:
        raise ValueError("--output-dir cannot be used with auto-continuing planned runs; each batch needs its own output directory.")
    while True:
        try:
            _run_once(args)
        except ValueError as exc:
            if auto_continue_planned and "No pending examples remain" in str(exc):
                print("TraceFaith dataset plan is complete; no pending examples remain.")
                break
            raise
        if not auto_continue_planned:
            break


def _run_once(args: argparse.Namespace) -> None:
    scienceqa_count, ai2d_count, mmmu_pro_count = _resolved_counts(args)
    output_dir = args.output_dir or ROOT / "results" / "tracefaith_intervention_model_scienceqa_ai2d_mmmupro"
    dataset_config = {}
    planned_state_path: Path | None = None
    planned_examples: list[Example] | None = None
    if args.planned_run:
        planned_state_path = args.dataset_plan_state
        planned_examples, output_dir, planned_config = select_planned_run(
            plan_path=args.dataset_plan_jsonl,
            state_path=args.dataset_plan_state,
            output_dir=args.output_dir,
            scienceqa_count=scienceqa_count,
            ai2d_count=ai2d_count,
            mmmu_pro_count=mmmu_pro_count,
            default_output_root=ROOT / "results",
        )
        examples = planned_examples
        dataset_config.update(
            {
                **planned_config,
                "scienceqa_count_requested": scienceqa_count,
                "scienceqa_count_loaded": sum(example.dataset == "scienceqa" for example in examples),
                "scienceqa_split": args.scienceqa_split,
                "ai2d_count_requested": ai2d_count,
                "ai2d_count_loaded": sum(example.dataset == "ai2d" for example in examples),
                "ai2d_split": "from_dataset_plan",
                "ai2d_source": str(args.dataset_plan_jsonl),
                "mmmu_pro_count_requested": mmmu_pro_count,
                "mmmu_pro_count_loaded": sum(example.dataset == "mmmu_pro" for example in examples),
                "mmmu_pro_split": "from_dataset_plan",
                "mmmu_pro_source": str(args.dataset_plan_jsonl),
                "mmmu_pro_science_subjects_only": not args.mmmu_pro_include_non_science,
            }
        )
    else:
        load_scienceqa_count = scienceqa_count
        load_ai2d_count = ai2d_count
        load_mmmu_pro_count = mmmu_pro_count
        if args.random_sample:
            load_scienceqa_count = max(scienceqa_count, scienceqa_count * args.selection_pool_multiplier)
            load_ai2d_count = max(ai2d_count, ai2d_count * args.selection_pool_multiplier)
            load_mmmu_pro_count = max(mmmu_pro_count, mmmu_pro_count * args.selection_pool_multiplier)
        examples, dataset_config = load_mixed_tracefaith_datasets(
            scienceqa_count=load_scienceqa_count,
            ai2d_count=load_ai2d_count,
            mmmu_pro_count=load_mmmu_pro_count,
            scienceqa_split=args.scienceqa_split,
            ai2d_split=args.ai2d_split,
            mmmu_pro_split=args.mmmu_pro_split,
            scienceqa_image_dir=args.scienceqa_image_dir / args.scienceqa_split,
            ai2d_image_dir=args.ai2d_image_dir / (args.ai2d_split or "auto"),
            mmmu_pro_image_dir=args.mmmu_pro_image_dir / (args.mmmu_pro_split or "auto"),
            ai2d_local_jsonl_path=args.ai2d_local_jsonl,
            mmmu_pro_local_jsonl_path=args.mmmu_pro_local_jsonl,
            mmmu_pro_science_subjects_only=not args.mmmu_pro_include_non_science,
            require_explanation=args.require_explanation,
            diagram_only=args.scienceqa_diagram_only,
        )
        excluded_ids = _read_excluded_example_ids(args.exclude_selected_jsonl)
        if args.random_sample:
            examples = _sample_examples(
                examples=examples,
                scienceqa_count=scienceqa_count,
                ai2d_count=ai2d_count,
                mmmu_pro_count=mmmu_pro_count,
                seed=args.sample_seed,
                excluded_ids=excluded_ids,
            )
            dataset_config.update(
                {
                    "random_sample": True,
                    "sample_seed": args.sample_seed,
                    "selection_pool_multiplier": args.selection_pool_multiplier,
                    "excluded_manifest_paths": [str(path) for path in args.exclude_selected_jsonl],
                    "excluded_example_count": len(excluded_ids),
                    "scienceqa_count_loaded": sum(example.dataset == "scienceqa" for example in examples),
                    "ai2d_count_loaded": sum(example.dataset == "ai2d" for example in examples),
                    "mmmu_pro_count_loaded": sum(example.dataset == "mmmu_pro" for example in examples),
                }
            )
    config = {
        "experiment": "tracefaith",
        "experiment_stage": "intervention_model_scienceqa_ai2d_mmmupro",
        "dataset": "scienceqa+ai2d+mmmu_pro",
        "target_model": args.target_model,
        "intervention_model": args.intervention_model,
        "judge_model": None,
        "runner": "huggingface",
        "evaluation_schedule": "trace_generation_then_intervention_generation_then_target_rerun",
        "main_pipeline_only": True,
        "judge_baselines_deferred": True,
        "load_in_4bit": not args.no_4bit,
        "target_quantization": args.target_quantization,
        "intervention_quantization": args.intervention_quantization,
        "require_gpu": not args.allow_cpu_offload,
        "min_visual_tokens": args.min_visual_tokens,
        "max_visual_tokens": args.max_visual_tokens,
        "trace_max_new_tokens": args.trace_max_new_tokens,
        "intervention_max_new_tokens": args.intervention_max_new_tokens,
        "target_max_new_tokens": args.target_max_new_tokens,
        "run_target_rerun": not args.skip_target_rerun,
        "target_prompt_mode": "trace_primary",
        "target_abstention_enabled": True,
        "intervention_generation_mode": "one_call_per_example",
        "mmmu_pro_science_subjects_only": not args.mmmu_pro_include_non_science,
        "run_c2faith_baseline": False,
        "run_facte_baseline": False,
        "include_retargeted_support": True,
        "include_modality_extension": False,
        "evaluation_chunk_size": args.evaluation_chunk_size,
        **dataset_config,
        "trace_schema": [
            "visual_evidence",
            "textual_evidence",
            "binding_claim",
            "scientific_principle",
            "inference",
            "conclusion",
            "final_answer",
        ],
        "interventions": [
            "original_trace",
            "paraphrase",
            "wrong_binding",
            "retargeted_support",
            "wrong_principle",
            "wrong_inference",
            "delete_binding_claim",
        ],
        "target_conditions": [
            "no_trace",
            "original_trace",
            "paraphrase",
            "wrong_binding",
            "retargeted_support",
            "wrong_principle",
            "wrong_inference",
            "delete_binding_claim",
            "no_image_with_original_trace",
            "text_only_no_trace",
        ],
    }
    target_runner = TargetRerunModel(
        model_id=args.target_model,
        load_in_4bit=not args.no_4bit,
        quantization=args.target_quantization,
        require_gpu=not args.allow_cpu_offload,
        min_visual_tokens=args.min_visual_tokens,
        max_visual_tokens=args.max_visual_tokens,
        trace_generation=HFGenerationConfig(max_new_tokens=args.trace_max_new_tokens),
        generation=HFGenerationConfig(max_new_tokens=args.target_max_new_tokens),
    )
    trace_generator = target_runner
    intervention_model = InterventionModel(
        model_id=args.intervention_model,
        load_in_4bit=not args.no_4bit,
        quantization=args.intervention_quantization,
        require_gpu=not args.allow_cpu_offload,
        generation=HFGenerationConfig(max_new_tokens=args.intervention_max_new_tokens),
    )
    target_evaluator = None if args.skip_target_rerun else target_runner
    summary = run_faithfulness_evaluation(
        examples=examples,
        output_dir=output_dir,
        config=config,
        trace_generator=trace_generator,
        judge=None,
        intervention_generator=intervention_model,
        target_evaluator=target_evaluator,
        run_target_rerun=not args.skip_target_rerun,
    )
    if args.planned_run and planned_state_path is not None:
        mark_planned_run_completed(planned_state_path, output_dir, examples)
    print(json.dumps(summary, indent=2))
    print(f"Wrote TraceFaith retargeted-support artifacts to {output_dir}")


def _explicit_batch_counts_provided(args: argparse.Namespace) -> bool:
    return args.scienceqa_count is not None or args.ai2d_count is not None or args.mmmu_pro_count is not None


def _resolved_counts(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.planned_run:
        default = args.planned_default_per_dataset
        return (
            args.scienceqa_count if args.scienceqa_count is not None else default,
            args.ai2d_count if args.ai2d_count is not None else default,
            args.mmmu_pro_count if args.mmmu_pro_count is not None else default,
        )
    return (
        args.scienceqa_count if args.scienceqa_count is not None else 10,
        args.ai2d_count if args.ai2d_count is not None else 10,
        args.mmmu_pro_count if args.mmmu_pro_count is not None else 0,
    )


def _read_excluded_example_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                example_id = payload.get("example_id")
                if example_id:
                    excluded.add(str(example_id))
    return excluded


def _sample_examples(
    examples: list[Example],
    scienceqa_count: int,
    ai2d_count: int,
    mmmu_pro_count: int,
    seed: int,
    excluded_ids: set[str],
) -> list[Example]:
    rng = random.Random(seed)
    available = [example for example in examples if example.example_id not in excluded_ids]
    scienceqa_examples = [example for example in available if example.dataset == "scienceqa"]
    ai2d_examples = [example for example in available if example.dataset == "ai2d"]
    mmmu_pro_examples = [example for example in available if example.dataset == "mmmu_pro"]
    if len(scienceqa_examples) < scienceqa_count:
        raise ValueError(
            f"Requested {scienceqa_count} ScienceQA examples after exclusions, found {len(scienceqa_examples)}"
        )
    if len(ai2d_examples) < ai2d_count:
        raise ValueError(f"Requested {ai2d_count} AI2D examples after exclusions, found {len(ai2d_examples)}")
    if len(mmmu_pro_examples) < mmmu_pro_count:
        raise ValueError(
            f"Requested {mmmu_pro_count} MMMU-Pro examples after exclusions, found {len(mmmu_pro_examples)}"
        )
    selected_scienceqa = rng.sample(scienceqa_examples, scienceqa_count)
    selected_ai2d = rng.sample(ai2d_examples, ai2d_count)
    selected_mmmu_pro = rng.sample(mmmu_pro_examples, mmmu_pro_count)
    return sorted(selected_scienceqa, key=lambda item: item.example_id) + sorted(
        selected_ai2d,
        key=lambda item: item.example_id,
    ) + sorted(selected_mmmu_pro, key=lambda item: item.example_id)


if __name__ == "__main__":
    main()
