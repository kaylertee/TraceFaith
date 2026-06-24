from __future__ import annotations

import argparse
import json
from pathlib import Path

from aria.hf_models import (
    DEFAULT_JUDGE_MODEL_ID,
    DEFAULT_TARGET_MODEL_ID,
    HFGenerationConfig,
    JudgeModel,
    TargetRerunModel,
)
from aria.loaders import load_mixed_scienceqa_ai2d
from aria.pipeline import run_faithfulness_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TraceFaith with modality-reliance extension conditions.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "tracefaith_modality_scienceqa_ai2d_20q",
    )
    parser.add_argument("--scienceqa-count", type=int, default=10)
    parser.add_argument("--ai2d-count", type=int, default=10)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL_ID)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL_ID)
    parser.add_argument("--scienceqa-split", default="validation")
    parser.add_argument("--ai2d-split", default=None)
    parser.add_argument("--ai2d-local-jsonl", type=Path, default=None)
    parser.add_argument(
        "--scienceqa-image-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "scienceqa_images",
    )
    parser.add_argument(
        "--ai2d-image-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "ai2d_images",
    )
    parser.add_argument("--require-explanation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scienceqa-diagram-only", action="store_true")
    parser.add_argument("--skip-target-rerun", action="store_true")
    parser.add_argument("--evaluation-chunk-size", type=int, default=1)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--target-quantization", choices=["auto", "prequantized", "bnb4", "none"], default="bnb4")
    parser.add_argument("--judge-quantization", choices=["auto", "prequantized", "bnb4", "none"], default="bnb4")
    parser.add_argument("--allow-cpu-offload", action="store_true")
    parser.add_argument("--min-visual-tokens", type=int, default=256)
    parser.add_argument("--max-visual-tokens", type=int, default=768)
    parser.add_argument("--trace-max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-max-new-tokens", type=int, default=192)
    parser.add_argument("--target-max-new-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples, dataset_config = load_mixed_scienceqa_ai2d(
        scienceqa_count=args.scienceqa_count,
        ai2d_count=args.ai2d_count,
        scienceqa_split=args.scienceqa_split,
        ai2d_split=args.ai2d_split,
        scienceqa_image_dir=args.scienceqa_image_dir / args.scienceqa_split,
        ai2d_image_dir=args.ai2d_image_dir / (args.ai2d_split or "auto"),
        ai2d_local_jsonl_path=args.ai2d_local_jsonl,
        require_explanation=args.require_explanation,
        diagram_only=args.scienceqa_diagram_only,
    )
    config = {
        "experiment": "tracefaith",
        "experiment_stage": "modality_scienceqa_ai2d_20q",
        "dataset": "scienceqa+ai2d",
        "target_model": args.target_model,
        "judge_model": args.judge_model,
        "runner": "huggingface",
        "evaluation_schedule": "trace_and_target_rerun_single_target_load_then_judge",
        "load_in_4bit": not args.no_4bit,
        "target_quantization": args.target_quantization,
        "judge_quantization": args.judge_quantization,
        "require_gpu": not args.allow_cpu_offload,
        "min_visual_tokens": args.min_visual_tokens,
        "max_visual_tokens": args.max_visual_tokens,
        "trace_max_new_tokens": args.trace_max_new_tokens,
        "judge_max_new_tokens": args.judge_max_new_tokens,
        "target_max_new_tokens": args.target_max_new_tokens,
        "run_target_rerun": not args.skip_target_rerun,
        "target_prompt_mode": "trace_primary",
        "include_modality_extension": True,
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
            "wrong_principle",
            "wrong_inference",
            "delete_binding_claim",
            "wrong_visual",
            "wrong_textual",
        ],
        "target_conditions": [
            "no_trace",
            "original_trace",
            "paraphrase",
            "wrong_binding",
            "wrong_principle",
            "wrong_inference",
            "delete_binding_claim",
            "wrong_visual",
            "wrong_textual",
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
    judge = JudgeModel(
        model_id=args.judge_model,
        load_in_4bit=not args.no_4bit,
        quantization=args.judge_quantization,
        require_gpu=not args.allow_cpu_offload,
        generation=HFGenerationConfig(max_new_tokens=args.judge_max_new_tokens),
    )
    target_evaluator = None if args.skip_target_rerun else target_runner
    summary = run_faithfulness_evaluation(
        examples=examples,
        output_dir=args.output_dir,
        config=config,
        trace_generator=trace_generator,
        judge=judge,
        target_evaluator=target_evaluator,
        run_target_rerun=not args.skip_target_rerun,
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote TraceFaith modality artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
