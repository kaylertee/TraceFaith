from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aria.hf_models import (
    DEFAULT_JUDGE_MODEL_ID,
    DEFAULT_TARGET_MODEL_ID,
    HFGenerationConfig,
    JudgeModel,
    TargetRerunModel,
    TraceGeneratorModel,
)
from aria.loaders import load_scienceqa
from aria.schemas import BindingClaim, Evidence, Example, Modality, StructuredTrace, SupportClaim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TraceFaith Hugging Face model load/generation speed."
    )
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL_ID)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL_ID)
    parser.add_argument("--target-quantization", default="bnb4", choices=["auto", "prequantized", "bnb4", "none"])
    parser.add_argument("--judge-quantization", default="bnb4", choices=["auto", "prequantized", "bnb4", "none"])
    parser.add_argument("--target-dtype", default="auto")
    parser.add_argument("--judge-dtype", default="auto")
    parser.add_argument("--target-awq-backend", default="gemm")
    parser.add_argument("--judge-awq-backend", default="gemm")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--trace-max-new-tokens", type=int, default=256)
    parser.add_argument("--target-max-new-tokens", type=int, default=16)
    parser.add_argument("--judge-max-new-tokens", type=int, default=128)
    parser.add_argument("--min-visual-tokens", type=int, default=256)
    parser.add_argument("--max-visual-tokens", type=int, default=768)
    parser.add_argument("--bench-target", action="store_true", default=True)
    parser.add_argument("--no-bench-target", action="store_false", dest="bench_target")
    parser.add_argument("--bench-judge", action="store_true")
    parser.add_argument("--allow-cpu-offload", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="results/model_speed_benchmarks/awq_first",
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    example = _load_benchmark_example(args.split, args.example_index)
    config = {
        "target_model": args.target_model,
        "judge_model": args.judge_model,
        "target_quantization": args.target_quantization,
        "judge_quantization": args.judge_quantization,
        "target_dtype": args.target_dtype,
        "judge_dtype": args.judge_dtype,
        "target_awq_backend": args.target_awq_backend,
        "judge_awq_backend": args.judge_awq_backend,
        "split": args.split,
        "example_index": args.example_index,
        "example_id": example.example_id,
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "trace_max_new_tokens": args.trace_max_new_tokens,
        "target_max_new_tokens": args.target_max_new_tokens,
        "judge_max_new_tokens": args.judge_max_new_tokens,
        "min_visual_tokens": args.min_visual_tokens,
        "max_visual_tokens": args.max_visual_tokens,
        "require_gpu": not args.allow_cpu_offload,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    if args.bench_target:
        rows.extend(_benchmark_target(args, example))
    if args.bench_judge:
        rows.extend(_benchmark_judge(args, example))

    _write_rows(output_dir / "benchmark_rows.csv", rows)
    (output_dir / "benchmark_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(_summarize(rows), indent=2))


def _load_benchmark_example(split: str, example_index: int) -> Example:
    examples = load_scienceqa(
        max_examples=example_index + 1,
        split=split,
        require_image=True,
        require_explanation=True,
    )
    if len(examples) <= example_index:
        raise ValueError(f"Could not load ScienceQA example index {example_index} from split {split!r}.")
    return examples[example_index]


def _benchmark_target(args: argparse.Namespace, example: Example) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trace_model = TraceGeneratorModel(
        model_id=args.target_model,
        torch_dtype=args.target_dtype,
        quantization=args.target_quantization,
        awq_backend=args.target_awq_backend,
        require_gpu=not args.allow_cpu_offload,
        min_visual_tokens=args.min_visual_tokens,
        max_visual_tokens=args.max_visual_tokens,
        generation=HFGenerationConfig(max_new_tokens=args.trace_max_new_tokens),
    )
    load_seconds = _time_call(lambda: trace_model._load())
    rows.append(_row("target_trace_load", args.target_model, args.target_quantization, load_seconds))

    generated_trace: StructuredTrace | None = None
    for run_type, count in [("warmup", args.warmup_runs), ("timed", args.timed_runs)]:
        for index in range(count):
            seconds, trace = _time_call_with_result(lambda: trace_model.generate_trace(example))
            generated_trace = trace
            rows.append(
                _row(
                    "target_trace_generation",
                    args.target_model,
                    args.target_quantization,
                    seconds,
                    run_type=run_type,
                    run_index=index,
                    output_chars=len(trace.metadata.get("raw_output") or ""),
                    final_answer=trace.final_answer,
                )
            )
    trace_model.unload()

    target_model = TargetRerunModel(
        model_id=args.target_model,
        torch_dtype=args.target_dtype,
        quantization=args.target_quantization,
        awq_backend=args.target_awq_backend,
        require_gpu=not args.allow_cpu_offload,
        min_visual_tokens=args.min_visual_tokens,
        max_visual_tokens=args.max_visual_tokens,
        generation=HFGenerationConfig(max_new_tokens=args.target_max_new_tokens),
    )
    load_seconds = _time_call(lambda: target_model._load())
    rows.append(_row("target_rerun_load", args.target_model, args.target_quantization, load_seconds))
    trace_for_rerun = generated_trace or _fallback_trace(example)
    for run_type, count in [("warmup", args.warmup_runs), ("timed", args.timed_runs)]:
        for index in range(count):
            seconds, target_output = _time_call_with_result(
                lambda: target_model.evaluate("benchmark_original_trace", trace_for_rerun, example)
            )
            rows.append(
                _row(
                    "target_rerun",
                    args.target_model,
                    args.target_quantization,
                    seconds,
                    run_type=run_type,
                    run_index=index,
                    output_chars=len(target_output.raw_output or ""),
                    final_answer=target_output.final_answer,
                )
            )
    target_model.unload()
    return rows


def _benchmark_judge(args: argparse.Namespace, example: Example) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    judge = JudgeModel(
        model_id=args.judge_model,
        torch_dtype=args.judge_dtype,
        quantization=args.judge_quantization,
        awq_backend=args.judge_awq_backend,
        require_gpu=not args.allow_cpu_offload,
        generation=HFGenerationConfig(max_new_tokens=args.judge_max_new_tokens),
    )
    load_seconds = _time_call(lambda: judge._load())
    rows.append(_row("judge_load", args.judge_model, args.judge_quantization, load_seconds))
    trace = _fallback_trace(example)
    for run_type, count in [("warmup", args.warmup_runs), ("timed", args.timed_runs)]:
        for index in range(count):
            seconds, judge_output = _time_call_with_result(
                lambda: judge.judge("benchmark_original_trace", trace, example)
            )
            rows.append(
                _row(
                    "judge_generation",
                    args.judge_model,
                    args.judge_quantization,
                    seconds,
                    run_type=run_type,
                    run_index=index,
                    output_chars=len(judge_output.raw_output or ""),
                    final_answer=trace.final_answer,
                    faithfulness_score=judge_output.faithfulness_score,
                )
            )
    judge.unload()
    return rows


def _fallback_trace(example: Example) -> StructuredTrace:
    answer = example.correct_answer or "A"
    return StructuredTrace(
        visual_evidence=[
            Evidence(
                id="V1",
                modality=Modality.IMAGE,
                content="The image provides the visual evidence needed for the science question.",
                source_ref="image",
            )
        ],
        textual_evidence=[
            Evidence(
                id="T1",
                modality=Modality.TEXT,
                content=example.question,
                source_ref="question",
            )
        ],
        binding_claim=BindingClaim(
            id="B1",
            text=f"The evidence supports answer option {answer}.",
            supports=["V1", "T1"],
            uses_visual=["V1"],
            uses_textual=["T1"],
            bound_answer_option=answer,
        ),
        scientific_principle=SupportClaim(
            id="P1",
            text="Use the relevant scientific relationship shown by the question and image.",
            supports=["V1", "T1"],
        ),
        inference=SupportClaim(
            id="I1",
            text=f"Applying the evidence and principle points to option {answer}.",
            supports=["B1", "P1"],
        ),
        conclusion=f"The supported answer is {answer}.",
        final_answer=answer,
    )


def _time_call(fn: Any) -> float:
    seconds, _ = _time_call_with_result(fn)
    return seconds


def _time_call_with_result(fn: Any) -> tuple[float, Any]:
    _reset_cuda_peak()
    start = time.perf_counter()
    result = fn()
    _synchronize_cuda()
    return time.perf_counter() - start, result


def _row(
    task: str,
    model_id: str,
    quantization: str,
    seconds: float,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "task": task,
        "model_id": model_id,
        "quantization": quantization,
        "seconds": round(seconds, 4),
        "cuda_peak_allocated_gb": _cuda_peak_allocated_gb(),
        "cuda_peak_reserved_gb": _cuda_peak_reserved_gb(),
    }
    row.update(extra)
    return row


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timed = [row for row in rows if row.get("run_type") == "timed"]
    summary: dict[str, Any] = {"row_count": len(rows), "timed_row_count": len(timed)}
    for task in sorted({str(row["task"]) for row in timed}):
        values = [float(row["seconds"]) for row in timed if row["task"] == task]
        if values:
            summary[f"{task}_mean_seconds"] = round(sum(values) / len(values), 4)
            summary[f"{task}_min_seconds"] = round(min(values), 4)
            summary[f"{task}_max_seconds"] = round(max(values), 4)
    return summary


def _reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _cuda_peak_allocated_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / (1024**3), 4)
    except Exception:
        return None


def _cuda_peak_reserved_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_reserved() / (1024**3), 4)
    except Exception:
        return None


if __name__ == "__main__":
    main()
