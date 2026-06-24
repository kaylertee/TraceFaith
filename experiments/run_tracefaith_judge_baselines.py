from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aria.hf_models import DEFAULT_INTERVENTION_MODEL_ID, HFGenerationConfig, JudgeModel
from aria.io import read_jsonl, write_json, write_jsonl
from aria.metrics import judge_by_component, judge_by_intervention, judge_summary
from aria.schemas import Example, ExpectedLabel, IntervenedTrace, JudgeOutput, TargetOutput


TRACE_BEARING_CONDITIONS = {
    "original_trace",
    "paraphrase",
    "wrong_binding",
    "retargeted_support",
    "wrong_principle",
    "wrong_inference",
    "delete_binding_claim",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or prepare TraceFaith optional judge baselines for a completed result directory."
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default=DEFAULT_INTERVENTION_MODEL_ID)
    parser.add_argument("--judge-quantization", choices=["auto", "prequantized", "bnb4", "none"], default="bnb4")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--allow-cpu-offload", action="store_true")
    parser.add_argument("--judge-max-new-tokens", type=int, default=192)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-prefix",
        default="judge_baseline",
        help="Prefix for judge baseline artifacts in the result directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    examples = {item.example_id: item for item in read_jsonl(result_dir / "processed_examples.jsonl", Example)}
    records = [
        record
        for record in read_jsonl(result_dir / "intervened_traces.jsonl", IntervenedTrace)
        if record.intervention.intervention_type in TRACE_BEARING_CONDITIONS
    ]
    records = sorted(records, key=lambda record: (record.example_id, record.intervention_id))
    if args.limit is not None:
        records = records[: args.limit]
    target_outputs = _read_optional_jsonl(result_dir / "target_outputs.jsonl", TargetOutput)
    target_by_id = {output.intervention_id: output for output in target_outputs}

    inputs = [_baseline_input_row(record, examples[record.example_id], target_by_id.get(record.intervention_id)) for record in records]
    input_path = result_dir / f"{args.output_prefix}_inputs.jsonl"
    write_jsonl(input_path, inputs)

    config = {
        "baseline_family": "tracefaith_optional_judge_baselines",
        "result_dir": str(result_dir),
        "judge_model": args.judge_model,
        "judge_quantization": args.judge_quantization,
        "load_in_4bit": not args.no_4bit,
        "require_gpu": not args.allow_cpu_offload,
        "n_inputs": len(inputs),
        "trace_bearing_conditions": sorted(TRACE_BEARING_CONDITIONS),
        "baselines": [
            "c2faith_style_detection_localization",
            "facte_style_judge_score",
        ],
    }
    write_json(result_dir / f"{args.output_prefix}_config.json", config)

    if args.prepare_only:
        print(json.dumps({"prepared_inputs": len(inputs), "path": str(input_path)}, indent=2))
        return

    judge = JudgeModel(
        model_id=args.judge_model,
        load_in_4bit=not args.no_4bit,
        quantization=args.judge_quantization,
        require_gpu=not args.allow_cpu_offload,
        generation=HFGenerationConfig(max_new_tokens=args.judge_max_new_tokens),
    )
    outputs: list[JudgeOutput] = []
    output_path = result_dir / f"{args.output_prefix}_outputs.jsonl"
    for index, record in enumerate(records, start=1):
        output = judge.judge(record.intervention_id, record.intervened_trace, examples[record.example_id])
        output.metadata["baseline_family"] = "c2faith_style_detection_and_facte_style_score"
        output.metadata["intervention_type"] = record.intervention.intervention_type
        output.metadata["expected_flawed_component"] = record.intervention.expected_flawed_component.value
        outputs.append(output)
        write_jsonl(output_path, outputs)
        print(f"[{index}/{len(records)}] judged {record.intervention_id}", flush=True)
    judge.unload()

    artifacts = _write_baseline_metrics(
        result_dir=result_dir,
        output_prefix=args.output_prefix,
        records=records,
        outputs=outputs,
        target_by_id=target_by_id,
    )
    print(json.dumps({"judged": len(outputs), **artifacts}, indent=2))


def _baseline_input_row(record: IntervenedTrace, example: Example, target_output: TargetOutput | None) -> dict[str, Any]:
    return {
        "intervention_id": record.intervention_id,
        "example_id": record.example_id,
        "dataset": example.dataset,
        "intervention_type_hidden_from_judge": True,
        "expected_label_for_metrics_only": record.intervention.expected_label.value,
        "expected_flawed_component_for_metrics_only": record.intervention.expected_flawed_component.value,
        "question": example.question,
        "choices": example.choices,
        "correct_answer": example.correct_answer,
        "target_answer_for_metrics_only": target_output.final_answer if target_output else None,
        "target_correct_for_metrics_only": target_output.is_correct if target_output else None,
    }


def _write_baseline_metrics(
    result_dir: Path,
    output_prefix: str,
    records: list[IntervenedTrace],
    outputs: list[JudgeOutput],
    target_by_id: dict[str, TargetOutput],
) -> dict[str, str]:
    judged_ids = {output.intervention_id for output in outputs}
    judged_records = [record for record in records if record.intervention_id in judged_ids]
    summary = judge_summary(judged_records, outputs)
    rows = _joined_rows(judged_records, outputs, target_by_id)
    detection = pd.DataFrame([summary | _behavior_correlation_summary(rows)])
    by_intervention = judge_by_intervention(judged_records, outputs)
    by_component = judge_by_component(judged_records, outputs)
    score = pd.DataFrame([_facte_style_score_summary(rows)])

    detection_path = result_dir / f"{output_prefix}_detection_localization_metrics.csv"
    intervention_path = result_dir / f"{output_prefix}_by_intervention.csv"
    component_path = result_dir / f"{output_prefix}_by_component.csv"
    score_path = result_dir / f"{output_prefix}_score_metrics.csv"
    rows_path = result_dir / f"{output_prefix}_joined_rows.csv"
    detection.to_csv(detection_path, index=False)
    by_intervention.to_csv(intervention_path, index=False)
    by_component.to_csv(component_path, index=False)
    score.to_csv(score_path, index=False)
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    return {
        "detection_metrics": str(detection_path),
        "score_metrics": str(score_path),
        "joined_rows": str(rows_path),
    }


def _joined_rows(
    records: list[IntervenedTrace],
    outputs: list[JudgeOutput],
    target_by_id: dict[str, TargetOutput],
) -> list[dict[str, Any]]:
    by_output = {output.intervention_id: output for output in outputs}
    original_score_by_example: dict[str, int] = {}
    original_correct_by_example: dict[str, bool | None] = {}
    original_answer_by_example: dict[str, str | None] = {}
    for record in records:
        if record.intervention.intervention_type != "original_trace":
            continue
        output = by_output.get(record.intervention_id)
        target = target_by_id.get(record.intervention_id)
        if output:
            original_score_by_example[record.example_id] = output.faithfulness_score
        if target:
            original_correct_by_example[record.example_id] = target.is_correct
            original_answer_by_example[record.example_id] = target.final_answer

    rows = []
    for record in records:
        output = by_output.get(record.intervention_id)
        target = target_by_id.get(record.intervention_id)
        if output is None:
            continue
        original_answer = original_answer_by_example.get(record.example_id)
        answer_flipped = (
            bool(target)
            and original_answer is not None
            and target.final_answer.strip().upper() != original_answer.strip().upper()
        )
        original_correct = original_correct_by_example.get(record.example_id)
        correctness_drop = (
            int(bool(original_correct)) - int(bool(target.is_correct))
            if target and original_correct is not None
            else 0
        )
        score_drop = original_score_by_example.get(record.example_id, output.faithfulness_score) - output.faithfulness_score
        rows.append(
            {
                "intervention_id": record.intervention_id,
                "example_id": record.example_id,
                "intervention_type": record.intervention.intervention_type,
                "expected_label": record.intervention.expected_label.value,
                "expected_flawed_component": record.intervention.expected_flawed_component.value,
                "judge_flagged": output.is_intervened_or_flawed,
                "judge_component": output.flawed_component.value,
                "judge_score": output.faithfulness_score,
                "judge_severity": 5 - output.faithfulness_score,
                "judge_score_drop_from_original": score_drop,
                "target_answer_flipped": float(answer_flipped),
                "target_correctness_drop": float(correctness_drop),
                "target_is_correct": target.is_correct if target else None,
                "target_answer": target.final_answer if target else None,
            }
        )
    return rows


def _behavior_correlation_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "judge_severity_vs_answer_flip_spearman": _spearman(
            [row["judge_severity"] for row in rows],
            [row["target_answer_flipped"] for row in rows],
        ),
        "judge_severity_vs_correctness_drop_spearman": _spearman(
            [row["judge_severity"] for row in rows],
            [row["target_correctness_drop"] for row in rows],
        ),
    }


def _facte_style_score_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    flip_rows = [row for row in rows if row["expected_label"] == ExpectedLabel.INTERVENED.value]
    return {
        "n_scored": len(rows),
        "n_intervened_scored": len(flip_rows),
        "auroc_judge_score_drop_predicts_answer_flip": _auroc(
            [row["judge_score_drop_from_original"] for row in flip_rows],
            [row["target_answer_flipped"] for row in flip_rows],
        ),
        "spearman_score_drop_vs_correctness_drop": _spearman(
            [row["judge_score_drop_from_original"] for row in flip_rows],
            [row["target_correctness_drop"] for row in flip_rows],
        ),
        "ranking_agreement_score_drop_vs_target_effect": _spearman(
            _mean_by_intervention(flip_rows, "judge_score_drop_from_original"),
            _mean_by_intervention(flip_rows, "target_answer_flipped"),
        ),
    }


def _mean_by_intervention(rows: list[dict[str, Any]], key: str) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["intervention_type"]].append(float(row[key]))
    return [mean(values) for _, values in sorted(grouped.items())]


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return _pearson(_ranks(xs), _ranks(ys))


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return numerator / (dx * dy) if dx and dy else 0.0


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = rank
        i = j + 1
    return ranks


def _auroc(scores: list[float], labels: list[float]) -> float:
    pairs = [(float(score), int(label)) for score, label in zip(scores, labels, strict=True)]
    positives = [score for score, label in pairs if label == 1]
    negatives = [score for score, label in pairs if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def _read_optional_jsonl(path: Path, model: type[Any]) -> list[Any]:
    if not path.exists():
        return []
    return read_jsonl(path, model)


if __name__ == "__main__":
    main()
