from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aria.interventions import (
    build_model_interventions_from_output,
    template_interventions_for_example,
    validation_rows,
)
from aria.io import read_jsonl, write_json, write_json_atomic, write_jsonl
from aria.metrics import (
    judge_by_component,
    judge_by_intervention,
    judge_summary,
    mediation_summary,
    modality_effect_rows,
    modality_summary,
    target_effect_rows,
)
from aria.schemas import (
    Example,
    GeneratedTraceRecord,
    InterventionModelOutput,
    IntervenedTrace,
    JudgeOutput,
    StructuredTrace,
    TargetOutput,
)


def run_faithfulness_evaluation(
    examples: list[Example],
    output_dir: Path,
    config: dict[str, Any],
    trace_generator: Any,
    judge: Any | None = None,
    intervention_generator: Any | None = None,
    target_evaluator: Any | None = None,
    run_target_rerun: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _initialize_incremental_artifacts(output_dir, config, examples)

    generated_trace_records: list[GeneratedTraceRecord] = []
    for example in examples:
        record = _read_trace_checkpoint(output_dir, example.example_id)
        if record is None:
            try:
                record = GeneratedTraceRecord(
                    example_id=example.example_id,
                    trace=trace_generator.generate_trace(example),
                    target_model=config.get("target_model") or trace_generator.model_name,
                    metadata={"dataset": example.dataset},
                )
            except Exception as exc:
                _write_trace_failure_checkpoint(
                    output_dir,
                    example.example_id,
                    {
                        "example_id": example.example_id,
                        "dataset": example.dataset,
                        "target_model": config.get("target_model") or getattr(trace_generator, "model_name", "unknown"),
                        "parse_error": str(exc),
                    },
                )
                raise
            _write_trace_checkpoint(output_dir, record)
        generated_trace_records.append(record)
        write_jsonl(output_dir / "original_traces.jsonl", generated_trace_records)
        _write_run_status(
            output_dir,
            status="running",
            stage="trace_generation",
            n_examples=len(examples),
            n_original_traces=len(generated_trace_records),
            n_completed_examples=0,
            pending_example_ids=[item.example_id for item in examples],
        )
    if intervention_generator is not None or trace_generator is not target_evaluator:
        _unload_runner(trace_generator)
    traces_by_example: dict[str, StructuredTrace] = {
        record.example_id: record.trace for record in generated_trace_records
    }
    write_jsonl(output_dir / "original_traces.jsonl", generated_trace_records)
    answer_choices_by_example = {example.example_id: example.choices for example in examples}
    reference_answers_by_example = {example.example_id: example.correct_answer for example in examples}
    example_metadata_by_id = {
        example.example_id: {"dataset": example.dataset, **example.metadata}
        for example in examples
    }
    examples_by_id = {example.example_id: example for example in examples}

    if run_target_rerun:
        if target_evaluator is None:
            raise ValueError("target_evaluator is required when run_target_rerun=True")

    checkpoint = _load_completed_checkpoint(
        output_dir=output_dir,
        run_target_rerun=run_target_rerun,
    )
    completed_example_ids = checkpoint["completed_example_ids"]
    intervened_traces: list[IntervenedTrace] = checkpoint["intervened_traces"]
    target_records: list[IntervenedTrace] = checkpoint["target_records"]
    judge_outputs: list[JudgeOutput] = checkpoint["judge_outputs"]
    target_outputs: list[TargetOutput] = checkpoint["target_outputs"]
    intervention_model_outputs: list[InterventionModelOutput] = checkpoint["intervention_model_outputs"]

    completed_examples = len(completed_example_ids)
    last_completed_example_id: str | None = None
    _write_run_status(
        output_dir,
        status="running",
        stage="evaluation",
        n_examples=len(examples),
        n_original_traces=len(generated_trace_records),
        n_completed_examples=completed_examples,
        pending_example_ids=[example.example_id for example in examples if example.example_id not in completed_example_ids],
    )
    pending_examples = [example for example in examples if example.example_id not in completed_example_ids]
    pending_payloads: dict[str, dict[str, Any]] = {}
    _write_run_status(
        output_dir,
        status="running",
        stage="intervention_generation" if intervention_generator is not None else "intervention_preparation",
        n_examples=len(examples),
        n_original_traces=len(generated_trace_records),
        n_completed_examples=completed_examples,
        last_completed_example_id=last_completed_example_id,
        pending_example_ids=[item.example_id for item in examples if item.example_id not in completed_example_ids],
    )
    for example in pending_examples:
        payload = _prepare_example_records(
            example=example,
            output_dir=output_dir,
            traces_by_example=traces_by_example,
            answer_choices_by_example=answer_choices_by_example,
            reference_answers_by_example=reference_answers_by_example,
            example_metadata_by_id=example_metadata_by_id,
            intervention_generator=intervention_generator,
            include_modality_interventions=bool(config.get("include_modality_extension")),
            include_retargeted_support=bool(config.get("include_retargeted_support")),
        )
        pending_payloads[example.example_id] = payload
        if payload.get("intervention_model_output") is not None:
            intervention_model_outputs.append(payload["intervention_model_output"])
            _write_intervention_model_outputs(output_dir / "intervention_model_outputs.jsonl", intervention_model_outputs)
        _write_run_status(
            output_dir,
            status="running",
            stage="intervention_generation" if intervention_generator is not None else "intervention_preparation",
            n_examples=len(examples),
            n_original_traces=len(generated_trace_records),
            n_completed_examples=completed_examples,
            last_completed_example_id=last_completed_example_id,
            pending_example_ids=[item.example_id for item in examples if item.example_id not in completed_example_ids],
        )
    if intervention_generator is not None:
        _unload_runner(intervention_generator)
    target_outputs_by_example: dict[str, list[TargetOutput]] = {example.example_id: [] for example in pending_examples}

    if run_target_rerun:
        _write_run_status(
            output_dir,
            status="running",
            stage="target_rerun",
            n_examples=len(examples),
            n_original_traces=len(generated_trace_records),
            n_completed_examples=completed_examples,
            last_completed_example_id=last_completed_example_id,
            pending_example_ids=[item.example_id for item in examples if item.example_id not in completed_example_ids],
        )
        if judge is not None:
            write_jsonl(output_dir / "judge_outputs.jsonl", judge_outputs)
        staged_target_records = list(target_records)
        staged_target_outputs = list(target_outputs)
        for example in pending_examples:
            payload = pending_payloads[example.example_id]
            staged_outputs = _read_target_stage_checkpoint(
                output_dir,
                example.example_id,
                [record.intervention_id for record in payload["target_records"]],
            )
            if staged_outputs is None:
                staged_outputs = [
                    _evaluate_target_record(record, target_evaluator, examples_by_id)
                    for record in payload["target_records"]
                ]
                _write_target_stage_checkpoint(
                    output_dir=output_dir,
                    example=example,
                    target_records=payload["target_records"],
                    validation=payload["validation"],
                    target_outputs=staged_outputs,
                )
            target_outputs_by_example[example.example_id] = staged_outputs
            staged_target_records.extend(payload["target_records"])
            staged_target_outputs.extend(staged_outputs)
            write_jsonl(output_dir / "intervened_traces.jsonl", staged_target_records)
            write_jsonl(output_dir / "target_outputs.jsonl", staged_target_outputs)
            _write_run_status(
                output_dir,
                status="running",
                stage="target_rerun",
                n_examples=len(examples),
                n_original_traces=len(generated_trace_records),
                n_completed_examples=completed_examples,
                last_completed_example_id=last_completed_example_id,
                pending_example_ids=[item.example_id for item in examples if item.example_id not in completed_example_ids],
            )
        _unload_runner(target_evaluator)

    stage_name = "baseline_judge_evaluation" if judge is not None else "example_checkpointing"
    _write_run_status(
        output_dir,
        status="running",
        stage=stage_name,
        n_examples=len(examples),
        n_original_traces=len(generated_trace_records),
        n_completed_examples=completed_examples,
        last_completed_example_id=last_completed_example_id,
        pending_example_ids=[item.example_id for item in examples if item.example_id not in completed_example_ids],
    )
    for example in pending_examples:
        payload = pending_payloads[example.example_id]
        example_interventions = payload["interventions"]
        example_target_records = payload["target_records"]
        example_validation = payload["validation"]
        example_judge_outputs = []
        if judge is not None:
            example_judge_outputs = [
                judge.judge(
                    record.intervention_id,
                    record.intervened_trace,
                    examples_by_id[record.example_id],
                )
                for record in example_interventions
            ]
        example_target_outputs = target_outputs_by_example[example.example_id]

        intervened_traces.extend(example_interventions)
        target_records.extend(example_target_records)
        judge_outputs.extend(example_judge_outputs)
        target_outputs.extend(example_target_outputs)

        _write_example_checkpoint(
            output_dir=output_dir,
            example=example,
            target_records=example_target_records,
            validation=example_validation,
            judge_outputs=example_judge_outputs,
            target_outputs=example_target_outputs,
            intervention_model_output=payload.get("intervention_model_output"),
        )
        _save_partial_aggregate_artifacts(
            output_dir=output_dir,
            examples=examples,
            generated_trace_records=generated_trace_records,
            target_records=target_records,
            judge_outputs=judge_outputs,
            target_outputs=target_outputs,
            intervention_model_outputs=intervention_model_outputs,
        )
        completed_example_ids.add(example.example_id)
        completed_examples += 1
        _write_run_status(
            output_dir,
            status="running",
            stage="example_complete",
            n_examples=len(examples),
            n_original_traces=len(generated_trace_records),
            n_completed_examples=completed_examples,
            last_completed_example_id=example.example_id,
            pending_example_ids=[item.example_id for item in examples if item.example_id not in completed_example_ids],
        )
        last_completed_example_id = example.example_id

    if judge is not None:
        _unload_runner(judge)
    if run_target_rerun and target_evaluator is not None and target_evaluator is not trace_generator:
        _unload_runner(target_evaluator)

    summary = judge_summary(intervened_traces, judge_outputs) if judge_outputs else {}
    if target_outputs:
        summary.update(mediation_summary(target_records, target_outputs, judge_outputs))
        if config.get("include_modality_extension"):
            effect_rows = target_effect_rows(target_records, target_outputs, judge_outputs)
            summary.update(modality_summary(effect_rows))
    summary.update(
        {
            "target_model": config.get("target_model"),
            "intervention_model": config.get("intervention_model"),
            "judge_model": config.get("judge_model"),
            "dataset": config.get("dataset"),
            "n_examples": len(examples),
        }
    )

    _save_final_artifacts(
        output_dir=output_dir,
        examples=examples,
        generated_trace_records=generated_trace_records,
        target_records=target_records,
        validation=[],
        judge_outputs=judge_outputs,
        target_outputs=target_outputs,
        intervention_model_outputs=intervention_model_outputs,
        summary=summary,
    )
    _write_run_status(
        output_dir,
        status="completed",
        stage="completed",
        n_examples=len(examples),
        n_original_traces=len(generated_trace_records),
        n_completed_examples=completed_examples,
        pending_example_ids=[],
    )
    return summary


def _evaluate_target_record(
    record: IntervenedTrace,
    target_evaluator: Any,
    examples_by_id: dict[str, Example],
) -> TargetOutput:
    trace_for_target = None if record.intervention.details.get("trace_mode") == "none" else record.intervened_trace
    include_image = bool(record.intervention.details.get("include_image", True))
    condition = record.intervention.details.get("condition", record.intervention.intervention_type)
    output = target_evaluator.evaluate(
        record.intervention_id,
        trace_for_target,
        examples_by_id[record.example_id],
        condition=condition,
        include_image=include_image,
    )
    output.metadata["expected_label"] = record.intervention.expected_label.value
    output.metadata["intervention_type"] = record.intervention.intervention_type
    output.metadata["condition"] = condition
    return output


def _prepare_example_records(
    example: Example,
    output_dir: Path,
    traces_by_example: dict[str, StructuredTrace],
    answer_choices_by_example: dict[str, list[Any]],
    reference_answers_by_example: dict[str, str],
    example_metadata_by_id: dict[str, dict[str, Any]],
    intervention_generator: Any | None = None,
    include_modality_interventions: bool = False,
    include_retargeted_support: bool = False,
) -> dict[str, Any]:
    template_records = template_interventions_for_example(
        example_id=example.example_id,
        traces_by_example=traces_by_example,
        answer_choices_by_example=answer_choices_by_example,
        reference_answers_by_example=reference_answers_by_example,
        example_metadata_by_id=example_metadata_by_id,
        include_modality_interventions=include_modality_interventions,
        include_retargeted_support=include_retargeted_support,
    )
    intervention_model_output: InterventionModelOutput | None = None
    if intervention_generator is not None:
        intervention_model_output = _read_intervention_checkpoint(output_dir, example.example_id)
        if intervention_model_output is None:
            intervention_model_output = intervention_generator.generate_interventions(
                example,
                traces_by_example[example.example_id],
                template_records,
            )
            try:
                model_records = build_model_interventions_from_output(intervention_model_output, template_records)
            except Exception:
                _write_intervention_failure_checkpoint(output_dir, intervention_model_output)
                raise
            intervention_model_output.interventions = model_records
            _write_intervention_checkpoint(output_dir, intervention_model_output)
        example_interventions = build_model_interventions_from_output(intervention_model_output, template_records)
        intervention_model_output.interventions = example_interventions
    else:
        example_interventions = template_records
    example_validation = validation_rows(example_interventions, answer_choices_by_example)
    example_target_records = [
        *example_interventions,
        *_shortcut_control_records(traces_by_example, [example]),
    ]
    _attach_example_metadata(example_interventions, example)
    _attach_example_metadata(example_target_records, example)
    return {
        "interventions": example_interventions,
        "target_records": example_target_records,
        "validation": example_validation,
        "intervention_model_output": intervention_model_output,
    }


def _initialize_incremental_artifacts(
    output_dir: Path,
    config: dict[str, Any],
    examples: list[Example],
) -> None:
    write_json(output_dir / "config.json", config)
    write_jsonl(output_dir / "selected_examples.jsonl", examples)
    write_jsonl(output_dir / "processed_examples.jsonl", examples)
    (output_dir / "checkpoints" / "traces").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints" / "interventions").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints" / "examples").mkdir(parents=True, exist_ok=True)
    _write_run_status(
        output_dir,
        status="running",
        stage="initialized",
        n_examples=len(examples),
        n_original_traces=0,
        n_completed_examples=0,
        pending_example_ids=[example.example_id for example in examples],
    )


def _save_final_artifacts(
    output_dir: Path,
    examples: list[Example],
    generated_trace_records: list[GeneratedTraceRecord],
    target_records: list[IntervenedTrace],
    validation: list[Any],
    judge_outputs: list[JudgeOutput],
    target_outputs: list[TargetOutput],
    intervention_model_outputs: list[InterventionModelOutput],
    summary: dict[str, Any],
) -> None:
    validation_rows_payload = validation or _validation_rows_from_completed_checkpoints(output_dir, examples)
    write_jsonl(output_dir / "original_traces.jsonl", generated_trace_records)
    write_jsonl(output_dir / "intervened_traces.jsonl", target_records)
    if intervention_model_outputs:
        _write_intervention_model_outputs(output_dir / "intervention_model_outputs.jsonl", intervention_model_outputs)
    if judge_outputs:
        write_jsonl(output_dir / "judge_outputs.jsonl", judge_outputs)
    write_jsonl(output_dir / "target_outputs.jsonl", target_outputs)
    pd.DataFrame(validation_rows_payload).to_csv(output_dir / "intervention_validation.csv", index=False)
    if target_outputs:
        effect_rows = target_effect_rows(target_records, target_outputs, judge_outputs)
        pd.DataFrame(effect_rows).to_csv(output_dir / "target_effects.csv", index=False)
        pd.DataFrame(
            [
                row
                for row in effect_rows
                if row["intervention_type"] in {"no_trace", "text_only_no_trace", "no_image_with_original_trace"}
            ]
        ).to_csv(output_dir / "shortcut_controls.csv", index=False)
        pd.DataFrame([mediation_summary(target_records, target_outputs, judge_outputs)]).to_csv(
            output_dir / "mediation_metrics.csv",
            index=False,
        )
        if summary.get("n_modality_examples") is not None:
            modality_rows = modality_effect_rows(effect_rows)
            pd.DataFrame(modality_rows).to_csv(output_dir / "modality_effects.csv", index=False)
            pd.DataFrame([modality_summary(effect_rows)]).to_csv(output_dir / "modality_summary.csv", index=False)

    pd.DataFrame([summary]).to_csv(output_dir / "metrics_summary.csv", index=False)
    if judge_outputs:
        judged_ids = {output.intervention_id for output in judge_outputs}
        judged_records = [record for record in target_records if record.intervention_id in judged_ids]
        judge_by_intervention(judged_records, judge_outputs).to_csv(
            output_dir / "metrics_by_intervention.csv",
            index=False,
        )
        judge_by_component(judged_records, judge_outputs).to_csv(
            output_dir / "metrics_by_component.csv",
            index=False,
        )


def _save_partial_aggregate_artifacts(
    output_dir: Path,
    examples: list[Example],
    generated_trace_records: list[GeneratedTraceRecord],
    target_records: list[IntervenedTrace],
    judge_outputs: list[JudgeOutput],
    target_outputs: list[TargetOutput],
    intervention_model_outputs: list[InterventionModelOutput],
) -> None:
    validation_rows_payload = _validation_rows_from_completed_checkpoints(output_dir, examples)
    write_jsonl(output_dir / "original_traces.jsonl", generated_trace_records)
    write_jsonl(output_dir / "intervened_traces.jsonl", target_records)
    if intervention_model_outputs:
        _write_intervention_model_outputs(output_dir / "intervention_model_outputs.jsonl", intervention_model_outputs)
    if judge_outputs:
        write_jsonl(output_dir / "judge_outputs.jsonl", judge_outputs)
    write_jsonl(output_dir / "target_outputs.jsonl", target_outputs)
    pd.DataFrame(validation_rows_payload).to_csv(output_dir / "intervention_validation.csv", index=False)
    if target_outputs:
        effect_rows = target_effect_rows(target_records, target_outputs, judge_outputs)
        pd.DataFrame(effect_rows).to_csv(output_dir / "target_effects.csv", index=False)
        pd.DataFrame(
            [
                row
                for row in effect_rows
                if row["intervention_type"] in {"no_trace", "text_only_no_trace", "no_image_with_original_trace"}
            ]
        ).to_csv(output_dir / "shortcut_controls.csv", index=False)
        if any(row["intervention_type"] in {"wrong_visual", "wrong_textual"} for row in effect_rows):
            pd.DataFrame(modality_effect_rows(effect_rows)).to_csv(output_dir / "modality_effects.csv", index=False)
            pd.DataFrame([modality_summary(effect_rows)]).to_csv(output_dir / "modality_summary.csv", index=False)


def _load_completed_checkpoint(
    output_dir: Path,
    run_target_rerun: bool,
) -> dict[str, Any]:
    completed_example_ids: set[str] = set()
    completed_target_records: list[IntervenedTrace] = []
    completed_judge_outputs: list[JudgeOutput] = []
    completed_target_outputs: list[TargetOutput] = []
    completed_intervention_model_outputs: list[InterventionModelOutput] = []

    for completed_path in sorted((output_dir / "checkpoints" / "examples").glob("*/completed.json")):
        try:
            payload = json.loads(completed_path.read_text(encoding="utf-8"))
            example_id = str(payload["example_id"])
            checkpoint_dir = completed_path.parent
            target_records = read_jsonl(checkpoint_dir / "intervened_traces.jsonl", IntervenedTrace)
            judge_outputs = _read_jsonl_if_exists(checkpoint_dir / "judge_outputs.jsonl", JudgeOutput)
            target_outputs = read_jsonl(checkpoint_dir / "target_outputs.jsonl", TargetOutput)
            intervention_model_outputs = _read_jsonl_if_exists(
                checkpoint_dir / "intervention_model_outputs.jsonl",
                InterventionModelOutput,
            )
            if run_target_rerun and not target_outputs:
                continue
        except Exception:
            continue
        completed_example_ids.add(example_id)
        completed_target_records.extend(target_records)
        completed_judge_outputs.extend(judge_outputs)
        completed_target_outputs.extend(target_outputs)
        completed_intervention_model_outputs.extend(intervention_model_outputs)
    return {
        "completed_example_ids": completed_example_ids,
        "intervened_traces": [
            record
            for record in completed_target_records
            if record.intervention.intervention_type not in {"no_trace", "text_only_no_trace", "no_image_with_original_trace"}
        ],
        "target_records": completed_target_records,
        "judge_outputs": completed_judge_outputs,
        "target_outputs": completed_target_outputs,
        "intervention_model_outputs": completed_intervention_model_outputs,
    }


def _write_example_checkpoint(
    output_dir: Path,
    example: Example,
    target_records: list[IntervenedTrace],
    validation: list[Any],
    judge_outputs: list[JudgeOutput],
    target_outputs: list[TargetOutput],
    intervention_model_output: InterventionModelOutput | None = None,
) -> None:
    checkpoint_dir = output_dir / "checkpoints" / "examples" / _safe_id(example.example_id)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed_path = checkpoint_dir / "completed.json"
    if completed_path.exists():
        completed_path.unlink()
    write_jsonl(checkpoint_dir / "intervened_traces.jsonl", target_records)
    if judge_outputs:
        write_jsonl(checkpoint_dir / "judge_outputs.jsonl", judge_outputs)
    if intervention_model_output is not None:
        _write_intervention_model_outputs(checkpoint_dir / "intervention_model_outputs.jsonl", [intervention_model_output])
    write_jsonl(checkpoint_dir / "target_outputs.jsonl", target_outputs)
    write_jsonl(checkpoint_dir / "intervention_validation.jsonl", [row.model_dump(mode="json") for row in validation])
    write_json_atomic(
        completed_path,
        {
            "example_id": example.example_id,
            "dataset": example.dataset,
            "n_intervened_traces": len(target_records),
            "n_judge_outputs": len(judge_outputs),
            "n_target_outputs": len(target_outputs),
        },
    )


def _read_target_stage_checkpoint(
    output_dir: Path,
    example_id: str,
    expected_intervention_ids: list[str],
) -> list[TargetOutput] | None:
    checkpoint_dir = output_dir / "checkpoints" / "examples" / _safe_id(example_id)
    ready_path = checkpoint_dir / "target_stage.json"
    outputs_path = checkpoint_dir / "target_outputs.jsonl"
    records_path = checkpoint_dir / "intervened_traces.jsonl"
    if not ready_path.exists() or not outputs_path.exists() or not records_path.exists():
        return None
    try:
        payload = json.loads(ready_path.read_text(encoding="utf-8"))
        outputs = read_jsonl(outputs_path, TargetOutput)
        expected_outputs = int(payload.get("n_target_outputs", -1))
        if len(outputs) != expected_outputs:
            return None
        if [output.intervention_id for output in outputs] != expected_intervention_ids:
            return None
        return outputs
    except Exception:
        return None


def _write_target_stage_checkpoint(
    output_dir: Path,
    example: Example,
    target_records: list[IntervenedTrace],
    validation: list[Any],
    target_outputs: list[TargetOutput],
) -> None:
    checkpoint_dir = output_dir / "checkpoints" / "examples" / _safe_id(example.example_id)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ready_path = checkpoint_dir / "target_stage.json"
    if ready_path.exists():
        ready_path.unlink()
    write_jsonl(checkpoint_dir / "intervened_traces.jsonl", target_records)
    write_jsonl(checkpoint_dir / "target_outputs.jsonl", target_outputs)
    write_jsonl(checkpoint_dir / "intervention_validation.jsonl", [row.model_dump(mode="json") for row in validation])
    write_json_atomic(
        ready_path,
        {
            "example_id": example.example_id,
            "dataset": example.dataset,
            "n_intervened_traces": len(target_records),
            "n_target_outputs": len(target_outputs),
        },
    )


def _write_run_status(output_dir: Path, **payload: Any) -> None:
    n_examples = int(payload.get("n_examples", 0) or 0)
    n_completed_examples = int(payload.pop("n_completed_examples", 0) or 0)
    payload.pop("n_original_traces", None)
    payload.update(
        {
            "completed_original_traces": _count_trace_checkpoints(output_dir),
            "completed_intervened_traces": _count_intervention_checkpoints(output_dir),
            "completed_target_reruns": _count_target_stage_checkpoints(output_dir),
            "fully_completed_examples": n_completed_examples,
        }
    )
    if n_examples:
        payload.setdefault("total_examples", n_examples)
    write_json_atomic(output_dir / "run_status.json", payload)


def _write_intervention_model_outputs(path: Path, outputs: list[InterventionModelOutput]) -> None:
    write_jsonl(path, [_slim_intervention_model_output(output) for output in outputs])


def _slim_intervention_model_output(output: InterventionModelOutput) -> InterventionModelOutput:
    return output.model_copy(update={"interventions": []}, deep=True)


def _count_trace_checkpoints(output_dir: Path) -> int:
    checkpoint_dir = output_dir / "checkpoints" / "traces"
    if not checkpoint_dir.exists():
        return 0
    return len([path for path in checkpoint_dir.glob("*.json") if not path.name.endswith(".failed.json")])


def _count_intervention_checkpoints(output_dir: Path) -> int:
    checkpoint_dir = output_dir / "checkpoints" / "interventions"
    if not checkpoint_dir.exists():
        return 0
    return len([path for path in checkpoint_dir.glob("*.json") if not path.name.endswith(".failed.json")])


def _count_target_stage_checkpoints(output_dir: Path) -> int:
    checkpoint_dir = output_dir / "checkpoints" / "examples"
    if not checkpoint_dir.exists():
        return 0
    return len(list(checkpoint_dir.glob("*/target_stage.json")))


def _read_jsonl_if_exists(path: Path, model: type[Any]) -> list[Any]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return read_jsonl(path, model)


def _records_by_example_id(records: list[GeneratedTraceRecord]) -> dict[str, GeneratedTraceRecord]:
    return {record.example_id: record for record in records}


def _records_by_intervention_id(records: list[Any]) -> dict[str, Any]:
    return {record.intervention_id: record for record in records}


def _trace_checkpoint_path(output_dir: Path, example_id: str) -> Path:
    return output_dir / "checkpoints" / "traces" / f"{_safe_id(example_id)}.json"


def _read_trace_checkpoint(output_dir: Path, example_id: str) -> GeneratedTraceRecord | None:
    path = _trace_checkpoint_path(output_dir, example_id)
    if not path.exists():
        return None
    try:
        return GeneratedTraceRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_trace_checkpoint(output_dir: Path, record: GeneratedTraceRecord) -> None:
    write_json_atomic(_trace_checkpoint_path(output_dir, record.example_id), record)
    failed_path = output_dir / "checkpoints" / "traces" / f"{_safe_id(record.example_id)}.failed.json"
    failed_path.unlink(missing_ok=True)


def _write_trace_failure_checkpoint(output_dir: Path, example_id: str, payload: dict[str, Any]) -> None:
    path = output_dir / "checkpoints" / "traces" / f"{_safe_id(example_id)}.failed.json"
    write_json_atomic(path, payload)


def _intervention_checkpoint_path(output_dir: Path, example_id: str) -> Path:
    return output_dir / "checkpoints" / "interventions" / f"{_safe_id(example_id)}.json"


def _read_intervention_checkpoint(output_dir: Path, example_id: str) -> InterventionModelOutput | None:
    path = _intervention_checkpoint_path(output_dir, example_id)
    if not path.exists():
        return None
    try:
        return InterventionModelOutput.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_intervention_checkpoint(output_dir: Path, output: InterventionModelOutput) -> None:
    write_json_atomic(_intervention_checkpoint_path(output_dir, output.example_id), output)


def _write_intervention_failure_checkpoint(output_dir: Path, output: InterventionModelOutput) -> None:
    path = output_dir / "checkpoints" / "interventions" / f"{_safe_id(output.example_id)}.failed.json"
    write_json_atomic(path, output)


def _validation_rows_from_completed_checkpoints(output_dir: Path, examples: list[Example]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        path = output_dir / "checkpoints" / "examples" / _safe_id(example.example_id) / "intervention_validation.jsonl"
        if path.exists():
            rows.extend(_read_jsonl_if_exists(path, AnyModel))
    return [row.payload for row in rows]


def _attach_example_metadata(records: list[IntervenedTrace], example: Example) -> None:
    for record in records:
        record.metadata["dataset"] = example.dataset
        record.metadata["source_index"] = example.metadata.get("source_index")
        record.metadata["expected_modality_profile"] = example.metadata.get("expected_modality_profile")


def _safe_id(example_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in example_id)


class AnyModel:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    @classmethod
    def model_validate(cls, payload: dict[str, Any]) -> AnyModel:
        return cls(payload)


def _unload_runner(runner: Any) -> None:
    unload = getattr(runner, "unload", None)
    if callable(unload):
        unload()


def _shortcut_control_records(traces_by_example: dict[str, StructuredTrace], examples: list[Example]) -> list[IntervenedTrace]:
    controls: list[IntervenedTrace] = []
    for example in examples:
        trace = traces_by_example[example.example_id]
        controls.extend(
            [
                _control_record(example.example_id, trace, "no_trace", trace_mode="none", include_image=True),
                _control_record(example.example_id, trace, "text_only_no_trace", trace_mode="none", include_image=False),
                _control_record(example.example_id, trace, "no_image_with_original_trace", trace_mode="support_trace", include_image=False),
            ]
        )
    return controls


def _control_record(
    example_id: str,
    trace: StructuredTrace,
    condition: str,
    trace_mode: str,
    include_image: bool,
) -> IntervenedTrace:
    return IntervenedTrace(
        intervention_id=f"{example_id}::{condition}",
        example_id=example_id,
        original_trace=trace,
        intervened_trace=trace,
        intervention={
            "intervention_type": condition,
            "target_component": "none",
            "expected_label": "original",
            "expected_flawed_component": "none",
            "known_location": False,
            "details": {
                "condition": condition,
                "operation": condition,
                "changed_field": "none",
                "unchanged_fields": "all",
                "is_flawed": False,
                "trace_mode": trace_mode,
                "include_image": include_image,
            },
        },
    )
