from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from math import sqrt
from statistics import mean
from typing import Any

import pandas as pd

from aria.schemas import Component, ExpectedLabel, IntervenedTrace, JudgeOutput, TargetOutput


def judge_summary(records: Iterable[IntervenedTrace], outputs: Iterable[JudgeOutput]) -> dict[str, Any]:
    pairs = _paired(records, outputs)
    original_pairs = [pair for pair in pairs if pair[0].intervention.expected_label == ExpectedLabel.ORIGINAL]
    intervened_pairs = [
        pair for pair in pairs if pair[0].intervention.expected_label == ExpectedLabel.INTERVENED
    ]

    return {
        "n": len(pairs),
        "intervention_detection_accuracy": intervention_detection_accuracy_from_pairs(pairs),
        "localization_accuracy": localization_accuracy_from_pairs(pairs),
        "false_positive_rate_original": false_positive_rate(original_pairs),
        "false_negative_rate_intervened": false_negative_rate(intervened_pairs),
        "avg_judge_score_original": _avg_score(original_pairs),
        "avg_judge_score_intervened": _avg_score(intervened_pairs),
        "judge_score_drop_original_to_intervened": _avg_score(original_pairs) - _avg_score(intervened_pairs),
    }


def judge_by_intervention(
    records: Iterable[IntervenedTrace],
    outputs: Iterable[JudgeOutput],
) -> pd.DataFrame:
    groups: dict[str, list[tuple[IntervenedTrace, JudgeOutput]]] = defaultdict(list)
    for pair in _paired(records, outputs):
        groups[pair[0].intervention.intervention_type].append(pair)

    rows = []
    for intervention_type, pairs in sorted(groups.items()):
        rows.append(
            {
                "intervention_type": intervention_type,
                "n": len(pairs),
                "detection_accuracy": intervention_detection_accuracy_from_pairs(pairs),
                "localization_accuracy": localization_accuracy_from_pairs(pairs),
                "avg_judge_score": _avg_score(pairs),
            }
        )
    return pd.DataFrame(rows)


def judge_by_component(
    records: Iterable[IntervenedTrace],
    outputs: Iterable[JudgeOutput],
) -> pd.DataFrame:
    groups: dict[str, list[tuple[IntervenedTrace, JudgeOutput]]] = defaultdict(list)
    for pair in _paired(records, outputs):
        component = pair[0].expected_flawed_component.value
        groups[component].append(pair)

    rows = []
    for component, pairs in sorted(groups.items()):
        rows.append(
            {
                "component": component,
                "n": len(pairs),
                "localization_accuracy": localization_accuracy_from_pairs(pairs),
                "detection_accuracy": intervention_detection_accuracy_from_pairs(pairs),
            }
        )
    return pd.DataFrame(rows)


def intervention_detection_accuracy(
    records: Iterable[IntervenedTrace],
    judge_labels: Iterable[bool],
) -> float:
    pairs = list(zip(records, judge_labels, strict=True))
    if not pairs:
        return 0.0
    correct = sum(record.label_is_intervened == judge_label for record, judge_label in pairs)
    return correct / len(pairs)


def localization_accuracy(
    records: Iterable[IntervenedTrace],
    predicted_components: Iterable[Component | str | None],
) -> float:
    pairs = [
        (record, _component(predicted))
        for record, predicted in zip(records, predicted_components, strict=True)
        if record.label_is_intervened and record.intervention.known_location
    ]
    if not pairs:
        return 0.0
    correct = sum(record.expected_flawed_component == predicted for record, predicted in pairs)
    return correct / len(pairs)


def target_summary(outputs: Iterable[TargetOutput]) -> dict[str, Any]:
    output_list = list(outputs)
    original_outputs = [item for item in output_list if item.metadata.get("expected_label") == "original"]
    intervened_outputs = [
        item for item in output_list if item.metadata.get("expected_label") == "intervened"
    ]
    return {
        "n_target_outputs": len(output_list),
        "answer_accuracy_original": _accuracy(original_outputs),
        "answer_accuracy_intervened": _accuracy(intervened_outputs),
        "avg_confidence_original": _avg_optional(original_outputs, "confidence"),
        "avg_confidence_intervened": _avg_optional(intervened_outputs, "confidence"),
        "avg_logit_margin_original": _avg_optional(original_outputs, "logit_margin"),
        "avg_logit_margin_intervened": _avg_optional(intervened_outputs, "logit_margin"),
    }


def mediation_summary(
    records: Iterable[IntervenedTrace],
    target_outputs: Iterable[TargetOutput],
    judge_outputs: Iterable[JudgeOutput] | None = None,
) -> dict[str, Any]:
    rows = target_effect_rows(records, target_outputs, judge_outputs)
    originally_correct_rows = [row for row in rows if row["original_condition_correct"] == 1.0]
    intervened_rows = [row for row in originally_correct_rows if row["expected_label"] == ExpectedLabel.INTERVENED.value]
    paraphrase_rows = [row for row in originally_correct_rows if row["intervention_type"] == "paraphrase"]
    binding_rows = [row for row in originally_correct_rows if row["intervention_type"] == "wrong_binding"]
    retargeted_rows = [row for row in originally_correct_rows if row["intervention_type"] == "retargeted_support"]
    no_trace_rows = [row for row in rows if row["intervention_type"] == "no_trace"]
    text_only_rows = [row for row in rows if row["intervention_type"] == "text_only_no_trace"]
    no_image_trace_rows = [row for row in rows if row["intervention_type"] == "no_image_with_original_trace"]
    delete_binding_rows = [row for row in originally_correct_rows if row["intervention_type"] == "delete_binding_claim"]
    summary = {
        "n_target_effects": len(rows),
        "n_originally_correct": len({row["example_id"] for row in originally_correct_rows}),
        "original_trace_accuracy": _avg_key([row for row in rows if row["intervention_type"] == "original_trace"], "is_correct"),
        "no_trace_accuracy": _avg_key(no_trace_rows, "is_correct"),
        "text_only_no_trace_accuracy": _avg_key(text_only_rows, "is_correct"),
        "no_image_with_trace_accuracy": _avg_key(no_image_trace_rows, "is_correct"),
        "answer_flip_rate_intervened": _avg_key(intervened_rows, "answer_flipped"),
        "answer_flip_rate": _avg_key(intervened_rows, "answer_flipped"),
        "n_a_rate": _avg_key(intervened_rows, "is_abstention"),
        "unsupportedness_detection_rate": _avg_key(intervened_rows, "unsupportedness_detected"),
        "answer_flip_rate_wrong_binding": _avg_key(binding_rows, "answer_flipped"),
        "answer_flip_rate_retargeted_support": _avg_key(retargeted_rows, "answer_flipped"),
        "answer_flip_rate_paraphrase": _avg_key(paraphrase_rows, "answer_flipped"),
        "primary_binding_vs_paraphrase_flip_delta": _avg_key(binding_rows, "answer_flipped")
        - _avg_key(paraphrase_rows, "answer_flipped"),
        "retargeted_support_vs_paraphrase_flip_delta": _avg_key(retargeted_rows, "answer_flipped")
        - _avg_key(paraphrase_rows, "answer_flipped"),
        "correctness_drop_intervened": _avg_key(intervened_rows, "correctness_drop"),
        "correctness_drop_wrong_binding": _avg_key(binding_rows, "correctness_drop"),
        "correctness_drop_retargeted_support": _avg_key(retargeted_rows, "correctness_drop"),
        "correctness_drop_delete_binding": _avg_key(delete_binding_rows, "correctness_drop"),
        "retargeted_support_following_rate": _avg_key(retargeted_rows, "followed_retargeted_support"),
        "net_flip_effect": _avg_key([row for row in intervened_rows if row["intervention_type"] != "paraphrase"], "answer_flipped")
        - _avg_key(paraphrase_rows, "answer_flipped"),
        "net_abstention_effect": _avg_key([row for row in intervened_rows if row["intervention_type"] != "paraphrase"], "is_abstention")
        - _avg_key(paraphrase_rows, "is_abstention"),
        "net_retarget_following_effect": _avg_key(retargeted_rows, "followed_retargeted_support")
        - _avg_key(paraphrase_rows, "answered_retargeted_distractor"),
        "net_correctness_drop": _avg_key(paraphrase_rows, "is_correct")
        - _avg_key([row for row in intervened_rows if row["intervention_type"] != "paraphrase"], "is_correct"),
        "necessity_score": _avg_key(
            [row for row in intervened_rows if row["intervention_type"] != "paraphrase"],
            "answer_flipped",
        ),
        "sufficiency_score": _avg_key([row for row in rows if row["intervention_type"] == "original_trace"], "is_correct"),
        "shortcut_index": _avg_key([*no_trace_rows, *text_only_rows, *no_image_trace_rows], "is_correct"),
        "corrupted_but_still_correct_rate": _corrupted_but_still_correct_rate(rows),
        "judge_effect_spearman": _spearman(
            [row["judge_severity"] for row in rows if row.get("judge_severity") is not None],
            [row["answer_flipped"] for row in rows if row.get("judge_severity") is not None],
        ),
    }
    for dataset in sorted({str(row.get("dataset")) for row in rows if row.get("dataset")}):
        dataset_rows = [row for row in rows if row.get("dataset") == dataset]
        dataset_binding_rows = [
            row
            for row in dataset_rows
            if row["original_condition_correct"] == 1.0 and row["intervention_type"] == "wrong_binding"
        ]
        dataset_retargeted_rows = [
            row
            for row in dataset_rows
            if row["original_condition_correct"] == 1.0 and row["intervention_type"] == "retargeted_support"
        ]
        summary[f"{dataset}_n_target_effects"] = len(dataset_rows)
        summary[f"{dataset}_original_trace_accuracy"] = _avg_key(
            [row for row in dataset_rows if row["intervention_type"] == "original_trace"],
            "is_correct",
        )
        summary[f"{dataset}_answer_flip_rate_wrong_binding"] = _avg_key(dataset_binding_rows, "answer_flipped")
        summary[f"{dataset}_correctness_drop_wrong_binding"] = _avg_key(dataset_binding_rows, "correctness_drop")
        summary[f"{dataset}_answer_flip_rate_retargeted_support"] = _avg_key(dataset_retargeted_rows, "answer_flipped")
        summary[f"{dataset}_retargeted_support_following_rate"] = _avg_key(dataset_retargeted_rows, "followed_retargeted_support")
        summary[f"{dataset}_n_a_rate"] = _avg_key(
            [
                row
                for row in dataset_rows
                if row["original_condition_correct"] == 1.0 and row["expected_label"] == ExpectedLabel.INTERVENED.value
            ],
            "is_abstention",
        )
    return summary


def modality_effect_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_example: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_example[row["example_id"]][row["intervention_type"]] = row

    output: list[dict[str, Any]] = []
    semantic_conditions = [
        "wrong_visual",
        "wrong_textual",
        "wrong_binding",
        "wrong_principle",
        "wrong_inference",
        "delete_binding_claim",
    ]
    for example_id, example_rows in sorted(by_example.items()):
        original = example_rows.get("original_trace", {})
        paraphrase = example_rows.get("paraphrase", {})
        visual = example_rows.get("wrong_visual", {})
        textual = example_rows.get("wrong_textual", {})
        binding = example_rows.get("wrong_binding", {})
        original_correct = original.get("is_correct") == 1.0
        paraphrase_stable = paraphrase.get("answer_flipped", 0.0) == 0.0
        semantic_effects = {
            condition: _condition_effect(example_rows.get(condition, {}))
            for condition in semantic_conditions
            if condition in example_rows
        }
        trace_responsive = bool(original_correct and paraphrase_stable and any(value > 0 for value in semantic_effects.values()))
        shortcut_available = any(
            example_rows.get(condition, {}).get("is_correct") == 1.0
            for condition in ["no_trace", "text_only_no_trace", "no_image_with_original_trace"]
        )
        output.append(
            {
                "example_id": example_id,
                "dataset": original.get("dataset"),
                "expected_modality_profile": original.get("expected_modality_profile"),
                "original_correct": float(original_correct),
                "paraphrase_stable": float(paraphrase_stable),
                "wrong_visual_available": float("wrong_visual" in example_rows),
                "wrong_textual_available": float("wrong_textual" in example_rows),
                "visual_effect": _condition_effect(visual),
                "textual_effect": _condition_effect(textual),
                "binding_effect": _condition_effect(binding),
                "trace_responsive": float(trace_responsive),
                "shortcut_available": float(shortcut_available),
                "observed_reliance_label": _observed_reliance_label(
                    trace_responsive=trace_responsive,
                    visual_effect=_condition_effect(visual),
                    textual_effect=_condition_effect(textual),
                    binding_effect=_condition_effect(binding),
                    shortcut_available=shortcut_available,
                ),
            }
        )
    return output


def modality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modality_rows = modality_effect_rows(rows)
    summary = _modality_summary_for_rows(modality_rows, prefix="")
    for dataset in sorted({str(row.get("dataset")) for row in modality_rows if row.get("dataset")}):
        summary.update(_modality_summary_for_rows([row for row in modality_rows if row.get("dataset") == dataset], prefix=f"{dataset}_"))
    return summary


def target_effect_rows(
    records: Iterable[IntervenedTrace],
    target_outputs: Iterable[TargetOutput],
    judge_outputs: Iterable[JudgeOutput] | None = None,
) -> list[dict[str, Any]]:
    records_by_id = {record.intervention_id: record for record in records}
    outputs_by_id = {output.intervention_id: output for output in target_outputs}
    judges_by_id = {output.intervention_id: output for output in judge_outputs or []}
    original_answer_by_example = {
        record.example_id: outputs_by_id[record.intervention_id].final_answer
        for record in records_by_id.values()
        if record.intervention.intervention_type == "original_trace" and record.intervention_id in outputs_by_id
    }
    original_correct_by_example = {
        record.example_id: outputs_by_id[record.intervention_id].is_correct
        for record in records_by_id.values()
        if record.intervention.intervention_type == "original_trace" and record.intervention_id in outputs_by_id
    }

    rows: list[dict[str, Any]] = []
    for intervention_id, output in sorted(outputs_by_id.items()):
        record = records_by_id.get(intervention_id)
        if record is None:
            continue
        original_answer = original_answer_by_example.get(record.example_id)
        original_correct = original_correct_by_example.get(record.example_id)
        answer_flipped = (
            original_answer is not None and output.final_answer.strip().upper() != original_answer.strip().upper()
        )
        correctness_drop = (
            int(bool(original_correct)) - int(bool(output.is_correct))
            if original_correct is not None and output.is_correct is not None
            else 0
        )
        judge = judges_by_id.get(intervention_id)
        target_distractor = record.intervention.details.get("target_distractor")
        followed_retargeted_support = (
            record.intervention.intervention_type == "retargeted_support"
            and target_distractor is not None
            and output.final_answer.strip().upper() == str(target_distractor).strip().upper()
        )
        answered_retargeted_distractor = (
            target_distractor is not None
            and output.final_answer.strip().upper() == str(target_distractor).strip().upper()
        )
        is_abstention = output.final_answer.strip().upper() in {"N/A", "NA", "N.A.", "UNKNOWN"}
        unsupportedness_detected = is_abstention and record.intervention.expected_label == ExpectedLabel.INTERVENED
        answer_category = _answer_category(output.final_answer, original_answer, target_distractor, output.is_correct)
        rows.append(
            {
                "intervention_id": intervention_id,
                "example_id": record.example_id,
                "dataset": record.metadata.get("dataset"),
                "expected_modality_profile": record.metadata.get("expected_modality_profile"),
                "intervention_type": record.intervention.intervention_type,
                "component": record.intervention.target_component.value,
                "expected_label": record.intervention.expected_label.value,
                "condition": record.intervention.details.get("condition", "full_input_trace"),
                "original_answer": original_answer,
                "target_answer": output.final_answer,
                "target_distractor": target_distractor,
                "followed_retargeted_support": float(followed_retargeted_support),
                "answered_retargeted_distractor": float(answered_retargeted_distractor),
                "is_abstention": float(is_abstention),
                "unsupportedness_detected": float(unsupportedness_detected),
                "answer_category": answer_category,
                "answer_flipped": float(answer_flipped),
                "is_correct": float(output.is_correct) if output.is_correct is not None else 0.0,
                "original_condition_correct": float(bool(original_correct)),
                "correctness_drop": float(correctness_drop),
                "confidence": output.confidence,
                "logit_margin": output.logit_margin,
                "judge_severity": (5 - judge.faithfulness_score) if judge else None,
                "judge_flagged": judge.is_intervened_or_flawed if judge else None,
            }
        )
    return rows


def answer_flip_rate(original_answers: Iterable[str], intervened_answers: Iterable[str]) -> float:
    pairs = list(zip(original_answers, intervened_answers, strict=True))
    if not pairs:
        return 0.0
    flips = sum(original.strip().upper() != changed.strip().upper() for original, changed in pairs)
    return flips / len(pairs)


def answer_preservation_rate(original_answers: Iterable[str], intervened_answers: Iterable[str]) -> float:
    return 1.0 - answer_flip_rate(original_answers, intervened_answers)


def intervention_detection_accuracy_from_pairs(
    pairs: list[tuple[IntervenedTrace, JudgeOutput]],
) -> float:
    if not pairs:
        return 0.0
    correct = sum(record.label_is_intervened == output.is_intervened_or_flawed for record, output in pairs)
    return correct / len(pairs)


def localization_accuracy_from_pairs(pairs: list[tuple[IntervenedTrace, JudgeOutput]]) -> float:
    eligible = [
        (record, output)
        for record, output in pairs
        if record.label_is_intervened and record.intervention.known_location
    ]
    if not eligible:
        return 0.0
    correct = sum(record.expected_flawed_component == output.flawed_component for record, output in eligible)
    return correct / len(eligible)


def false_positive_rate(pairs: list[tuple[IntervenedTrace, JudgeOutput]]) -> float:
    if not pairs:
        return 0.0
    positives = sum(output.is_intervened_or_flawed for _, output in pairs)
    return positives / len(pairs)


def false_negative_rate(pairs: list[tuple[IntervenedTrace, JudgeOutput]]) -> float:
    if not pairs:
        return 0.0
    negatives = sum(not output.is_intervened_or_flawed for _, output in pairs)
    return negatives / len(pairs)


# Backward-compatible names from the scaffold.
def detection_accuracy(records: Iterable[IntervenedTrace], judge_labels: Iterable[bool]) -> float:
    return intervention_detection_accuracy(records, judge_labels)


def _paired(
    records: Iterable[IntervenedTrace],
    outputs: Iterable[JudgeOutput],
) -> list[tuple[IntervenedTrace, JudgeOutput]]:
    output_by_id = {output.intervention_id: output for output in outputs}
    return [(record, output_by_id[record.intervention_id]) for record in records if record.intervention_id in output_by_id]


def _component(value: Component | str | None) -> Component:
    if isinstance(value, Component):
        return value
    if value is None:
        return Component.UNKNOWN
    try:
        return Component(value)
    except ValueError:
        return Component.UNKNOWN


def _avg_score(pairs: list[tuple[IntervenedTrace, JudgeOutput]]) -> float:
    if not pairs:
        return 0.0
    return mean(output.faithfulness_score for _, output in pairs)


def _accuracy(outputs: list[TargetOutput]) -> float:
    known = [item for item in outputs if item.is_correct is not None]
    if not known:
        return 0.0
    return sum(item.is_correct for item in known) / len(known)


def _avg_optional(outputs: list[TargetOutput], field: str) -> float:
    values = [getattr(item, field) for item in outputs if getattr(item, field) is not None]
    return mean(values) if values else 0.0


def _avg_key(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else 0.0


def _corrupted_but_still_correct_rate(rows: list[dict[str, Any]]) -> float:
    by_example: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_example[row["example_id"]][row["intervention_type"]] = row
    eligible = [
        example_rows
        for example_rows in by_example.values()
        if example_rows.get("original_trace", {}).get("is_correct") == 1.0
    ]
    if not eligible:
        return 0.0
    corrupted_but_still_correct = 0
    for example_rows in eligible:
        wrong_binding_correct = example_rows.get("wrong_binding", {}).get("is_correct") == 1.0
        shortcut_correct = any(
            example_rows.get(condition, {}).get("is_correct") == 1.0
            for condition in ["no_trace", "text_only_no_trace", "no_image_with_original_trace"]
        )
        corrupted_but_still_correct += int(wrong_binding_correct and shortcut_correct)
    return corrupted_but_still_correct / len(eligible)


def _condition_effect(row: dict[str, Any]) -> float:
    return max(float(row.get("answer_flipped") or 0.0), float(row.get("correctness_drop") or 0.0))


def _answer_category(
    answer: str,
    original_answer: str | None,
    target_distractor: Any,
    is_correct: bool | None,
) -> str:
    normalized = answer.strip().upper()
    if normalized in {"N/A", "NA", "N.A.", "UNKNOWN"}:
        return "abstention"
    if target_distractor is not None and normalized == str(target_distractor).strip().upper():
        return "retargeted_distractor"
    if original_answer is not None and normalized == original_answer.strip().upper():
        return "original_answer"
    if is_correct:
        return "correct_answer"
    return "other_wrong"


def _observed_reliance_label(
    trace_responsive: bool,
    visual_effect: float,
    textual_effect: float,
    binding_effect: float,
    shortcut_available: bool,
) -> str:
    if not trace_responsive:
        return "corruption_insensitive_or_indeterminate" if shortcut_available else "indeterminate"
    effects = {
        "visual_load_bearing": visual_effect,
        "textual_load_bearing": textual_effect,
        "binding_load_bearing": binding_effect,
    }
    max_effect = max(effects.values())
    winners = [label for label, value in effects.items() if value == max_effect and value > 0]
    return winners[0] if len(winners) == 1 else "mixed_or_ambiguous"


def _modality_summary_for_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    if not rows:
        return {
            f"{prefix}n_modality_examples": 0,
            f"{prefix}trace_responsive_rate": 0.0,
            f"{prefix}corruption_insensitive_or_indeterminate_rate": 0.0,
            f"{prefix}visual_effect_rate": 0.0,
            f"{prefix}textual_effect_rate": 0.0,
            f"{prefix}binding_effect_rate": 0.0,
            f"{prefix}modality_alignment_rate": 0.0,
        }
    return {
        f"{prefix}n_modality_examples": len(rows),
        f"{prefix}trace_responsive_rate": _avg_key(rows, "trace_responsive"),
        f"{prefix}corruption_insensitive_or_indeterminate_rate": mean(
            1.0 if row["observed_reliance_label"] == "corruption_insensitive_or_indeterminate" else 0.0
            for row in rows
        ),
        f"{prefix}visual_effect_rate": mean(1.0 if row["visual_effect"] > 0 else 0.0 for row in rows),
        f"{prefix}textual_effect_rate": mean(1.0 if row["textual_effect"] > 0 else 0.0 for row in rows),
        f"{prefix}binding_effect_rate": mean(1.0 if row["binding_effect"] > 0 else 0.0 for row in rows),
        f"{prefix}modality_alignment_rate": _modality_alignment_rate(rows),
    }


def _modality_alignment_rate(rows: list[dict[str, Any]]) -> float:
    eligible = [
        row for row in rows
        if row.get("trace_responsive") == 1.0 and row.get("expected_modality_profile") not in {None, "", "ambiguous"}
    ]
    if not eligible:
        return 0.0
    expected_to_label = {
        "image_led": "visual_load_bearing",
        "text_led": "textual_load_bearing",
        "joint_image_text": "binding_load_bearing",
    }
    aligned = 0
    count = 0
    for row in eligible:
        expected = expected_to_label.get(str(row.get("expected_modality_profile")))
        if expected is None:
            continue
        count += 1
        aligned += int(row.get("observed_reliance_label") == expected)
    return aligned / count if count else 0.0


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + end + 1) / 2
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)
