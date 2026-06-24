from __future__ import annotations

from copy import deepcopy
from typing import Any

from aria.parsing import parse_intervention_payload
from aria.schemas import (
    BindingClaim,
    Component,
    ExpectedLabel,
    InterventionMetadata,
    InterventionModelOutput,
    InterventionValidationRow,
    IntervenedTrace,
    StructuredTrace,
    SupportClaim,
)


OMITTED = "[omitted]"


def original(trace: StructuredTrace, example_id: str) -> IntervenedTrace:
    return _record(
        intervention_id=f"{example_id}::original_trace",
        example_id=example_id,
        original_trace=trace,
        intervened_trace=deepcopy(trace),
        intervention_type="original_trace",
        target_component=Component.NONE,
        expected_label=ExpectedLabel.ORIGINAL,
        expected_flawed_component=Component.NONE,
        known_location=False,
        details={
            "condition": "original_trace",
            "operation": "identity",
            "changed_field": "none",
            "unchanged_fields": "all",
            "is_flawed": False,
        },
    )


def paraphrase(
    trace: StructuredTrace,
    example_id: str,
    paraphrase_suffix: str = " Restated without changing the scientific meaning.",
) -> IntervenedTrace:
    mutated = deepcopy(trace)
    mutated.binding_claim.text = f"{mutated.binding_claim.text}{paraphrase_suffix}"
    return _record(
        intervention_id=f"{example_id}::paraphrase",
        example_id=example_id,
        original_trace=trace,
        intervened_trace=mutated,
        intervention_type="paraphrase",
        target_component=Component.NONE,
        expected_label=ExpectedLabel.ORIGINAL,
        expected_flawed_component=Component.NONE,
        known_location=False,
        details={
            "negative_control": True,
            "condition": "paraphrase",
            "operation": "surface_paraphrase",
            "changed_field": "binding_claim.text",
            "old_value": trace.binding_claim.text,
            "new_value": mutated.binding_claim.text,
            "unchanged_fields": "visual_evidence,textual_evidence,binding_claim.ids,binding_claim.supports,scientific_principle,inference,final_answer",
            "is_flawed": False,
        },
    )


def wrong_binding(
    trace: StructuredTrace,
    example_id: str,
    choices: dict[str, str] | None = None,
    reference_answer: str | None = None,
    replacement: str | None = None,
) -> IntervenedTrace:
    if isinstance(choices, str) and replacement is None:
        replacement = choices
        choices = None
    mutated = deepcopy(trace)
    old_option = _bound_or_reference_answer(trace, reference_answer)
    new_option = _choose_distractor(choices or {}, old_option)
    old_text = trace.binding_claim.text
    old_value = _format_option(old_option, choices)
    new_value = _format_option(new_option, choices)
    mutated.binding_claim.bound_answer_option = new_option
    mutated.binding_claim.text = replacement or (
        "The same cited visual and textual evidence is instead bound to "
        f"answer option {new_value}, while the evidence and principle are kept unchanged."
    )
    return _record(
        intervention_id=f"{example_id}::wrong_binding",
        example_id=example_id,
        original_trace=trace,
        intervened_trace=mutated,
        intervention_type="wrong_binding",
        target_component=Component.BINDING_CLAIM,
        expected_flawed_component=Component.BINDING_CLAIM,
        details={
            "condition": "wrong_binding",
            "operation": "answer_option_binding_swap",
            "changed_field": "binding_claim.bound_answer_option,binding_claim.text",
            "old_value": old_value,
            "new_value": new_value,
            "old_text": old_text,
            "new_text": mutated.binding_claim.text,
            "unchanged_fields": "visual_evidence,textual_evidence,scientific_principle,inference,conclusion,final_answer",
            "is_flawed": True,
            "validity": {
                "local_facts_preserved": True,
                "principle_preserved": True,
                "intended_fault_type": "wrong_binding",
                "distractor_is_same_item_option": new_option in (choices or {}),
            },
        },
    )


def retargeted_support(
    trace: StructuredTrace,
    example_id: str,
    choices: dict[str, str] | None = None,
    reference_answer: str | None = None,
) -> IntervenedTrace:
    choices = choices or {}
    mutated = deepcopy(trace)
    old_option = _bound_or_reference_answer(trace, reference_answer)
    target_option = _choose_distractor(choices, old_option)
    old_binding_text = trace.binding_claim.text
    old_inference_text = trace.inference.text
    target_value = _format_option(target_option, choices)
    mutated.binding_claim.bound_answer_option = target_option
    mutated.binding_claim.text = (
        f"The cited evidence supports answer option {target_value} as the answer-relevant object, condition, or relation."
    )
    mutated.inference.text = f"Using the stated principle, the cited evidence is most consistent with answer option {target_value}."
    return _record(
        intervention_id=f"{example_id}::retargeted_support",
        example_id=example_id,
        original_trace=trace,
        intervened_trace=mutated,
        intervention_type="retargeted_support",
        target_component=Component.BINDING_CLAIM,
        expected_flawed_component=Component.BINDING_CLAIM,
        details={
            "condition": "retargeted_support",
            "operation": "answer_option_support_retarget",
            "changed_field": "binding_claim.bound_answer_option,binding_claim.text,inference.text",
            "old_value": _format_option(old_option, choices),
            "new_value": target_value,
            "old_binding_text": old_binding_text,
            "new_binding_text": mutated.binding_claim.text,
            "old_inference_text": old_inference_text,
            "new_inference_text": mutated.inference.text,
            "unchanged_fields": "visual_evidence,textual_evidence,scientific_principle,conclusion,final_answer,binding_claim.ids,binding_claim.supports,inference.ids,inference.supports",
            "is_flawed": True,
            "target_distractor": target_option,
            "validity": {
                "intended_fault_type": "retargeted_support",
                "subtype": "answer_option_support_retarget",
                "distractor_is_same_item_option": target_option in choices,
                "deterministic_template": True,
            },
        },
    )


def wrong_principle(
    trace: StructuredTrace,
    example_id: str,
    donor_trace: StructuredTrace,
    donor_example_id: str,
    donor_metadata: dict[str, Any] | None = None,
) -> IntervenedTrace:
    mutated = deepcopy(trace)
    old_text = trace.scientific_principle.text
    mutated.scientific_principle = SupportClaim(
        id=trace.scientific_principle.id,
        text=donor_trace.scientific_principle.text,
        supports=list(trace.scientific_principle.supports),
    )
    return _record(
        intervention_id=f"{example_id}::wrong_principle",
        example_id=example_id,
        original_trace=trace,
        intervened_trace=mutated,
        intervention_type="wrong_principle",
        target_component=Component.SCIENTIFIC_PRINCIPLE,
        expected_flawed_component=Component.SCIENTIFIC_PRINCIPLE,
        details={
            "condition": "wrong_principle",
            "operation": "principle_swap_from_donor",
            "changed_field": "scientific_principle.text",
            "old_value": old_text,
            "new_value": mutated.scientific_principle.text,
            "unchanged_fields": "visual_evidence,textual_evidence,binding_claim,inference,conclusion,final_answer",
            "is_flawed": True,
            "donor_example_id": donor_example_id,
            "donor_metadata": donor_metadata or {},
        },
    )


def wrong_visual(
    trace: StructuredTrace,
    example_id: str,
    donor_trace: StructuredTrace,
    donor_example_id: str,
    donor_metadata: dict[str, Any] | None = None,
) -> IntervenedTrace | None:
    if not trace.visual_evidence or not donor_trace.visual_evidence:
        return None
    mutated = deepcopy(trace)
    old_evidence = trace.visual_evidence[0]
    donor_evidence = donor_trace.visual_evidence[0]
    mutated.visual_evidence[0].content = donor_evidence.content
    mutated.visual_evidence[0].source_ref = donor_evidence.source_ref or donor_evidence.provenance
    mutated.visual_evidence[0].provenance = donor_evidence.provenance or donor_evidence.source_ref
    return _record(
        intervention_id=f"{example_id}::wrong_visual",
        example_id=example_id,
        original_trace=trace,
        intervened_trace=mutated,
        intervention_type="wrong_visual",
        target_component=Component.VISUAL_EVIDENCE,
        expected_flawed_component=Component.VISUAL_EVIDENCE,
        details={
            "condition": "wrong_visual",
            "operation": "visual_evidence_swap_from_donor",
            "changed_field": "visual_evidence[0].content,visual_evidence[0].source_ref",
            "old_value": old_evidence.content,
            "new_value": donor_evidence.content,
            "unchanged_fields": "visual_evidence.ids,visual_evidence.support_links,textual_evidence,binding_claim,scientific_principle,inference,conclusion,final_answer",
            "is_flawed": True,
            "donor_example_id": donor_example_id,
            "donor_metadata": donor_metadata or {},
            "eligibility": {
                "source_has_visual_evidence": True,
                "donor_has_visual_evidence": True,
            },
        },
    )


def wrong_textual(
    trace: StructuredTrace,
    example_id: str,
    donor_trace: StructuredTrace,
    donor_example_id: str,
    donor_metadata: dict[str, Any] | None = None,
) -> IntervenedTrace | None:
    if not trace.textual_evidence or not donor_trace.textual_evidence:
        return None
    mutated = deepcopy(trace)
    old_evidence = trace.textual_evidence[0]
    donor_evidence = donor_trace.textual_evidence[0]
    mutated.textual_evidence[0].content = donor_evidence.content
    mutated.textual_evidence[0].source_ref = donor_evidence.source_ref or donor_evidence.provenance
    mutated.textual_evidence[0].provenance = donor_evidence.provenance or donor_evidence.source_ref
    return _record(
        intervention_id=f"{example_id}::wrong_textual",
        example_id=example_id,
        original_trace=trace,
        intervened_trace=mutated,
        intervention_type="wrong_textual",
        target_component=Component.TEXTUAL_EVIDENCE,
        expected_flawed_component=Component.TEXTUAL_EVIDENCE,
        details={
            "condition": "wrong_textual",
            "operation": "textual_evidence_swap_from_donor",
            "changed_field": "textual_evidence[0].content,textual_evidence[0].source_ref",
            "old_value": old_evidence.content,
            "new_value": donor_evidence.content,
            "unchanged_fields": "textual_evidence.ids,textual_evidence.support_links,visual_evidence,binding_claim,scientific_principle,inference,conclusion,final_answer",
            "is_flawed": True,
            "donor_example_id": donor_example_id,
            "donor_metadata": donor_metadata or {},
            "eligibility": {
                "source_has_textual_evidence": True,
                "donor_has_textual_evidence": True,
            },
        },
    )


def wrong_inference(
    trace: StructuredTrace,
    example_id: str,
    choices: dict[str, str] | None = None,
    reference_answer: str | None = None,
    replacement: str | None = None,
) -> IntervenedTrace:
    if isinstance(choices, str) and replacement is None:
        replacement = choices
        choices = None
    mutated = deepcopy(trace)
    old_option = _bound_or_reference_answer(trace, reference_answer)
    new_option = _choose_distractor(choices or {}, old_option)
    old_text = trace.inference.text
    mutated.inference.text = replacement or (
        f"Using the unchanged evidence and principle, infer that option {_format_option(new_option, choices)} follows."
    )
    return _record(
        intervention_id=f"{example_id}::wrong_inference",
        example_id=example_id,
        original_trace=trace,
        intervened_trace=mutated,
        intervention_type="wrong_inference",
        target_component=Component.INFERENCE,
        expected_flawed_component=Component.INFERENCE,
        details={
            "condition": "wrong_inference",
            "operation": "inference_answer_swap",
            "changed_field": "inference.text",
            "old_value": old_text,
            "new_value": mutated.inference.text,
            "unchanged_fields": "visual_evidence,textual_evidence,binding_claim,scientific_principle,conclusion,final_answer",
            "is_flawed": True,
            "old_answer_option": old_option or "",
            "new_answer_option": new_option,
        },
    )


def delete_component(trace: StructuredTrace, example_id: str, component: Component) -> IntervenedTrace:
    mutated = deepcopy(trace)
    if component != Component.BINDING_CLAIM:
        raise ValueError("Experiment 1 deletion supports only binding_claim")
    old_text = trace.binding_claim.text
    mutated.binding_claim = BindingClaim(
        id=trace.binding_claim.id,
        text=OMITTED,
        supports=[],
        uses_visual=[],
        uses_textual=[],
    )
    return _record(
        intervention_id=f"{example_id}::delete_binding_claim",
        example_id=example_id,
        original_trace=trace,
        intervened_trace=mutated,
        intervention_type="delete_binding_claim",
        target_component=component,
        expected_flawed_component=component,
        details={
            "condition": "delete_binding_claim",
            "operation": "delete_binding_claim",
            "changed_field": "binding_claim",
            "old_value": old_text,
            "new_value": OMITTED,
            "unchanged_fields": "visual_evidence,textual_evidence,scientific_principle,inference,conclusion,final_answer",
            "is_flawed": True,
        },
    )


def build_faithfulness_interventions(
    traces_by_example: dict[str, StructuredTrace],
    answer_choices_by_example: dict[str, dict[str, str]] | None = None,
    reference_answers_by_example: dict[str, str | None] | None = None,
    example_metadata_by_id: dict[str, dict[str, Any]] | None = None,
    selected_example_ids: set[str] | None = None,
    include_modality_interventions: bool = False,
    include_retargeted_support: bool = False,
) -> list[IntervenedTrace]:
    """Create the Experiment 1 deterministic intervention set."""
    answer_choices_by_example = answer_choices_by_example or {}
    reference_answers_by_example = reference_answers_by_example or {}
    example_metadata_by_id = example_metadata_by_id or {}
    records: list[IntervenedTrace] = []

    for example_id, trace in traces_by_example.items():
        if selected_example_ids is not None and example_id not in selected_example_ids:
            continue
        choices = answer_choices_by_example.get(example_id, {})
        reference_answer = reference_answers_by_example.get(example_id)
        records.extend(
            [
                original(trace, example_id),
                paraphrase(trace, example_id),
                wrong_binding(trace, example_id, choices=choices, reference_answer=reference_answer),
                *(
                    [retargeted_support(trace, example_id, choices=choices, reference_answer=reference_answer)]
                    if include_retargeted_support
                    else []
                ),
                wrong_inference(trace, example_id, choices=choices, reference_answer=reference_answer),
                delete_component(trace, example_id, Component.BINDING_CLAIM),
            ]
        )
        donor = _find_principle_donor(example_id, traces_by_example, example_metadata_by_id)
        if donor is not None:
            donor_id, donor_trace = donor
            records.append(
                wrong_principle(
                    trace,
                    example_id,
                    donor_trace=donor_trace,
                    donor_example_id=donor_id,
                    donor_metadata=example_metadata_by_id.get(donor_id, {}),
                )
            )
        if include_modality_interventions:
            visual_donor = _find_evidence_donor(example_id, traces_by_example, example_metadata_by_id, Component.VISUAL_EVIDENCE)
            if visual_donor is not None:
                donor_id, donor_trace = visual_donor
                visual_record = wrong_visual(
                    trace,
                    example_id,
                    donor_trace=donor_trace,
                    donor_example_id=donor_id,
                    donor_metadata=example_metadata_by_id.get(donor_id, {}),
                )
                if visual_record is not None:
                    records.append(visual_record)
            textual_donor = _find_evidence_donor(example_id, traces_by_example, example_metadata_by_id, Component.TEXTUAL_EVIDENCE)
            if textual_donor is not None:
                donor_id, donor_trace = textual_donor
                textual_record = wrong_textual(
                    trace,
                    example_id,
                    donor_trace=donor_trace,
                    donor_example_id=donor_id,
                    donor_metadata=example_metadata_by_id.get(donor_id, {}),
                )
                if textual_record is not None:
                    records.append(textual_record)
    return records


def template_interventions_for_example(
    example_id: str,
    traces_by_example: dict[str, StructuredTrace],
    answer_choices_by_example: dict[str, dict[str, str]] | None = None,
    reference_answers_by_example: dict[str, str | None] | None = None,
    example_metadata_by_id: dict[str, dict[str, Any]] | None = None,
    include_modality_interventions: bool = False,
    include_retargeted_support: bool = True,
) -> list[IntervenedTrace]:
    return build_faithfulness_interventions(
        traces_by_example=traces_by_example,
        answer_choices_by_example=answer_choices_by_example,
        reference_answers_by_example=reference_answers_by_example,
        example_metadata_by_id=example_metadata_by_id,
        selected_example_ids={example_id},
        include_modality_interventions=include_modality_interventions,
        include_retargeted_support=include_retargeted_support,
    )


def build_model_interventions_from_output(
    output: InterventionModelOutput,
    template_records: list[IntervenedTrace],
) -> list[IntervenedTrace]:
    if output.parse_error:
        raise ValueError(f"Intervention model output was not parseable: {output.parse_error}")
    if output.raw_output is None:
        raise ValueError("Intervention model output has no raw_output")

    payload = parse_intervention_payload(output.raw_output)
    templates_by_type = {
        record.intervention.intervention_type: record
        for record in template_records
        if record.intervention.intervention_type != "original_trace"
    }
    items = payload.get("interventions")
    if items is None:
        items = _items_from_flat_intervention_payload(payload, templates_by_type)
    records = [
        deepcopy(record)
        for record in template_records
        if record.intervention.intervention_type == "original_trace"
    ]
    seen_types: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each generated intervention must be an object")
        intervention_type = str(item.get("intervention_type") or "").strip()
        template = templates_by_type.get(intervention_type)
        if template is None:
            raise ValueError(f"Unexpected intervention_type from intervention model: {intervention_type!r}")
        if intervention_type == "delete_binding_claim":
            trace_payload = {}
        else:
            trace_payload = _component_patch_from_model_item(item)
            if not isinstance(trace_payload, dict) or not trace_payload:
                raise ValueError(f"{intervention_type} has no changed component fields")
        generated_trace = _structured_trace_from_model_payload(template.intervened_trace, trace_payload)
        record = deepcopy(template)
        record.intervened_trace = generated_trace
        record.intervention.details = {
            **record.intervention.details,
            "operation": f"intervention_model_{record.intervention.details.get('operation', intervention_type)}",
            "generated_by_intervention_model": True,
            "intervention_model": output.intervention_model,
        }
        records.append(record)
        seen_types.add(intervention_type)

    missing = sorted(set(templates_by_type) - seen_types)
    if missing:
        raise ValueError(f"Intervention model omitted required interventions: {', '.join(missing)}")
    return records


def _items_from_flat_intervention_payload(
    payload: dict[str, Any],
    templates_by_type: dict[str, IntervenedTrace],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if "paraphrase" in templates_by_type:
        patch: dict[str, Any] = {}
        _copy_flat_payload_field(patch, payload, "paraphrase_binding_claim", "binding_claim", "text")
        _copy_flat_payload_field(patch, payload, "paraphrase_scientific_principle", "scientific_principle", "text")
        _copy_flat_payload_field(patch, payload, "paraphrase_principle", "scientific_principle", "text")
        _copy_flat_payload_field(patch, payload, "paraphrase_inference", "inference", "text")
        if patch:
            items.append({"intervention_type": "paraphrase", "component_patch": patch})
    if "wrong_binding" in templates_by_type:
        patch = {}
        _copy_flat_payload_field(patch, payload, "wrong_binding_binding_claim", "binding_claim", "text")
        _copy_flat_payload_field(
            patch,
            payload,
            "wrong_binding_bound_answer_option",
            "binding_claim",
            "bound_answer_option",
        )
        if patch:
            items.append({"intervention_type": "wrong_binding", "component_patch": patch})
    if "retargeted_support" in templates_by_type:
        patch = {}
        _copy_flat_payload_field(patch, payload, "retargeted_support_binding_claim", "binding_claim", "text")
        _copy_flat_payload_field(
            patch,
            payload,
            "retargeted_support_bound_answer_option",
            "binding_claim",
            "bound_answer_option",
        )
        _copy_flat_payload_field(patch, payload, "retargeted_support_inference", "inference", "text")
        if patch:
            items.append({"intervention_type": "retargeted_support", "component_patch": patch})
    if "wrong_inference" in templates_by_type:
        patch = {}
        _copy_flat_payload_field(patch, payload, "wrong_inference_inference", "inference", "text")
        if patch:
            items.append({"intervention_type": "wrong_inference", "component_patch": patch})
    if "wrong_principle" in templates_by_type:
        patch = {}
        _copy_flat_payload_field(
            patch,
            payload,
            "wrong_principle_scientific_principle",
            "scientific_principle",
            "text",
        )
        _copy_flat_payload_field(patch, payload, "wrong_principle_principle", "scientific_principle", "text")
        if patch:
            items.append({"intervention_type": "wrong_principle", "component_patch": patch})
    if "delete_binding_claim" in templates_by_type:
        items.append({"intervention_type": "delete_binding_claim", "component_patch": {}})
    return items


def _copy_flat_payload_field(
    patch: dict[str, Any],
    payload: dict[str, Any],
    source_key: str,
    target_component: str,
    target_key: str,
) -> None:
    value = payload.get(source_key)
    if isinstance(value, str) and value.strip():
        patch.setdefault(target_component, {})[target_key] = value.strip()


def _structured_trace_from_model_payload(template_trace: StructuredTrace, trace_payload: dict[str, Any]) -> StructuredTrace:
    full_payload = template_trace.model_dump(mode="json")
    _deep_update(full_payload, trace_payload)
    return StructuredTrace.model_validate(full_payload)


def _component_patch_from_model_item(item: dict[str, Any]) -> dict[str, Any]:
    patch = item.get("component_patch") or item.get("intervened_trace") or item.get("patch")
    if isinstance(patch, dict):
        return patch

    compact_patch: dict[str, Any] = {}
    _copy_compact_text_field(
        compact_patch,
        item,
        target_component="binding_claim",
        source_keys=("binding_claim_text", "binding_text"),
        target_key="text",
    )
    _copy_compact_text_field(
        compact_patch,
        item,
        target_component="scientific_principle",
        source_keys=("scientific_principle_text", "principle_text"),
        target_key="text",
    )
    _copy_compact_text_field(
        compact_patch,
        item,
        target_component="inference",
        source_keys=("inference_text",),
        target_key="text",
    )
    _copy_compact_text_field(
        compact_patch,
        item,
        target_component="binding_claim",
        source_keys=("bound_answer_option", "binding_answer_option"),
        target_key="bound_answer_option",
    )
    return compact_patch


def _copy_compact_text_field(
    patch: dict[str, Any],
    item: dict[str, Any],
    *,
    target_component: str,
    source_keys: tuple[str, ...],
    target_key: str,
) -> None:
    for source_key in source_keys:
        value = item.get(source_key)
        if isinstance(value, str) and value.strip():
            patch.setdefault(target_component, {})[target_key] = value.strip()
            return


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def validation_rows(records: list[IntervenedTrace], answer_choices_by_example: dict[str, dict[str, str]]) -> list[InterventionValidationRow]:
    return [_validation_row(record, answer_choices_by_example.get(record.example_id, {})) for record in records]


def _record(
    intervention_id: str,
    example_id: str,
    original_trace: StructuredTrace,
    intervened_trace: StructuredTrace,
    intervention_type: str,
    target_component: Component,
    expected_label: ExpectedLabel = ExpectedLabel.INTERVENED,
    expected_flawed_component: Component = Component.UNKNOWN,
    known_location: bool = True,
    details: dict[str, Any] | None = None,
) -> IntervenedTrace:
    return IntervenedTrace(
        intervention_id=intervention_id,
        example_id=example_id,
        original_trace=original_trace,
        intervened_trace=intervened_trace,
        intervention=InterventionMetadata(
            intervention_type=intervention_type,
            target_component=target_component,
            expected_label=expected_label,
            expected_flawed_component=expected_flawed_component,
            known_location=known_location,
            details=details or {},
        ),
    )


def _validation_row(record: IntervenedTrace, choices: dict[str, str]) -> InterventionValidationRow:
    details = record.intervention.details
    checks = _validate_record(record, choices)
    return InterventionValidationRow(
        example_id=record.example_id,
        intervention_id=record.intervention_id,
        intervention_type=record.intervention.intervention_type,
        operation=str(details.get("operation", "")),
        changed_field=str(details.get("changed_field", "")),
        old_value=str(details.get("old_value", "")),
        new_value=str(details.get("new_value", "")),
        unchanged_fields=str(details.get("unchanged_fields", "")),
        is_flawed=bool(details.get("is_flawed", record.label_is_intervened)),
        auto_validation_pass=all(checks.values()),
        validation_notes="; ".join(name for name, passed in checks.items() if not passed),
        human_validation_status="",
    )


def _validate_record(record: IntervenedTrace, choices: dict[str, str]) -> dict[str, bool]:
    original_trace = record.original_trace
    changed_trace = record.intervened_trace
    intervention_type = record.intervention.intervention_type
    checks = {
        "target_prompt_masks_answer_tail": True,
        "intended_field_changed": True,
        "preserved_evidence": True,
        "preserved_principle": True,
        "valid_distractor": True,
        "paraphrase_preserves_ids": True,
    }
    if intervention_type == "wrong_binding":
        old_option = original_trace.binding_claim.bound_answer_option or original_trace.final_answer
        new_option = changed_trace.binding_claim.bound_answer_option
        checks["intended_field_changed"] = new_option != old_option and changed_trace.binding_claim.text != original_trace.binding_claim.text
        checks["preserved_evidence"] = changed_trace.visual_evidence == original_trace.visual_evidence and changed_trace.textual_evidence == original_trace.textual_evidence
        checks["preserved_principle"] = changed_trace.scientific_principle == original_trace.scientific_principle
        checks["valid_distractor"] = bool(new_option) and new_option in choices and new_option != old_option
    elif intervention_type == "retargeted_support":
        old_option = original_trace.binding_claim.bound_answer_option or original_trace.final_answer
        new_option = changed_trace.binding_claim.bound_answer_option
        checks["intended_field_changed"] = (
            new_option != old_option
            and changed_trace.binding_claim.text != original_trace.binding_claim.text
            and changed_trace.inference.text != original_trace.inference.text
        )
        checks["preserved_evidence"] = changed_trace.visual_evidence == original_trace.visual_evidence and changed_trace.textual_evidence == original_trace.textual_evidence
        checks["preserved_principle"] = changed_trace.scientific_principle == original_trace.scientific_principle
        checks["preserved_answer_tail"] = (
            changed_trace.conclusion == original_trace.conclusion
            and changed_trace.final_answer == original_trace.final_answer
        )
        checks["preserved_support_ids"] = (
            changed_trace.binding_claim.id == original_trace.binding_claim.id
            and changed_trace.binding_claim.supports == original_trace.binding_claim.supports
            and changed_trace.inference.id == original_trace.inference.id
            and changed_trace.inference.supports == original_trace.inference.supports
        )
        checks["valid_distractor"] = bool(new_option) and new_option in choices and new_option != old_option
    elif intervention_type == "paraphrase":
        checks["intended_field_changed"] = changed_trace.binding_claim.text != original_trace.binding_claim.text
        checks["paraphrase_preserves_ids"] = (
            changed_trace.binding_claim.id == original_trace.binding_claim.id
            and changed_trace.binding_claim.supports == original_trace.binding_claim.supports
        )
    elif intervention_type == "delete_binding_claim":
        checks["intended_field_changed"] = changed_trace.binding_claim.text == OMITTED
        checks["preserved_evidence"] = changed_trace.visual_evidence == original_trace.visual_evidence and changed_trace.textual_evidence == original_trace.textual_evidence
        checks["preserved_principle"] = changed_trace.scientific_principle == original_trace.scientific_principle
    elif intervention_type == "wrong_visual":
        checks["intended_field_changed"] = (
            bool(original_trace.visual_evidence)
            and bool(changed_trace.visual_evidence)
            and changed_trace.visual_evidence[0].content != original_trace.visual_evidence[0].content
        )
        checks["preserved_evidence"] = changed_trace.textual_evidence == original_trace.textual_evidence
        checks["preserved_principle"] = changed_trace.scientific_principle == original_trace.scientific_principle
        checks["preserved_binding"] = changed_trace.binding_claim == original_trace.binding_claim
        checks["preserved_inference"] = changed_trace.inference == original_trace.inference
        checks["preserved_answer_tail"] = (
            changed_trace.conclusion == original_trace.conclusion
            and changed_trace.final_answer == original_trace.final_answer
        )
        checks["preserved_support_ids"] = (
            [item.id for item in changed_trace.visual_evidence] == [item.id for item in original_trace.visual_evidence]
            and changed_trace.binding_claim.supports == original_trace.binding_claim.supports
        )
    elif intervention_type == "wrong_textual":
        checks["intended_field_changed"] = (
            bool(original_trace.textual_evidence)
            and bool(changed_trace.textual_evidence)
            and changed_trace.textual_evidence[0].content != original_trace.textual_evidence[0].content
        )
        checks["preserved_evidence"] = changed_trace.visual_evidence == original_trace.visual_evidence
        checks["preserved_principle"] = changed_trace.scientific_principle == original_trace.scientific_principle
        checks["preserved_binding"] = changed_trace.binding_claim == original_trace.binding_claim
        checks["preserved_inference"] = changed_trace.inference == original_trace.inference
        checks["preserved_answer_tail"] = (
            changed_trace.conclusion == original_trace.conclusion
            and changed_trace.final_answer == original_trace.final_answer
        )
        checks["preserved_support_ids"] = (
            [item.id for item in changed_trace.textual_evidence] == [item.id for item in original_trace.textual_evidence]
            and changed_trace.binding_claim.supports == original_trace.binding_claim.supports
        )
    return checks


def _bound_or_reference_answer(trace: StructuredTrace, reference_answer: str | None) -> str | None:
    return trace.binding_claim.bound_answer_option or reference_answer or trace.final_answer


def _choose_distractor(choices: dict[str, str], old_option: str | None) -> str:
    normalized_old = old_option.strip().upper() if old_option else None
    for label in sorted(choices):
        if label != normalized_old:
            return label
    for label in ["A", "B", "C", "D"]:
        if label != normalized_old:
            return label
    return "UNKNOWN"


def _format_option(label: str | None, choices: dict[str, str] | None) -> str:
    if not label:
        return "unknown"
    text = (choices or {}).get(label)
    return f"{label}: {text}" if text else label


def _find_principle_donor(
    example_id: str,
    traces_by_example: dict[str, StructuredTrace],
    metadata_by_id: dict[str, dict[str, Any]],
) -> tuple[str, StructuredTrace] | None:
    source_metadata = metadata_by_id.get(example_id, {})
    candidates = [(candidate_id, trace) for candidate_id, trace in traces_by_example.items() if candidate_id != example_id]
    if not candidates:
        return None

    def score(candidate_id: str) -> tuple[int, str]:
        candidate_metadata = metadata_by_id.get(candidate_id, {})
        return (
            int(candidate_metadata.get("subject") == source_metadata.get("subject"))
            + int(candidate_metadata.get("topic") == source_metadata.get("topic"))
            + int(candidate_metadata.get("category") == source_metadata.get("category"))
            + int(candidate_metadata.get("skill") == source_metadata.get("skill")),
            candidate_id,
        )

    return max(candidates, key=lambda item: score(item[0]))


def _find_evidence_donor(
    example_id: str,
    traces_by_example: dict[str, StructuredTrace],
    metadata_by_id: dict[str, dict[str, Any]],
    component: Component,
) -> tuple[str, StructuredTrace] | None:
    candidates: list[tuple[str, StructuredTrace]] = []
    for candidate_id, trace in traces_by_example.items():
        if candidate_id == example_id:
            continue
        if component == Component.VISUAL_EVIDENCE and trace.visual_evidence:
            candidates.append((candidate_id, trace))
        elif component == Component.TEXTUAL_EVIDENCE and trace.textual_evidence:
            candidates.append((candidate_id, trace))
    if not candidates:
        return None
    source_metadata = metadata_by_id.get(example_id, {})

    def score(candidate_id: str) -> tuple[int, int, int, int, int, str]:
        candidate_metadata = metadata_by_id.get(candidate_id, {})
        return (
            int(candidate_metadata.get("dataset") == source_metadata.get("dataset")),
            int(candidate_metadata.get("topic") == source_metadata.get("topic")),
            int(candidate_metadata.get("skill") == source_metadata.get("skill")),
            int(candidate_metadata.get("task") == source_metadata.get("task")),
            int(candidate_metadata.get("subject") == source_metadata.get("subject")),
            candidate_id,
        )

    return max(candidates, key=lambda item: score(item[0]))
