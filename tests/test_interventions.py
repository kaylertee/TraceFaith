from aria.interventions import (
    build_model_interventions_from_output,
    build_faithfulness_interventions,
    delete_component,
    original,
    paraphrase,
    retargeted_support,
    wrong_binding,
    wrong_textual,
    wrong_visual,
    wrong_inference,
    wrong_principle,
)
from aria.schemas import Component, InterventionModelOutput, StructuredTrace


def example_trace() -> StructuredTrace:
    return StructuredTrace(
        visual_evidence=[
            {
                "modality": "image",
                "content": "The object casts a longer shadow when the light source is lower.",
                "provenance": "diagram shadow",
            }
        ],
        textual_evidence=[
            {"modality": "text", "content": "The question asks which light angle makes a longer shadow."}
        ],
        binding_claim={
            "text": "The lower light source in the diagram is the condition named in the question.",
            "bound_answer_option": "B",
        },
        scientific_principle={"text": "Light travels in straight lines, so lower light angles create longer shadows."},
        inference={"text": "Because the light source is lower, the shadow should become longer."},
        conclusion="The observation supports the option saying the shadow is longer.",
        final_answer="B",
    )


def test_wrong_binding_labels_component_and_validity_metadata() -> None:
    trace = example_trace()
    record = wrong_binding(trace, "example", choices={"A": "high light", "B": "low light"}, reference_answer="B")

    assert record.label_is_intervened is True
    assert record.expected_flawed_component == Component.BINDING_CLAIM
    assert record.intervened_trace.binding_claim.bound_answer_option == "A"
    assert record.intervened_trace.visual_evidence == trace.visual_evidence
    assert record.intervened_trace.scientific_principle == trace.scientific_principle
    assert record.intervention.details["validity"]["intended_fault_type"] == "wrong_binding"


def test_retargeted_support_retargets_binding_and_inference_to_distractor() -> None:
    trace = example_trace()
    record = retargeted_support(trace, "example", choices={"A": "high light", "B": "low light"}, reference_answer="B")

    assert record.intervention.intervention_type == "retargeted_support"
    assert record.intervened_trace.binding_claim.bound_answer_option == "A"
    assert record.intervened_trace.binding_claim.text != trace.binding_claim.text
    assert record.intervened_trace.inference.text != trace.inference.text
    assert "retarget" not in record.intervened_trace.binding_claim.text.lower()
    assert "retarget" not in record.intervened_trace.inference.text.lower()
    assert record.intervened_trace.visual_evidence == trace.visual_evidence
    assert record.intervened_trace.textual_evidence == trace.textual_evidence
    assert record.intervened_trace.scientific_principle == trace.scientific_principle
    assert record.intervention.details["operation"] == "answer_option_support_retarget"
    assert record.intervention.details["target_distractor"] == "A"


def test_wrong_inference_mutates_only_inference() -> None:
    trace = example_trace()
    record = wrong_inference(trace, "example", choices={"A": "high light", "B": "low light"}, reference_answer="B")

    assert record.intervened_trace.inference.text != trace.inference.text
    assert record.intervened_trace.textual_evidence == trace.textual_evidence
    assert record.intervened_trace.binding_claim == trace.binding_claim
    assert record.expected_flawed_component == Component.INFERENCE


def test_wrong_principle_uses_donor_metadata() -> None:
    trace = example_trace()
    donor = StructuredTrace(
        visual_evidence=[{"modality": "image", "content": "A donor visual fact."}],
        textual_evidence=[{"modality": "text", "content": "A donor text fact."}],
        binding_claim={"text": "A donor binding.", "bound_answer_option": "A"},
        scientific_principle={"text": "A donor principle."},
        inference={"text": "A donor inference."},
        conclusion="The donor supports A.",
        final_answer="A",
    )
    record = wrong_principle(trace, "example", donor_trace=donor, donor_example_id="donor")

    assert record.intervened_trace.scientific_principle.text == "A donor principle."
    assert record.expected_flawed_component == Component.SCIENTIFIC_PRINCIPLE
    assert record.intervention.details["donor_example_id"] == "donor"


def test_wrong_visual_mutates_only_visual_evidence() -> None:
    trace = example_trace()
    donor = example_trace()
    donor.visual_evidence[0].content = "A donor visual fact."
    record = wrong_visual(trace, "example", donor_trace=donor, donor_example_id="donor")

    assert record is not None
    assert record.intervened_trace.visual_evidence[0].content == "A donor visual fact."
    assert record.intervened_trace.visual_evidence[0].id == trace.visual_evidence[0].id
    assert record.intervened_trace.textual_evidence == trace.textual_evidence
    assert record.intervened_trace.binding_claim == trace.binding_claim
    assert record.intervened_trace.scientific_principle == trace.scientific_principle
    assert record.expected_flawed_component == Component.VISUAL_EVIDENCE


def test_wrong_textual_mutates_only_textual_evidence() -> None:
    trace = example_trace()
    donor = example_trace()
    donor.textual_evidence[0].content = "A donor textual fact."
    record = wrong_textual(trace, "example", donor_trace=donor, donor_example_id="donor")

    assert record is not None
    assert record.intervened_trace.textual_evidence[0].content == "A donor textual fact."
    assert record.intervened_trace.textual_evidence[0].id == trace.textual_evidence[0].id
    assert record.intervened_trace.visual_evidence == trace.visual_evidence
    assert record.intervened_trace.binding_claim == trace.binding_claim
    assert record.intervened_trace.scientific_principle == trace.scientific_principle
    assert record.expected_flawed_component == Component.TEXTUAL_EVIDENCE


def test_delete_component_omits_binding_claim() -> None:
    record = delete_component(example_trace(), "example", Component.BINDING_CLAIM)

    assert record.intervened_trace.binding_claim.text == "[omitted]"
    assert record.expected_flawed_component == Component.BINDING_CLAIM


def test_paraphrase_control_is_labelled_original() -> None:
    record = paraphrase(example_trace(), "example")

    assert record.label_is_intervened is False
    assert record.expected_flawed_component == Component.NONE
    assert record.intervention.details["negative_control"] is True


def test_build_faithfulness_interventions_uses_new_family() -> None:
    records = build_faithfulness_interventions({"example": example_trace(), "donor": example_trace()})
    intervention_types = {record.intervention.intervention_type for record in records}

    assert "wrong_binding" in intervention_types
    assert "paraphrase" in intervention_types
    assert "wrong_principle" in intervention_types
    assert "delete_binding_claim" in intervention_types
    assert sum(record.intervention.intervention_type == "original_trace" for record in records) == 2


def test_build_faithfulness_interventions_can_scope_examples_while_using_donors() -> None:
    records = build_faithfulness_interventions(
        {"example": example_trace(), "donor": example_trace()},
        selected_example_ids={"example"},
    )

    assert {record.example_id for record in records} == {"example"}
    assert any(record.intervention.intervention_type == "wrong_principle" for record in records)


def test_build_faithfulness_interventions_can_include_modality_family() -> None:
    records = build_faithfulness_interventions(
        {"example": example_trace(), "donor": example_trace()},
        selected_example_ids={"example"},
        include_modality_interventions=True,
    )
    intervention_types = {record.intervention.intervention_type for record in records}

    assert "wrong_visual" in intervention_types
    assert "wrong_textual" in intervention_types


def test_build_faithfulness_interventions_can_include_retargeted_support() -> None:
    records = build_faithfulness_interventions(
        {"example": example_trace(), "donor": example_trace()},
        answer_choices_by_example={
            "example": {"A": "high light", "B": "low light"},
            "donor": {"A": "high light", "B": "low light"},
        },
        reference_answers_by_example={"example": "B", "donor": "B"},
        selected_example_ids={"example"},
        include_retargeted_support=True,
    )
    retargeted = [record for record in records if record.intervention.intervention_type == "retargeted_support"]

    assert len(retargeted) == 1
    assert retargeted[0].intervention.details["target_distractor"] == "A"


def test_model_intervention_output_can_patch_only_changed_components() -> None:
    template_records = build_faithfulness_interventions(
        {"example": example_trace()},
        answer_choices_by_example={"example": {"A": "high light", "B": "low light"}},
        reference_answers_by_example={"example": "B"},
        selected_example_ids={"example"},
        include_retargeted_support=True,
    )
    output = InterventionModelOutput(
        example_id="example",
        intervention_model="intervention-model",
        raw_output="""{
          "paraphrase_binding_claim": "The same lower-light condition is restated in different words.",
          "wrong_binding_binding_claim": "The cited evidence identifies answer option A as the condition described by the question.",
          "wrong_binding_bound_answer_option": "A",
          "retargeted_support_binding_claim": "The visual and textual evidence together support answer option A as the relevant condition.",
          "retargeted_support_bound_answer_option": "A",
          "retargeted_support_inference": "Using the stated principle, the observed condition is most consistent with answer option A.",
          "wrong_inference_inference": "Using the stated principle, the observed condition is most consistent with answer option A."
        }""",
    )

    records = build_model_interventions_from_output(output, template_records)
    paraphrase_record = next(record for record in records if record.intervention.intervention_type == "paraphrase")

    assert paraphrase_record.intervened_trace.binding_claim.text.startswith("The same lower-light")
    assert paraphrase_record.intervened_trace.scientific_principle == example_trace().scientific_principle
    assert paraphrase_record.intervened_trace.inference == example_trace().inference
    assert paraphrase_record.intervened_trace.conclusion == example_trace().conclusion
    delete_record = next(record for record in records if record.intervention.intervention_type == "delete_binding_claim")
    assert delete_record.intervened_trace.binding_claim.text == "[omitted]"
