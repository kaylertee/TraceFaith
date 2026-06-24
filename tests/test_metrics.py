from aria.interventions import original, paraphrase, wrong_binding, wrong_inference
from aria.metrics import (
    answer_flip_rate,
    intervention_detection_accuracy,
    judge_summary,
    localization_accuracy,
    mediation_summary,
    target_effect_rows,
)
from aria.schemas import JudgeOutput, StructuredTrace, TargetOutput


def example_trace() -> StructuredTrace:
    return StructuredTrace(
        visual_evidence=[{"modality": "image", "content": "The lower light casts a longer shadow."}],
        textual_evidence=[{"modality": "text", "content": "The question asks which shadow is longer."}],
        binding_claim="The lower light in the image is the condition asked about in the text.",
        scientific_principle="Lower light angles create longer shadows.",
        inference="The lower light should cast the longer shadow.",
        conclusion="The reasoning supports answer B.",
        final_answer="B",
    )


def test_intervention_detection_accuracy() -> None:
    trace = example_trace()
    records = [original(trace, "example"), wrong_inference(trace, "example", "Wrong inference.")]

    assert intervention_detection_accuracy(records, [False, True]) == 1.0


def test_localization_accuracy_ignores_original_records() -> None:
    trace = example_trace()
    records = [original(trace, "example"), wrong_inference(trace, "example", "Wrong inference.")]

    assert localization_accuracy(records, [None, "inference"]) == 1.0


def test_answer_flip_rate() -> None:
    assert answer_flip_rate(["A", "B", "C"], ["A", "D", "C"]) == 1 / 3


def test_judge_summary() -> None:
    trace = example_trace()
    records = [original(trace, "example"), wrong_inference(trace, "example", "Wrong inference.")]
    outputs = [
        JudgeOutput(
            intervention_id=records[0].intervention_id,
            judge_model="test-judge",
            faithfulness_score=5,
            is_intervened_or_flawed=False,
            flawed_component="none",
        ),
        JudgeOutput(
            intervention_id=records[1].intervention_id,
            judge_model="test-judge",
            faithfulness_score=2,
            is_intervened_or_flawed=True,
            flawed_component="inference",
        ),
    ]

    summary = judge_summary(records, outputs)

    assert summary["intervention_detection_accuracy"] == 1.0
    assert summary["localization_accuracy"] == 1.0
    assert summary["judge_score_drop_original_to_intervened"] == 3


def test_mediation_summary_and_target_effect_rows() -> None:
    trace = example_trace()
    records = [
        original(trace, "example"),
        wrong_binding(trace, "example", "Wrong binding."),
        paraphrase(trace, "example"),
    ]
    target_outputs = [
        TargetOutput(intervention_id=records[0].intervention_id, target_model="target", final_answer="B", is_correct=True),
        TargetOutput(intervention_id=records[1].intervention_id, target_model="target", final_answer="A", is_correct=False),
        TargetOutput(intervention_id=records[2].intervention_id, target_model="target", final_answer="B", is_correct=True),
    ]
    judge_outputs = [
        JudgeOutput(
            intervention_id=records[0].intervention_id,
            judge_model="judge",
            faithfulness_score=5,
            is_intervened_or_flawed=False,
            flawed_component="none",
        ),
        JudgeOutput(
            intervention_id=records[1].intervention_id,
            judge_model="judge",
            faithfulness_score=1,
            is_intervened_or_flawed=True,
            flawed_component="binding_claim",
        ),
        JudgeOutput(
            intervention_id=records[2].intervention_id,
            judge_model="judge",
            faithfulness_score=5,
            is_intervened_or_flawed=False,
            flawed_component="none",
        ),
    ]

    rows = target_effect_rows(records, target_outputs, judge_outputs)
    summary = mediation_summary(records, target_outputs, judge_outputs)
    binding_row = next(row for row in rows if row["intervention_type"] == "wrong_binding")

    assert binding_row["answer_flipped"] == 1.0
    assert summary["answer_flip_rate_wrong_binding"] == 1.0
    assert summary["answer_flip_rate_paraphrase"] == 0.0
    assert summary["primary_binding_vs_paraphrase_flip_delta"] == 1.0
    assert summary["necessity_score"] == 1.0
    assert "corrupted_but_still_correct_rate" in summary
    assert "trace_bypass_rate" not in summary


def test_tracefaith_net_effect_metrics_count_abstention() -> None:
    trace = example_trace()
    records = [
        original(trace, "example"),
        paraphrase(trace, "example"),
        wrong_binding(trace, "example", choices={"A": "wrong", "B": "right"}, reference_answer="B"),
    ]
    target_outputs = [
        TargetOutput(intervention_id=records[0].intervention_id, target_model="target", final_answer="B", is_correct=True),
        TargetOutput(intervention_id=records[1].intervention_id, target_model="target", final_answer="B", is_correct=True),
        TargetOutput(intervention_id=records[2].intervention_id, target_model="target", final_answer="n/a", is_correct=False),
    ]

    rows = target_effect_rows(records, target_outputs)
    summary = mediation_summary(records, target_outputs)
    binding_row = next(row for row in rows if row["intervention_type"] == "wrong_binding")

    assert binding_row["is_abstention"] == 1.0
    assert binding_row["unsupportedness_detected"] == 1.0
    assert binding_row["answer_flipped"] == 1.0
    assert binding_row["answer_category"] == "abstention"
    assert summary["net_abstention_effect"] == 1.0
    assert summary["net_flip_effect"] == 1.0
    assert summary["net_correctness_drop"] == 1.0
