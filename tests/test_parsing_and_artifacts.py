from pathlib import Path

import pytest
from pydantic import ValidationError

from aria.hf_models import _extract_answer, _parse_and_sanitize_intervention_payload, _parse_and_validate_intervention_payload
from aria.io import read_jsonl
from aria.parsing import parse_intervention_payload, parse_judge_output, parse_structured_trace
from aria.prompts import render_judge_prompt, render_target_rerun_prompt, render_trace_generation_prompt, render_trace_repair_prompt
from aria.prompts import render_intervention_generation_prompt
from scripts.build_supervisor_docx import _original_traces_path
from aria.schemas import (
    Component,
    Example,
    ExpectedLabel,
    GeneratedTraceRecord,
    IntervenedTrace,
    InterventionMetadata,
    StructuredTrace,
)


def test_parse_structured_trace_repairs_code_fence_and_trailing_text() -> None:
    text = """```json
{
  "visual_evidence": [{"modality": "image", "content": "The glass has droplets.", "provenance": "image"}],
  "textual_evidence": [{"modality": "text", "content": "The glass is cold.", "provenance": "question"}],
  "binding_claim": "The droplets on the cold glass are the observation to explain.",
  "scientific_principle": "Condensation changes gas to liquid.",
  "inference": "Water vapor becomes droplets on the cold glass.",
  "conclusion": "The reasoning supports answer B.",
  "final_answer": "B"
}
```
extra text from model
"""

    trace = parse_structured_trace(text)

    assert trace.final_answer == "B"
    assert trace.textual_evidence[0].content == "The glass is cold."
    assert trace.visual_evidence[0].provenance == "image"


def test_parse_structured_trace_normalizes_verbose_choice_label() -> None:
    text = """{
      "visual_evidence": [{"modality": "image", "content": "The figure shows weather."}],
      "textual_evidence": [{"modality": "text", "content": "The question asks for current conditions."}],
      "binding_claim": "The figure condition is the current atmospheric condition asked about.",
      "scientific_principle": "Weather is short-term atmospheric condition.",
      "inference": "The passage describes current conditions.",
      "conclusion": "The reasoning supports answer A.",
      "final_answer": "A. weather"
    }"""

    trace = parse_structured_trace(text)

    assert trace.final_answer == "A"


def test_parse_structured_trace_repairs_invalid_json_backslash_escape() -> None:
    text = r"""{
      "visual_evidence": [{"modality": "image", "content": "The symbol \\theta is shown with an invalid \escape."}],
      "textual_evidence": [{"modality": "text", "content": "The question asks about the angle."}],
      "binding_claim": "The visual angle is the queried value.",
      "scientific_principle": "Angles can be represented by theta.",
      "inference": "Use the displayed angle.",
      "conclusion": "The reasoning supports answer A.",
      "final_answer": "A"
    }"""

    trace = parse_structured_trace(text)

    assert trace.final_answer == "A"
    assert "invalid" in trace.visual_evidence[0].content


def test_parse_intervention_payload_uses_first_complete_json_object() -> None:
    text = """{
      "interventions": [
        {
          "intervention_type": "paraphrase",
          "intervened_trace": {
            "visual_evidence": [{"modality": "image", "content": "Same visual evidence."}],
            "textual_evidence": [{"modality": "text", "content": "Same textual evidence."}],
            "binding_claim": "Same binding.",
            "scientific_principle": "Same principle.",
            "inference": "Same inference.",
            "conclusion": "Same conclusion.",
            "final_answer": "A"
          }
        }
      ]
    }
    {"extra": "model accidentally emitted a second object"}"""

    payload = parse_intervention_payload(text)

    assert payload["interventions"][0]["intervention_type"] == "paraphrase"


def test_parse_intervention_payload_accepts_top_level_list_with_intro_text() -> None:
    text = """Here are the intervention traces in JSON format:

    [
      {
        "intervention_type": "paraphrase",
        "intervened_trace": {
          "visual_evidence": [{"modality": "image", "content": "Same visual evidence."}],
          "textual_evidence": [{"modality": "text", "content": "Same textual evidence."}],
          "binding_claim": "Same binding.",
          "scientific_principle": "Same principle.",
          "inference": "Same inference.",
          "conclusion": "Same conclusion.",
          "final_answer": "A"
        }
      }
    ]
    """

    payload = parse_intervention_payload(text)

    assert payload["interventions"][0]["intervention_type"] == "paraphrase"


def test_parse_intervention_payload_accepts_flat_component_dictionary() -> None:
    text = """{
      "paraphrase_binding_claim": "Same meaning in different words.",
      "wrong_binding_binding_claim": "The evidence supports option A.",
      "wrong_binding_bound_answer_option": "A",
      "wrong_inference_inference": "The stated principle supports option A."
    }"""

    payload = parse_intervention_payload(text)

    assert payload["wrong_binding_bound_answer_option"] == "A"
    assert "interventions" not in payload


def test_parse_structured_trace_drops_empty_evidence_entries() -> None:
    text = """{
      "visual_evidence": [{"modality": "image", "content": "The figure shows weather."}],
      "textual_evidence": [{"modality": "text", "content": ""}],
      "binding_claim": "The figure condition is the current atmospheric condition asked about.",
      "scientific_principle": "Weather is short-term atmospheric condition.",
      "inference": "The passage describes current conditions.",
      "conclusion": "The reasoning supports answer A.",
      "final_answer": "A"
    }"""

    trace = parse_structured_trace(text)

    assert len(trace.visual_evidence) == 1
    assert trace.textual_evidence == []


def test_parse_structured_trace_normalizes_schema_alternative_modality_values() -> None:
    text = """{
      "visual_evidence": [{"modality": "image|diagram", "content": "The figure shows weather."}],
      "textual_evidence": [{"modality": "text|unknown", "content": "The question asks about weather."}],
      "binding_claim": "The figure condition is the current atmospheric condition asked about.",
      "scientific_principle": "Weather is short-term atmospheric condition.",
      "inference": "The passage describes current conditions.",
      "conclusion": "The reasoning supports answer A.",
      "final_answer": "A"
    }"""

    trace = parse_structured_trace(text)

    assert trace.visual_evidence[0].modality == "image"
    assert trace.textual_evidence[0].modality == "text"


def test_parse_structured_trace_does_not_semantically_repair_missing_fields() -> None:
    text = """{"scientific_principle": "A principle", "final_answer": "A"}"""

    with pytest.raises(ValidationError):
        parse_structured_trace(text)


def test_parse_judge_output_normalizes_schema_values_only() -> None:
    text = """{
      "faithfulness_score": 0.5,
      "is_intervened_or_flawed": true,
      "flawed_component": "inference|conclusion|final_answer",
      "missing_step": false,
      "final_answer_supported": false,
      "short_reason": "Multiple fields look inconsistent."
    }"""

    output = parse_judge_output(text, intervention_id="x", judge_model="judge")

    assert output.faithfulness_score == 1
    assert output.flawed_component == "unknown"
    assert output.parse_error is None


def test_judge_prompt_excludes_trace_metadata() -> None:
    trace = StructuredTrace(
        visual_evidence=[{"modality": "image", "content": "Observed visual fact."}],
        textual_evidence=[{"modality": "text", "content": "Observed textual fact."}],
        binding_claim="Corrupted binding.",
        scientific_principle="Corrupted principle.",
        inference="Corrupted inference.",
        conclusion="Corrupted conclusion.",
        final_answer="A",
        metadata={"raw_output": "original trace metadata should not appear"},
    )
    record = IntervenedTrace(
        intervention_id="x::wrong_principle",
        example_id="x",
        original_trace=trace,
        intervened_trace=trace,
        intervention=InterventionMetadata(
            intervention_type="wrong_principle",
            target_component=Component.SCIENTIFIC_PRINCIPLE,
            expected_label=ExpectedLabel.INTERVENED,
            expected_flawed_component=Component.SCIENTIFIC_PRINCIPLE,
        ),
    )
    example = Example(example_id="x", question="Which option?", choices={"A": "Alpha"}, correct_answer="A")

    prompt = render_judge_prompt(record, example)

    assert "original trace metadata should not appear" not in prompt
    assert "Corrupted principle." in prompt
    assert "Corrupted binding." in prompt


def test_target_rerun_prompt_uses_trace_primary_wording_without_answer_leakage() -> None:
    trace = StructuredTrace(
        visual_evidence=[{"id": "V1", "modality": "image", "content": "Observed visual fact."}],
        textual_evidence=[{"id": "T1", "modality": "text", "content": "Observed textual fact."}],
        binding_claim={
            "id": "B1",
            "text": "The evidence supports the relevant option.",
            "uses_visual": ["V1"],
            "uses_textual": ["T1"],
            "supports": ["V1", "T1"],
        },
        scientific_principle={"id": "P1", "text": "Scientific rule.", "supports": ["T1"]},
        inference={"id": "I1", "text": "Apply the rule.", "supports": ["B1", "P1"]},
        conclusion="This conclusion should not appear in target support traces.",
        final_answer="C",
    )
    example = Example(example_id="x", question="Which option?", choices={"A": "Alpha", "C": "Charlie"}, correct_answer="C")

    prompt = render_target_rerun_prompt(trace, example)

    assert "support trace as the primary reasoning basis" in prompt
    assert "Do not ignore the trace" in prompt
    assert "Output only the answer option" in prompt
    assert "This conclusion should not appear" not in prompt
    assert '"final_answer"' not in prompt


def test_no_trace_target_prompt_is_distinct_from_trace_primary_prompt() -> None:
    example = Example(example_id="x", question="Which option?", choices={"A": "Alpha"}, correct_answer="A")

    prompt = render_target_rerun_prompt(None, example, condition="no_trace")

    assert "No reasoning trace is provided" in prompt
    assert "provided support trace as the primary reasoning basis" not in prompt
    assert "Structured trace:" not in prompt


def test_trace_generation_prompt_still_requests_original_structured_trace() -> None:
    example = Example(example_id="x", question="Which option?", choices={"A": "Alpha"}, correct_answer="A")

    prompt = render_trace_generation_prompt(example)

    assert "Required JSON schema" in prompt
    assert '"visual_evidence"' in prompt
    assert '"binding_claim"' in prompt
    assert '"final_answer"' in prompt
    assert "Output only the answer option" not in prompt


def test_intervention_generation_prompt_requests_compact_component_patches() -> None:
    example = Example(example_id="x", question="Which option?", choices={"A": "Alpha"}, correct_answer="A")
    trace = StructuredTrace(
        visual_evidence=[{"modality": "image", "content": "Observed visual fact."}],
        textual_evidence=[{"modality": "text", "content": "Observed textual fact."}],
        binding_claim={"text": "Observed binding.", "bound_answer_option": "A"},
        scientific_principle={"text": "Scientific rule."},
        inference={"text": "Apply the rule."},
        conclusion="The reasoning supports A.",
        final_answer="A",
    )

    prompt = render_intervention_generation_prompt(
        example,
        trace,
        [{"intervention_type": "wrong_inference", "target_component": "inference"}],
    )

    assert '"wrong_inference_inference"' in prompt
    assert '"retargeted_support_binding_claim"' in prompt
    assert "Do not re-output the full trace" in prompt
    assert "Output only keys for requested interventions" in prompt
    assert "Do not output notes" in prompt
    assert "Do not output intervention_type" in prompt
    assert "Do not output nested trace JSON" in prompt
    assert "misapplied" in prompt
    assert '"component_patch"' not in prompt
    assert '"intervened_trace"' not in prompt


def test_trace_repair_prompt_requests_valid_json_only() -> None:
    example = Example(example_id="x", question="Which option?", choices={"A": "Alpha"}, correct_answer="A")

    prompt = render_trace_repair_prompt(example, '{"visual_evidence": [', "JSONDecodeError")

    assert "Repair the model output into valid JSON" in prompt
    assert "Return only one JSON object" in prompt
    assert "JSONDecodeError" in prompt
    assert '{"visual_evidence": [' in prompt


def test_example_accepts_legacy_gold_answer_alias() -> None:
    example = Example.model_validate(
        {"example_id": "x", "question": "Which option?", "choices": {"A": "Alpha"}, "gold_answer": "a"}
    )

    assert example.correct_answer == "A"
    assert "gold_answer" not in example.model_dump(mode="json")
    assert example.model_dump(mode="json")["correct_answer"] == "A"


def test_extract_answer_accepts_direct_answer_labels() -> None:
    assert _extract_answer("A") == "A"
    assert _extract_answer("n/a") == "N/A"
    assert _extract_answer('{"answer": "B"}') == "B"
    assert _extract_answer("B.") == "B"
    assert _extract_answer("Answer: C") == "C"
    assert _extract_answer('{"final_answer": "D"}') == "D"


def test_intervention_payload_rejects_giveaway_trace_wording() -> None:
    with pytest.raises(ValueError, match="giveaway wording"):
        _parse_and_validate_intervention_payload(
            """{
              "retargeted_support_binding_claim": "The support path is retargeted to option A.",
              "retargeted_support_inference": "The evidence supports option A."
            }"""
        )

    payload = _parse_and_validate_intervention_payload(
        """{
          "retargeted_support_binding_claim": "The visual and textual evidence support option A.",
          "retargeted_support_inference": "Using the stated principle, the observed condition is most consistent with option A."
        }"""
    )

    assert payload["retargeted_support_binding_claim"].startswith("The visual")


def test_intervention_payload_sanitizer_removes_giveaway_trace_wording() -> None:
    payload, violations = _parse_and_sanitize_intervention_payload(
        """{
          "retargeted_support_binding_claim": "The support path is deterministically retargeted to answer option A: Salt Lake City.",
          "retargeted_support_bound_answer_option": "A: Salt Lake City",
          "retargeted_support_inference": "Following this retargeted support path, apply the unchanged evidence and principle to answer option A: Salt Lake City."
        }"""
    )

    assert violations
    assert payload["retargeted_support_bound_answer_option"] == "A"
    assert "retarget" not in payload["retargeted_support_binding_claim"].lower()
    assert "retarget" not in payload["retargeted_support_inference"].lower()


def test_faithfulness_trace_artifacts_are_pydantic_validated() -> None:
    result_dir = Path("results/faithfulness_evaluation_scienceqa")
    if not result_dir.exists():
        pytest.skip("faithfulness evaluation ScienceQA artifacts have not been generated")

    generated = read_jsonl(_original_traces_path(result_dir), GeneratedTraceRecord)
    intervened = read_jsonl(result_dir / "intervened_traces.jsonl", IntervenedTrace)

    assert generated
    assert intervened
    assert generated[0].trace.final_answer
    assert intervened[0].intervention.intervention_type


def test_report_loader_prefers_original_traces_and_falls_back_to_legacy(tmp_path: Path) -> None:
    preferred = tmp_path / "original_traces.jsonl"
    legacy = tmp_path / "generated_traces.jsonl"

    legacy.write_text("{}\n", encoding="utf-8")
    assert _original_traces_path(tmp_path) == legacy

    preferred.write_text("{}\n", encoding="utf-8")
    assert _original_traces_path(tmp_path) == preferred
