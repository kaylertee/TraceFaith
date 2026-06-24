from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from aria.schemas import Component, JudgeOutput, StructuredTrace


INTERVENTION_FLAT_KEYS = {
    "paraphrase_binding_claim",
    "paraphrase_scientific_principle",
    "paraphrase_principle",
    "paraphrase_inference",
    "wrong_binding_binding_claim",
    "wrong_binding_bound_answer_option",
    "retargeted_support_binding_claim",
    "retargeted_support_bound_answer_option",
    "retargeted_support_inference",
    "wrong_inference_inference",
    "wrong_principle_scientific_principle",
    "wrong_principle_principle",
}


def parse_structured_trace(text: str) -> StructuredTrace:
    """Parse model-declared structured JSON without semantic reinterpretation."""
    payload = _loads_json_object(text)
    return StructuredTrace.model_validate(payload)


def parse_judge_output(text: str, intervention_id: str, judge_model: str) -> JudgeOutput:
    try:
        payload = _loads_json_object(text)
        _normalize_judge_payload(payload)
        payload.setdefault("intervention_id", intervention_id)
        payload.setdefault("judge_model", judge_model)
        payload.setdefault("raw_output", text)
        return JudgeOutput.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        return JudgeOutput(
            intervention_id=intervention_id,
            judge_model=judge_model,
            faithfulness_score=1,
            is_intervened_or_flawed=True,
            flawed_component="unknown",
            missing_step=True,
            final_answer_supported=False,
            short_reason="Judge output could not be parsed.",
            raw_output=text,
            parse_error=str(exc),
        )


def parse_intervention_payload(text: str) -> dict[str, Any]:
    payload = _loads_json_value(text)
    if isinstance(payload, list):
        payload = {"interventions": payload}
    if not isinstance(payload, dict):
        raise ValueError("Expected intervention payload to be a JSON object or intervention list")
    interventions = payload.get("interventions")
    if interventions is None and any(key in payload for key in INTERVENTION_FLAT_KEYS):
        return payload
    if not isinstance(interventions, list):
        raise ValueError("Expected intervention payload to contain an interventions list")
    return payload


def _loads_json_object(text: str) -> dict[str, Any]:
    payload = _loads_json_value(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def _loads_json_value(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        payload = _json_loads_lenient(text)
    except json.JSONDecodeError:
        extracted = _extract_first_json_value(text)
        if extracted is None:
            raise
        payload = _json_loads_lenient(extracted)
    return payload


def _json_loads_lenient(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        sanitized = _escape_invalid_json_backslashes(text)
        if sanitized == text:
            raise first_error
        return json.loads(sanitized)


def _escape_invalid_json_backslashes(text: str) -> str:
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _extract_first_json_value(text: str) -> str | None:
    starts = [index for index in (text.find("{"), text.find("[")) if index != -1]
    if not starts:
        return None
    start = min(starts)
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start : index + 1]
    return None


def _normalize_judge_payload(payload: dict[str, Any]) -> None:
    score = payload.get("faithfulness_score")
    if isinstance(score, float):
        payload["faithfulness_score"] = round(score)
    if isinstance(score, int | float):
        payload["faithfulness_score"] = min(5, max(1, int(round(score))))

    component = payload.get("flawed_component")
    if isinstance(component, str):
        component = component.strip().lower()
        if component == "evidence":
            component = "visual_evidence"
        elif component == "principle":
            component = "scientific_principle"
        payload["flawed_component"] = component if component in {item.value for item in Component} else "unknown"
