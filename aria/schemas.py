from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    DIAGRAM = "diagram"
    EXPERIMENTAL = "experimental"
    DATA = "data"
    UNKNOWN = "unknown"


class Component(StrEnum):
    VISUAL_EVIDENCE = "visual_evidence"
    TEXTUAL_EVIDENCE = "textual_evidence"
    BINDING_CLAIM = "binding_claim"
    SCIENTIFIC_PRINCIPLE = "scientific_principle"
    INFERENCE = "inference"
    CONCLUSION = "conclusion"
    FINAL_ANSWER = "final_answer"
    ASSUMPTIONS = "assumptions"
    NONE = "none"
    UNKNOWN = "unknown"


class ExpectedLabel(StrEnum):
    ORIGINAL = "original"
    INTERVENED = "intervened"


class Evidence(BaseModel):
    id: str = ""
    modality: Modality = Modality.UNKNOWN
    content: str
    provenance: str | None = None
    source_ref: str | None = None
    object_ids: list[str] = Field(default_factory=list)
    span: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_source_ref(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        payload = dict(payload)
        modality = payload.get("modality")
        if isinstance(modality, str) and "|" in modality:
            for candidate in modality.split("|"):
                candidate = candidate.strip().lower()
                if candidate in {item.value for item in Modality}:
                    payload["modality"] = candidate
                    break
        if "source_ref" not in payload and payload.get("provenance"):
            payload["source_ref"] = payload["provenance"]
        if "provenance" not in payload and payload.get("source_ref"):
            payload["provenance"] = payload["source_ref"]
        return payload

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("evidence content cannot be empty")
        return value


class SupportClaim(BaseModel):
    id: str = ""
    text: str
    supports: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def wrap_string_claim(cls, payload: Any) -> Any:
        if isinstance(payload, str):
            return {"text": payload}
        return payload

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("claim text cannot be empty")
        return value

    def __str__(self) -> str:
        return self.text

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.text == other
        return super().__eq__(other)


class BindingClaim(SupportClaim):
    uses_visual: list[str] = Field(default_factory=list)
    uses_textual: list[str] = Field(default_factory=list)
    binds_subject: str | None = None
    binds_condition: str | None = None
    binds_principle: str | None = None
    bound_answer_option: str | None = None
    relation_type: str | None = None
    referents: list[str] = Field(default_factory=list)

    @field_validator("bound_answer_option")
    @classmethod
    def normalize_bound_answer_option(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().upper()
        match = re.match(r"^([A-D])(?:\b|[\s\.\):_-])", value)
        return match.group(1) if match else value


class StructuredTrace(BaseModel):
    visual_evidence: list[Evidence] = Field(default_factory=list)
    textual_evidence: list[Evidence] = Field(default_factory=list)
    binding_claim: BindingClaim
    scientific_principle: SupportClaim
    inference: SupportClaim
    conclusion: str
    final_answer: str = "unknown"
    assumptions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_trace(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        payload = dict(payload)
        for evidence_key in ["visual_evidence", "textual_evidence", "evidence"]:
            if evidence_key in payload and isinstance(payload[evidence_key], list):
                payload[evidence_key] = [
                    item
                    for item in payload[evidence_key]
                    if not isinstance(item, dict) or str(item.get("content", "")).strip()
                ]
        if "scientific_principle" not in payload and "principle" in payload:
            payload["scientific_principle"] = payload.pop("principle")
        if "binding_claim" not in payload:
            payload["binding_claim"] = payload.get("inference") or "unknown"
        if "evidence" in payload and (
            "visual_evidence" not in payload or "textual_evidence" not in payload
        ):
            visual: list[Any] = []
            textual: list[Any] = []
            for item in payload.pop("evidence") or []:
                modality = item.get("modality") if isinstance(item, dict) else None
                if modality in {Modality.TEXT.value, Modality.UNKNOWN.value, None}:
                    textual.append(item)
                else:
                    visual.append(item)
            payload.setdefault("visual_evidence", visual)
            payload.setdefault("textual_evidence", textual)
        return payload

    @model_validator(mode="after")
    def fill_default_ids_and_supports(self) -> StructuredTrace:
        for index, evidence in enumerate(self.visual_evidence, start=1):
            if not evidence.id:
                evidence.id = f"V{index}"
        for index, evidence in enumerate(self.textual_evidence, start=1):
            if not evidence.id:
                evidence.id = f"T{index}"
        if not self.binding_claim.id:
            self.binding_claim.id = "B1"
        if not self.scientific_principle.id:
            self.scientific_principle.id = "P1"
        if not self.inference.id:
            self.inference.id = "I1"
        if not self.binding_claim.uses_visual:
            self.binding_claim.uses_visual = [item.id for item in self.visual_evidence if item.id]
        if not self.binding_claim.uses_textual:
            self.binding_claim.uses_textual = [item.id for item in self.textual_evidence if item.id]
        if not self.binding_claim.supports:
            self.binding_claim.supports = [
                *self.binding_claim.uses_visual,
                *self.binding_claim.uses_textual,
            ]
        if not self.scientific_principle.supports:
            self.scientific_principle.supports = [
                *[item.id for item in self.textual_evidence if item.id],
                *[item.id for item in self.visual_evidence if item.id],
            ]
        if not self.inference.supports:
            self.inference.supports = [self.binding_claim.id, self.scientific_principle.id]
        return self

    @field_validator("conclusion", "final_answer")
    @classmethod
    def text_fields_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("trace fields cannot be empty")
        return value

    @field_validator("final_answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        value = value.strip().upper()
        match = re.match(r"^([A-D])(?:\b|[\s\.\):_-])", value)
        if match:
            return match.group(1)
        return value if value else "UNKNOWN"

    @property
    def evidence(self) -> list[Evidence]:
        return [*self.visual_evidence, *self.textual_evidence]

    @property
    def principle(self) -> str:
        return self.scientific_principle.text

    def support_trace_dump(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            exclude={"metadata", "conclusion", "final_answer"},
        )


Trace = StructuredTrace


class Example(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    example_id: str
    dataset: str = "scienceqa"
    question: str
    choices: dict[str, str] = Field(default_factory=dict)
    correct_answer: str | None = Field(default=None, validation_alias=AliasChoices("correct_answer", "gold_answer"))
    context: str | None = None
    lecture: str | None = None
    explanation: str | None = None
    image_path: str | None = None
    image_metadata: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("correct_answer")
    @classmethod
    def normalize_correct_answer(cls, value: str | None) -> str | None:
        return value.strip().upper() if value else None


class InterventionMetadata(BaseModel):
    intervention_type: str
    target_component: Component
    expected_label: ExpectedLabel
    expected_flawed_component: Component = Component.UNKNOWN
    known_location: bool = True
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_intervention_type(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        payload = dict(payload)
        legacy_paraphrase = "paraphrase_" + "preserve_semantics"
        if payload.get("intervention_type") == legacy_paraphrase:
            payload["intervention_type"] = "paraphrase"
        details = payload.get("details")
        if isinstance(details, dict) and details.get("condition") == legacy_paraphrase:
            payload["details"] = {**details, "condition": "paraphrase"}
        return payload

    @field_validator("expected_label", mode="before")
    @classmethod
    def normalize_expected_label(cls, value: str | ExpectedLabel) -> str | ExpectedLabel:
        return ExpectedLabel.ORIGINAL if value == "clean" else value

    @field_validator("target_component", "expected_flawed_component", mode="before")
    @classmethod
    def normalize_legacy_component(cls, value: str | Component) -> str | Component:
        if value == "evidence":
            return Component.VISUAL_EVIDENCE
        if value == "principle":
            return Component.SCIENTIFIC_PRINCIPLE
        return value


class IntervenedTrace(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    intervention_id: str
    example_id: str
    original_trace: StructuredTrace = Field(alias="clean_trace")
    intervened_trace: StructuredTrace
    intervention: InterventionMetadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def label_is_intervened(self) -> bool:
        return self.intervention.expected_label == ExpectedLabel.INTERVENED

    @property
    def expected_flawed_component(self) -> Component:
        return self.intervention.expected_flawed_component


class GeneratedTraceRecord(BaseModel):
    example_id: str
    trace: StructuredTrace
    target_model: str = "unknown"
    raw_output: str | None = None
    parse_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InterventionModelOutput(BaseModel):
    example_id: str
    intervention_model: str = "unknown"
    interventions: list[IntervenedTrace] = Field(default_factory=list)
    raw_output: str | None = None
    parse_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeOutput(BaseModel):
    intervention_id: str
    judge_model: str
    faithfulness_score: int = Field(ge=1, le=5)
    is_intervened_or_flawed: bool
    flawed_component: Component = Component.UNKNOWN
    missing_step: bool = False
    final_answer_supported: bool = True
    short_reason: str = ""
    raw_output: str | None = None
    parse_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("short_reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        return value.strip()


class TargetOutput(BaseModel):
    intervention_id: str
    target_model: str
    final_answer: str
    is_correct: bool | None = None
    confidence: float | None = None
    logit_margin: float | None = None
    raw_output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("final_answer")
    @classmethod
    def normalize_final_answer(cls, value: str) -> str:
        value = value.strip().upper()
        if value in {"N/A", "NA", "N.A.", "UNKNOWN"}:
            return "N/A" if value != "UNKNOWN" else "UNKNOWN"
        if re.search(r"\bN\s*/?\s*A\b", value):
            return "N/A"
        match = re.match(r"^([A-D])(?:\b|[\s\.\):_-])", value)
        if match:
            return match.group(1)
        return value if value else "UNKNOWN"

    @model_validator(mode="after")
    def validate_confidence(self) -> TargetOutput:
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return self


class InterventionValidationRow(BaseModel):
    example_id: str
    intervention_id: str
    intervention_type: str
    operation: str
    changed_field: str
    old_value: str
    new_value: str
    unchanged_fields: str
    is_flawed: bool
    auto_validation_pass: bool
    validation_notes: str = ""
    human_validation_status: str = ""
