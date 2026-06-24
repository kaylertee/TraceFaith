from __future__ import annotations

import gc
import json
import os
import re
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from aria.parsing import parse_intervention_payload, parse_judge_output, parse_structured_trace
from aria.prompts import (
    render_intervention_generation_prompt,
    render_judge_prompt,
    render_trace_repair_prompt,
    render_target_rerun_prompt,
    render_trace_generation_prompt,
)
from aria.schemas import Example, IntervenedTrace, InterventionModelOutput, JudgeOutput, StructuredTrace, TargetOutput


@dataclass
class HFGenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False


DEFAULT_TARGET_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_JUDGE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_INTERVENTION_MODEL_ID = DEFAULT_JUDGE_MODEL_ID
FORBIDDEN_GENERATED_TRACE_TERMS = (
    r"\bwrong\b",
    r"\bcorrupt\w*\b",
    r"\bflawed?\b",
    r"\bmisappl\w*\b",
    r"\bretarget\w*\b",
    r"\bdistractor\b",
    r"\bintervention\b",
    r"\bperturb\w*\b",
)


class TraceGeneratorModel:
    """Hugging Face runner for structured trace generation."""

    def __init__(
        self,
        model_id: str = DEFAULT_TARGET_MODEL_ID,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = True,
        quantization: str = "auto",
        awq_backend: str | None = None,
        require_gpu: bool = True,
        min_visual_tokens: int = 256,
        max_visual_tokens: int = 768,
        generation: HFGenerationConfig | None = None,
    ) -> None:
        self.model_name = model_id
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.quantization = quantization
        self.awq_backend = awq_backend
        self.require_gpu = require_gpu
        self.min_visual_tokens = min_visual_tokens
        self.max_visual_tokens = max_visual_tokens
        self.generation = generation or HFGenerationConfig(max_new_tokens=512)
        self._model: Any | None = None
        self._processor: Any | None = None

    def generate_trace(self, example: Example) -> StructuredTrace:
        model, processor = self._load()
        prompt = render_trace_generation_prompt(example)
        output_text = self._generate_vl_text(prompt, example.image_path)
        repair_output_text = None
        parse_error = None
        try:
            trace = parse_structured_trace(output_text)
        except Exception as exc:
            parse_error = str(exc)
            repair_prompt = render_trace_repair_prompt(example, output_text, parse_error)
            repair_output_text = self._generate_vl_text(
                repair_prompt,
                example.image_path,
                HFGenerationConfig(
                    max_new_tokens=max(self.generation.max_new_tokens, 1024),
                    temperature=self.generation.temperature,
                    do_sample=self.generation.do_sample,
                ),
            )
            try:
                trace = parse_structured_trace(repair_output_text)
            except Exception as repair_exc:
                strict_repair_prompt = (
                    f"{repair_prompt}\n\nThe previous repair was still invalid because:\n{repair_exc}\n"
                    "Regenerate one complete JSON object now. Include every required top-level field from the schema. "
                    "Use concise unknown placeholders where content is missing. Return no Markdown or prose."
                )
                repair_output_text = self._generate_vl_text(
                    strict_repair_prompt,
                    example.image_path,
                    HFGenerationConfig(
                        max_new_tokens=max(self.generation.max_new_tokens, 2048),
                        temperature=self.generation.temperature,
                        do_sample=self.generation.do_sample,
                    ),
                )
                trace = parse_structured_trace(repair_output_text)
        trace.metadata["generated_by"] = self.model_name
        trace.metadata["raw_output"] = output_text
        if parse_error is not None:
            trace.metadata["initial_parse_error"] = parse_error
            trace.metadata["repair_raw_output"] = repair_output_text
        return trace

    def _generate_vl_text(
        self,
        prompt: str,
        image_path: str | None,
        generation: HFGenerationConfig | None = None,
    ) -> str:
        model, processor = self._load()
        messages = [_vision_language_user_message(prompt, image_path)]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._build_inputs(processor, text, messages)
        return _generate_text(model, processor, inputs, generation or self.generation)

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "TraceGeneratorModel requires transformers with vision-language model support."
            ) from exc

        load_kwargs = _model_load_kwargs(
            self.model_id,
            self.torch_dtype,
            self.load_in_4bit,
            self.quantization,
            awq_backend=self.awq_backend,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=_device_map_for_load(self.device_map, self.require_gpu),
            **load_kwargs,
        )
        _align_awq_vl_output_head_dtype(self._model, self.model_id)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=self.min_visual_tokens * 28 * 28,
            max_pixels=self.max_visual_tokens * 28 * 28,
        )
        return self._model, self._processor

    def unload(self) -> None:
        self._model = None
        self._processor = None
        _clear_cuda_cache()

    def _build_inputs(self, processor: Any, text: str, messages: list[dict[str, Any]]) -> Any:
        images = None
        videos = None
        if _message_has_image(messages):
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as exc:
                raise ImportError(
                    "Image examples require qwen-vl-utils. Install qwen-vl-utils or run text-only examples."
                ) from exc
            images, videos = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        return _move_inputs_to_model(inputs, self._model)


class JudgeModel:
    """Hugging Face runner for text-only judge evaluation."""

    def __init__(
        self,
        model_id: str = DEFAULT_JUDGE_MODEL_ID,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = True,
        quantization: str = "auto",
        awq_backend: str | None = None,
        require_gpu: bool = True,
        generation: HFGenerationConfig | None = None,
    ) -> None:
        self.model_name = model_id
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.quantization = quantization
        self.awq_backend = awq_backend
        self.require_gpu = require_gpu
        self.generation = generation or HFGenerationConfig(max_new_tokens=192)
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def judge(self, intervention_id: str, trace: StructuredTrace, example: Example) -> JudgeOutput:
        model, tokenizer = self._load()
        prompt = render_judge_prompt(
            IntervenedTrace(
                intervention_id=intervention_id,
                example_id=example.example_id,
                original_trace=trace,
                intervened_trace=trace,
                intervention={
                    "intervention_type": "unknown_to_judge",
                    "target_component": "unknown",
                    "expected_label": "intervened",
                    "expected_flawed_component": "unknown",
                    "known_location": False,
                },
            ),
            example,
        )
        output_text = _generate_chat_text(model, tokenizer, prompt, self.generation)
        return parse_judge_output(output_text, intervention_id=intervention_id, judge_model=self.model_name)

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("JudgeModel requires transformers.") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=_device_map_for_load(self.device_map, self.require_gpu),
            **_model_load_kwargs(
                self.model_id,
                self.torch_dtype,
                self.load_in_4bit,
                self.quantization,
                awq_backend=self.awq_backend,
            ),
        )
        return self._model, self._tokenizer

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        _clear_cuda_cache()


class InterventionModel(JudgeModel):
    """Hugging Face text runner that creates natural-language intervened traces."""

    def generate_interventions(
        self,
        example: Example,
        original_trace: StructuredTrace,
        template_records: list[IntervenedTrace],
    ) -> InterventionModelOutput:
        model, tokenizer = self._load()
        specs = [
            {
                "intervention_id": record.intervention_id,
                "intervention_type": record.intervention.intervention_type,
                "target_component": record.intervention.target_component.value,
                "expected_flawed_component": record.intervention.expected_flawed_component.value,
                "details": record.intervention.details,
            }
            for record in template_records
            if record.intervention.intervention_type not in {"original_trace", "delete_binding_claim"}
        ]
        prompt = render_intervention_generation_prompt(example, original_trace, specs)
        output_text = _generate_chat_text(model, tokenizer, prompt, self.generation)
        try:
            payload, violations = _parse_and_sanitize_intervention_payload(output_text)
            return InterventionModelOutput(
                example_id=example.example_id,
                intervention_model=self.model_name,
                raw_output=json.dumps(payload, indent=2),
                metadata={"requested_intervention_count": len(specs), "sanitized_generated_text": bool(violations)},
            )
        except Exception as first_exc:
            retry_generation = HFGenerationConfig(
                max_new_tokens=max(self.generation.max_new_tokens * 2, 1024),
                temperature=self.generation.temperature,
                do_sample=self.generation.do_sample,
            )
            retry_prompt = (
                f"{prompt}\n\nThe previous output was invalid because generated trace text revealed that it was "
                "an intervention or corruption. Regenerate only the flat JSON object. Keep all generated trace text "
                "neutral and ordinary; do not use words like wrong, corrupted, flawed, misapplied, retargeted, "
                "distractor, intervention, or perturbation in any value."
            )
            retry_output_text = _generate_chat_text(model, tokenizer, retry_prompt, retry_generation)
            try:
                payload, violations = _parse_and_sanitize_intervention_payload(retry_output_text)
                return InterventionModelOutput(
                    example_id=example.example_id,
                    intervention_model=self.model_name,
                    raw_output=json.dumps(payload, indent=2),
                    metadata={
                        "requested_intervention_count": len(specs),
                        "initial_parse_error": str(first_exc),
                        "retry_max_new_tokens": retry_generation.max_new_tokens,
                        "sanitized_generated_text": bool(violations),
                    },
                )
            except Exception as exc:
                combined_error = (
                    f"{exc}; initial parse error before retry: {first_exc}; "
                    f"retry_max_new_tokens={retry_generation.max_new_tokens}"
                )
            return InterventionModelOutput(
                example_id=example.example_id,
                intervention_model=self.model_name,
                raw_output=retry_output_text,
                parse_error=combined_error,
                metadata={
                    "requested_intervention_count": len(specs),
                    "initial_raw_output": output_text,
                    "initial_parse_error": str(first_exc),
                    "retry_max_new_tokens": retry_generation.max_new_tokens,
                },
            )


def _parse_and_validate_intervention_payload(text: str) -> dict[str, Any]:
    payload, violations = _parse_and_sanitize_intervention_payload(text)
    if violations:
        formatted = ", ".join(f"{key}={term}" for key, term in violations[:5])
        raise ValueError(f"Generated trace text contains giveaway wording: {formatted}")
    return payload


def _parse_and_sanitize_intervention_payload(text: str) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    payload = parse_intervention_payload(text)
    violations = _generated_trace_text_violations(payload)
    sanitized = deepcopy(payload)
    _sanitize_generated_component_strings_in_place(sanitized)
    return sanitized, violations


def _generated_trace_text_violations(payload: dict[str, Any]) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    for key, value in _iter_generated_component_strings(payload):
        lowered = value.lower()
        for pattern in FORBIDDEN_GENERATED_TRACE_TERMS:
            if re.search(pattern, lowered):
                violations.append((key, pattern))
                break
    return violations


def _iter_generated_component_strings(payload: dict[str, Any]) -> list[tuple[str, str]]:
    generated: list[tuple[str, str]] = []
    interventions = payload.get("interventions")
    if isinstance(interventions, list):
        for index, item in enumerate(interventions):
            if not isinstance(item, dict):
                continue
            for key in (
                "binding_claim_text",
                "scientific_principle_text",
                "principle_text",
                "inference_text",
            ):
                value = item.get(key)
                if isinstance(value, str):
                    generated.append((f"interventions[{index}].{key}", value))
            patch = item.get("component_patch") or item.get("intervened_trace") or item.get("patch")
            if isinstance(patch, dict):
                generated.extend(_iter_patch_text_strings(patch, prefix=f"interventions[{index}]"))
        return generated

    for key, value in payload.items():
        if key.endswith(("_binding_claim", "_scientific_principle", "_principle", "_inference")) and isinstance(value, str):
            generated.append((key, value))
    return generated


def _iter_patch_text_strings(patch: dict[str, Any], prefix: str) -> list[tuple[str, str]]:
    generated: list[tuple[str, str]] = []
    for key, value in patch.items():
        current = f"{prefix}.{key}"
        if isinstance(value, dict):
            generated.extend(_iter_patch_text_strings(value, current))
        elif key == "text" and isinstance(value, str):
            generated.append((current, value))
    return generated


def _sanitize_generated_component_strings_in_place(payload: dict[str, Any]) -> None:
    interventions = payload.get("interventions")
    if isinstance(interventions, list):
        for item in interventions:
            if not isinstance(item, dict):
                continue
            for key in (
                "binding_claim_text",
                "scientific_principle_text",
                "principle_text",
                "inference_text",
            ):
                if isinstance(item.get(key), str):
                    item[key] = _sanitize_generated_trace_text(item[key])
            for key in ("bound_answer_option", "binding_answer_option"):
                if isinstance(item.get(key), str):
                    item[key] = _normalize_answer_label_text(item[key])
            patch = item.get("component_patch") or item.get("intervened_trace") or item.get("patch")
            if isinstance(patch, dict):
                _sanitize_patch_text_in_place(patch)
        return

    for key, value in list(payload.items()):
        if key.endswith(("_binding_claim", "_scientific_principle", "_principle", "_inference")) and isinstance(value, str):
            payload[key] = _sanitize_generated_trace_text(value)
        elif key.endswith("_bound_answer_option") and isinstance(value, str):
            payload[key] = _normalize_answer_label_text(value)


def _sanitize_patch_text_in_place(patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict):
            _sanitize_patch_text_in_place(value)
        elif key == "text" and isinstance(value, str):
            patch[key] = _sanitize_generated_trace_text(value)
        elif key == "bound_answer_option" and isinstance(value, str):
            patch[key] = _normalize_answer_label_text(value)


def _sanitize_generated_trace_text(text: str) -> str:
    replacements = [
        (r"\bis\s+deterministically\s+retargeted\s+to\b", "supports"),
        (r"\bis\s+retargeted\s+to\b", "supports"),
        (r"\bdeterministically\s+retargeted\s+to\b", "supports"),
        (r"\bretargeted\s+to\b", "supports"),
        (r"\bretargeted\s+support\s+path\b", "support path"),
        (r"\binstead\s+bound\s+to\b", "supports"),
        (r"\bmisappl\w*\b", "applied"),
        (r"\bdistractor\b", "answer option"),
        (r"\bwrong\b", "alternative"),
        (r"\bcorrupt\w*\b", "changed"),
        (r"\bflawed?\b", "changed"),
        (r"\bintervention\b", "reasoning"),
        (r"\bperturb\w*\b", "changed"),
        (r"\bretarget\w*\b", "support"),
        (r",?\s*while the evidence and principle are kept unchanged\.?", "."),
        (r"\s+", " "),
    ]
    sanitized = text.strip()
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized.strip()


def _normalize_answer_label_text(text: str) -> str:
    match = re.match(r"\s*([A-Z])\b", text.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else text.strip()


class TargetRerunModel(TraceGeneratorModel):
    """Re-run the target model with an intervened trace and ask for the final answer only."""

    def __init__(
        self,
        *args: Any,
        generation: HFGenerationConfig | None = None,
        trace_generation: HFGenerationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.target_generation = generation or HFGenerationConfig(max_new_tokens=16)
        super().__init__(*args, generation=trace_generation or HFGenerationConfig(max_new_tokens=512), **kwargs)

    def evaluate(
        self,
        intervention_id: str,
        trace: StructuredTrace | None,
        example: Example,
        condition: str = "original_trace",
        include_image: bool = True,
    ) -> TargetOutput:
        model, processor = self._load()
        prompt = render_target_rerun_prompt(trace, example, condition=condition)
        messages = [_vision_language_user_message(prompt, example.image_path if include_image else None)]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._build_inputs(processor, text, messages)
        output_text = _generate_text(model, processor, inputs, self.target_generation)
        final_answer = _extract_answer(output_text)
        is_correct = final_answer == example.correct_answer if example.correct_answer else None
        return TargetOutput(
            intervention_id=intervention_id,
            target_model=self.model_name,
            final_answer=final_answer,
            is_correct=is_correct,
            raw_output=output_text,
            metadata={"condition": condition, "include_image": include_image},
        )


def _vision_language_user_message(prompt: str, image_path: str | None) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if image_path:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def _message_has_image(messages: list[dict[str, Any]]) -> bool:
    return any(item.get("type") == "image" for message in messages for item in message.get("content", []))


def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = None
    return inputs.to(device) if device is not None else inputs


def _device_map_for_load(device_map: str, require_gpu: bool) -> Any:
    if not require_gpu:
        return device_map
    try:
        import torch
    except ImportError as exc:
        raise ImportError("GPU-required loading needs torch installed.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU-required loading was requested, but CUDA is not available. "
            "Pass --allow-cpu-offload to permit CPU/RAM fallback."
        )
    if device_map == "auto":
        return {"": 0}
    return device_map


def _model_load_kwargs(
    model_id: str,
    torch_dtype: str,
    load_in_4bit: bool,
    quantization: str = "auto",
    awq_backend: str | None = None,
) -> dict[str, Any]:
    quantization = quantization.lower()
    if quantization not in {"auto", "bnb4", "prequantized", "none"}:
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    if quantization == "prequantized" or (quantization == "auto" and _is_prequantized_model(model_id)):
        kwargs: dict[str, Any] = {"torch_dtype": _torch_dtype_for_load(torch_dtype, model_id)}
        if "-awq" in model_id.lower() and "vl" in model_id.lower():
            kwargs["config"] = _awq_vl_config_override(model_id, awq_backend)
        elif awq_backend and "-awq" in model_id.lower():
            _ensure_active_env_bin_on_path()
            from transformers import AwqConfig
            from transformers.utils.quantization_config import AwqBackend

            kwargs["quantization_config"] = AwqConfig(
                backend=AwqBackend(awq_backend),
            )
        return kwargs
    if quantization == "none" or not load_in_4bit:
        return {"torch_dtype": _torch_dtype_for_load(torch_dtype, model_id)}
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError("4-bit model loading requires torch, transformers, and bitsandbytes.") from exc
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    }


def _is_prequantized_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(marker in lowered for marker in ["-awq", "gptq", "int4", "int3", "fp8"])


def _torch_dtype_for_load(torch_dtype: str, model_id: str) -> Any:
    lowered = torch_dtype.lower()
    if lowered == "auto":
        if "-awq" in model_id.lower() and "vl" in model_id.lower():
            import torch

            return torch.float16
        return "auto"
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if lowered not in mapping:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return mapping[lowered]


def _ensure_active_env_bin_on_path() -> None:
    env_bin = os.path.dirname(sys.executable)
    env_root = os.path.dirname(env_bin)
    path = os.environ.get("PATH", "")
    entries = path.split(os.pathsep) if path else []
    if env_bin not in entries:
        os.environ["PATH"] = os.pathsep.join([env_bin, *entries])
    if not os.environ.get("CUDA_HOME"):
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc_path))
        elif os.path.exists(os.path.join(env_bin, "nvcc")):
            os.environ["CUDA_HOME"] = env_root


def _awq_vl_config_override(model_id: str, awq_backend: str | None) -> Any:
    _ensure_active_env_bin_on_path()
    import torch
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    config.torch_dtype = torch.float16
    if hasattr(config, "text_config"):
        config.text_config.torch_dtype = torch.float16
    if hasattr(config, "vision_config"):
        config.vision_config.torch_dtype = torch.float16
    quantization_config = getattr(config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        quantization_config["modules_to_not_convert"] = [
            "visual",
            "model.visual",
            "visual.*",
            "model.visual.*",
        ]
        if awq_backend and awq_backend != "auto":
            quantization_config["version"] = awq_backend
    return config


def _align_awq_vl_output_head_dtype(model: Any, model_id: str) -> None:
    if "-awq" not in model_id.lower() or "vl" not in model_id.lower():
        return
    try:
        import torch
    except ImportError:
        return
    if hasattr(model, "lm_head") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model.lm_head.to(dtype=torch.bfloat16)


def _clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _generate_text(model: Any, processor: Any, inputs: Any, generation: HFGenerationConfig) -> str:
    input_length = inputs.input_ids.shape[-1]
    kwargs = {
        "max_new_tokens": generation.max_new_tokens,
        "do_sample": generation.do_sample,
    }
    if generation.do_sample:
        kwargs["temperature"] = generation.temperature
    generated = model.generate(**inputs, **kwargs)
    generated = generated[:, input_length:]
    return processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _generate_chat_text(model: Any, tokenizer: Any, prompt: str, generation: HFGenerationConfig) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[-1]
    kwargs = {
        "max_new_tokens": generation.max_new_tokens,
        "do_sample": generation.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if generation.do_sample:
        kwargs["temperature"] = generation.temperature
    generated = model.generate(**inputs, **kwargs)
    generated = generated[:, input_length:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def _extract_answer(text: str) -> str:
    upper = text.strip().upper()
    if upper in {"N/A", "NA", "N.A.", "UNKNOWN"}:
        return "N/A" if upper != "UNKNOWN" else "UNKNOWN"
    if re.search(r"\bN\s*/?\s*A\b", upper):
        return "N/A"
    try:
        import json

        payload = json.loads(text.strip())
        if isinstance(payload, dict):
            for key in ["answer", "final_answer", "target_answer"]:
                value = payload.get(key)
                if isinstance(value, str):
                    return _extract_answer(value)
    except Exception:
        pass
    if upper == "UNKNOWN":
        return "UNKNOWN"
    for label in ["A", "B", "C", "D"]:
        if (
            f'"{label}"' in upper
            or re.search(rf"\bANSWER\s*[:\-]?\s*{label}\b", upper)
            or re.fullmatch(rf"{label}[\.\)]?", upper)
        ):
            return label
    match = re.search(r"\b([A-D])\b", upper)
    if match:
        return match.group(1)
    return "UNKNOWN"
