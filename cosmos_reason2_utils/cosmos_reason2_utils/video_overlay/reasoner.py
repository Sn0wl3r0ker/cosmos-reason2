# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference backends for the video overlay pipeline."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2

from cosmos_reason2_utils.init import init_script
from cosmos_reason2_utils.text import create_conversation, create_conversation_openai
from cosmos_reason2_utils.video_overlay.config import OverlayStyleConfig, VideoOverlayConfig
from cosmos_reason2_utils.video_overlay.draw import draw_multiline_text, render_debug_text
from cosmos_reason2_utils.video_overlay.io_utils import ensure_dir, write_json
from cosmos_reason2_utils.video_overlay.models import (
    InferenceManifest,
    InferenceResult,
    SampledFrame,
    SamplingManifest,
)
from cosmos_reason2_utils.vision import PIXELS_PER_TOKEN

_FRAME_PATTERN = re.compile(r"frame_(\d+)")
_DEFAULT_SAMPLING = {
    "max_tokens": 1024,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
    "temperature": 0.7,
    "seed": 3407,
}


class CosmosReasoner(ABC):
    """Abstract image reasoner."""

    @abstractmethod
    def infer_image(self, image_path: str, prompt: str) -> str:
        """Run image inference and return final assistant text."""


class MockCosmosReasoner(CosmosReasoner):
    """Deterministic mock backend for local testing."""

    def infer_image(self, image_path: str, prompt: str) -> str:
        match = _FRAME_PATTERN.search(Path(image_path).stem)
        frame_idx = int(match.group(1)) if match else -1
        return f"mock result for frame {frame_idx}"


class OnlineCosmosReasoner(CosmosReasoner):
    """OpenAI-compatible online backend."""

    def __init__(self, config: VideoOverlayConfig):
        init_script()
        import openai

        self._config = config
        self._client = openai.OpenAI(
            api_key="EMPTY",
            base_url=f"http://{config.host}:{config.port}/v1",
        )
        models = self._client.models.list()
        self._model_id = _select_online_model_id(config.model, models.data)

    def infer_image(self, image_path: str, prompt: str) -> str:
        messages = create_conversation_openai(
            system_prompt=self._config.system_prompt,
            user_prompt=prompt,
            images=[image_path],
        )
        completion = self._client.chat.completions.create(
            messages=messages,
            model=self._model_id,
            extra_body=_sampling_kwargs(self._config)
            | {"mm_processor_kwargs": _online_mm_kwargs(self._config)},
        )
        choices = getattr(completion, "choices", None)
        if not choices:
            raise RuntimeError("Online inference returned no choices.")
        content = choices[0].message.content or ""
        return content.strip()


class OfflineCosmosReasoner(CosmosReasoner):
    """Local offline vLLM backend."""

    def __init__(self, config: VideoOverlayConfig):
        init_script()
        import qwen_vl_utils
        import transformers
        import vllm

        self._config = config
        self._qwen_vl_utils = qwen_vl_utils
        self._processor: transformers.Qwen3VLProcessor = (
            transformers.AutoProcessor.from_pretrained(config.model)
        )
        self._sampling_params = vllm.SamplingParams(**_sampling_kwargs(config))
        self._llm = vllm.LLM(
            model=config.model,
            revision=config.revision,
            max_model_len=config.max_model_len,
            limit_mm_per_prompt={"image": 1},
        )

    def infer_image(self, image_path: str, prompt: str) -> str:
        conversation = create_conversation(
            system_prompt=self._config.system_prompt,
            user_prompt=prompt,
            images=[image_path],
            vision_kwargs=_offline_vision_kwargs(self._config),
        )
        llm_input = _build_offline_input(
            conversation=conversation,
            processor=self._processor,
            qwen_vl_utils=self._qwen_vl_utils,
        )
        outputs = self._llm.generate([llm_input], sampling_params=self._sampling_params)
        generated = outputs[0].outputs
        if not generated:
            raise RuntimeError("Offline inference returned no outputs.")
        return generated[0].text.strip()


def create_reasoner(config: VideoOverlayConfig) -> CosmosReasoner:
    """Construct the requested inference backend."""
    if config.backend == "mock":
        return MockCosmosReasoner()
    if config.backend == "online":
        return OnlineCosmosReasoner(config)
    if config.backend == "offline":
        return OfflineCosmosReasoner(config)
    raise ValueError(f"Unsupported backend: {config.backend}")


def infer_samples(
    *,
    sampling_manifest: SamplingManifest,
    sampled_metadata_path: str | Path,
    prompt: str,
    reasoner: CosmosReasoner,
    backend: str,
    results_out: str | Path | None = None,
    debug_dir: str | Path | None = None,
    overlay_style: OverlayStyleConfig | None = None,
) -> InferenceManifest:
    """Run inference for sampled frames and optionally persist artifacts."""
    results: list[InferenceResult] = []
    for sampled_frame in sampling_manifest.sampled_frames:
        output_text = reasoner.infer_image(sampled_frame.image_path, prompt)
        results.append(
            InferenceResult(
                sample_id=sampled_frame.sample_id,
                frame_idx=sampled_frame.frame_idx,
                timestamp_sec=sampled_frame.timestamp_sec,
                prompt=prompt,
                output_text=output_text,
            )
        )

    validate_results_against_samples(sampling_manifest.sampled_frames, results)
    manifest = InferenceManifest(
        video=sampling_manifest.video,
        infer_fps=sampling_manifest.infer_fps,
        sample_every=sampling_manifest.sample_every,
        backend=backend,  # type: ignore[arg-type]
        prompt=prompt,
        sampled_metadata_path=str(sampled_metadata_path),
        results=results,
    )
    if results_out is not None:
        write_json(manifest, results_out)
    if debug_dir is not None:
        if overlay_style is None:
            raise ValueError("overlay_style is required when debug_dir is set.")
        write_debug_images(
            sampled_frames=sampling_manifest.sampled_frames,
            results=results,
            debug_dir=debug_dir,
            overlay_style=overlay_style,
        )
    return manifest


def validate_results_against_samples(
    sampled_frames: list[SampledFrame],
    results: list[InferenceResult],
) -> None:
    """Validate that results are perfectly aligned with sampled frames."""
    if len(sampled_frames) != len(results):
        raise ValueError(
            f"Expected {len(sampled_frames)} results, received {len(results)}."
        )

    sampled_by_id = {item.sample_id: item for item in sampled_frames}
    if len(sampled_by_id) != len(sampled_frames):
        raise ValueError("Duplicate sample_id in sampled metadata.")

    result_ids: set[int] = set()
    for result in results:
        if result.sample_id in result_ids:
            raise ValueError(f"Duplicate result for sample_id {result.sample_id}.")
        result_ids.add(result.sample_id)
        sampled = sampled_by_id.get(result.sample_id)
        if sampled is None:
            raise ValueError(f"Result references unknown sample_id {result.sample_id}.")
        if sampled.frame_idx != result.frame_idx:
            raise ValueError(
                f"Result frame mismatch for sample_id {result.sample_id}: "
                f"{result.frame_idx} != {sampled.frame_idx}."
            )
        if not math.isclose(
            sampled.timestamp_sec, result.timestamp_sec, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"Result timestamp mismatch for sample_id {result.sample_id}: "
                f"{result.timestamp_sec} != {sampled.timestamp_sec}."
            )

    missing = sorted(sampled_by_id.keys() - result_ids)
    if missing:
        raise ValueError(f"Missing inference results for sample_ids {missing}.")


def write_debug_images(
    *,
    sampled_frames: list[SampledFrame],
    results: list[InferenceResult],
    debug_dir: str | Path,
    overlay_style: OverlayStyleConfig,
) -> None:
    """Render per-sampled-frame debug images."""
    sampled_by_id = {frame.sample_id: frame for frame in sampled_frames}
    debug_dir = ensure_dir(debug_dir)
    for result in results:
        sampled = sampled_by_id[result.sample_id]
        frame = cv2.imread(sampled.image_path)
        if frame is None:
            raise ValueError(f"Failed to read sampled image: {sampled.image_path}")
        text = render_debug_text(
            sample_id=result.sample_id,
            frame_idx=result.frame_idx,
            timestamp_sec=result.timestamp_sec,
            output_text=result.output_text,
        )
        draw_multiline_text(frame, text, overlay_style)
        debug_name = (
            f"debug_sample_{result.sample_id:06d}_frame_{result.frame_idx:06d}.jpg"
        )
        debug_path = debug_dir / debug_name
        if not cv2.imwrite(str(debug_path), frame):
            raise ValueError(f"Failed to write debug image: {debug_path}")


def _sampling_kwargs(config: VideoOverlayConfig) -> dict[str, Any]:
    kwargs = dict(_DEFAULT_SAMPLING)
    kwargs["max_tokens"] = config.max_tokens
    for key in [
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "repetition_penalty",
        "seed",
    ]:
        value = getattr(config, key)
        if value is not None:
            kwargs[key] = value
    return kwargs


def _select_online_model_id(config_model: str | None, models: list[Any]) -> str:
    """Select a model id from the online server without requiring retrieve()."""
    model_ids = [getattr(model, "id", None) for model in models]
    model_ids = [model_id for model_id in model_ids if isinstance(model_id, str)]
    if config_model:
        if config_model not in model_ids:
            raise ValueError(
                f"Model {config_model!r} was not returned by the server. "
                f"Available models: {model_ids}"
            )
        return config_model
    if not model_ids:
        raise RuntimeError("Online inference server returned no models.")
    return model_ids[0]


def _online_mm_kwargs(config: VideoOverlayConfig) -> dict[str, Any]:
    size: dict[str, int] = {}
    if config.min_pixels is not None:
        size["shortest_edge"] = config.min_pixels
    if config.total_pixels is not None:
        size["longest_edge"] = config.total_pixels
    mm_kwargs: dict[str, Any] = {"do_sample_frames": False}
    if size:
        mm_kwargs["size"] = size
    return mm_kwargs


def _offline_vision_kwargs(config: VideoOverlayConfig) -> dict[str, Any]:
    total_pixels = config.total_pixels
    if total_pixels is None:
        total_pixels = int((config.max_model_len - config.max_tokens) * PIXELS_PER_TOKEN * 0.9)
    if total_pixels <= 0:
        raise ValueError("max_model_len must be greater than max_tokens.")
    vision_kwargs: dict[str, Any] = {"total_pixels": total_pixels}
    if config.min_pixels is not None:
        vision_kwargs["min_pixels"] = config.min_pixels
    return vision_kwargs


def _build_offline_input(
    *,
    conversation: list[dict[str, Any]],
    processor: Any,
    qwen_vl_utils: Any,
) -> dict[str, Any]:
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_ids=False,
    )
    image_inputs, _, image_kwargs = qwen_vl_utils.process_vision_info(
        conversation,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if image_inputs is None:
        raise RuntimeError("Offline inference did not produce image inputs.")
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image_inputs},
        "mm_processor_kwargs": image_kwargs,
    }
