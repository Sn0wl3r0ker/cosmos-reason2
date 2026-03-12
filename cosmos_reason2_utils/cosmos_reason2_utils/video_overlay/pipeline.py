# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-level orchestration for the video overlay pipeline."""

from __future__ import annotations

from pathlib import Path

from cosmos_reason2_utils.video_overlay.config import OverlayStyleConfig, VideoOverlayConfig
from cosmos_reason2_utils.video_overlay.io_utils import load_prompt, read_json
from cosmos_reason2_utils.video_overlay.models import InferenceManifest, OverlayManifest, SamplingManifest
from cosmos_reason2_utils.video_overlay.overlay import render_overlay_video
from cosmos_reason2_utils.video_overlay.reasoner import create_reasoner, infer_samples
from cosmos_reason2_utils.video_overlay.sampling import sample_video


def run_sample(
    *,
    input_path: str | Path,
    infer_fps: float,
    sampled_dir: str | Path,
    metadata_out: str | Path,
) -> SamplingManifest:
    """Run the sampling pass."""
    return sample_video(
        input_path=input_path,
        infer_fps=infer_fps,
        sampled_dir=sampled_dir,
        metadata_out=metadata_out,
    )


def run_infer(
    *,
    metadata_path: str | Path,
    config: VideoOverlayConfig,
    results_out: str | Path,
    debug_dir: str | Path,
    overlay_style: OverlayStyleConfig,
) -> InferenceManifest:
    """Run the inference pass."""
    sampling_manifest = read_json(metadata_path, SamplingManifest)
    prompt = load_prompt(config.prompt, config.prompt_file)
    reasoner = create_reasoner(config)
    return infer_samples(
        sampling_manifest=sampling_manifest,
        sampled_metadata_path=str(metadata_path),
        prompt=prompt,
        reasoner=reasoner,
        backend=config.backend,
        results_out=results_out,
        debug_dir=debug_dir,
        overlay_style=overlay_style,
    )


def run_overlay(
    *,
    input_path: str | Path,
    results_path: str | Path,
    output_path: str | Path,
    overlay_style: OverlayStyleConfig,
    segments_out: str | Path,
) -> OverlayManifest:
    """Run the overlay rendering pass."""
    inference_manifest = read_json(results_path, InferenceManifest)
    return render_overlay_video(
        input_path=input_path,
        inference_manifest=inference_manifest,
        output_path=output_path,
        overlay_style=overlay_style,
        segments_out=segments_out,
    )
