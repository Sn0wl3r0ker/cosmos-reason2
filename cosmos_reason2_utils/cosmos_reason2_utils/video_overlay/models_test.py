# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from cosmos_reason2_utils.video_overlay.io_utils import read_json, write_json
from cosmos_reason2_utils.video_overlay.models import (
    InferenceManifest,
    InferenceResult,
    SampledFrame,
    SamplingManifest,
    VideoInfo,
)
from cosmos_reason2_utils.video_overlay.reasoner import validate_results_against_samples


def _video_info() -> VideoInfo:
    return VideoInfo(
        source_path="input.mp4",
        fps=30.0,
        frame_count=90,
        width=320,
        height=240,
    )


def _sampled_frames() -> list[SampledFrame]:
    return [
        SampledFrame(sample_id=0, frame_idx=0, timestamp_sec=0.0, image_path="0.jpg"),
        SampledFrame(sample_id=1, frame_idx=15, timestamp_sec=0.5, image_path="1.jpg"),
    ]


def test_sampling_manifest_json_round_trip(tmp_path: Path):
    manifest = SamplingManifest(
        video=_video_info(),
        infer_fps=2.0,
        sample_every=15,
        sampled_frames=_sampled_frames(),
    )
    path = tmp_path / "sampled.json"
    write_json(manifest, path)
    loaded = read_json(path, SamplingManifest)
    assert loaded == manifest


def test_inference_manifest_json_round_trip(tmp_path: Path):
    manifest = InferenceManifest(
        video=_video_info(),
        infer_fps=2.0,
        sample_every=15,
        backend="mock",
        prompt="Describe the scene.",
        sampled_metadata_path="sampled.json",
        results=[
            InferenceResult(
                sample_id=0,
                frame_idx=0,
                timestamp_sec=0.0,
                prompt="Describe the scene.",
                output_text="mock result for frame 0",
            )
        ],
    )
    path = tmp_path / "results.json"
    write_json(manifest, path)
    loaded = read_json(path, InferenceManifest)
    assert loaded == manifest


def test_validate_results_against_samples_detects_missing_result():
    with pytest.raises(ValueError, match="Expected 2 results, received 1"):
        validate_results_against_samples(
            _sampled_frames(),
            [
                InferenceResult(
                    sample_id=0,
                    frame_idx=0,
                    timestamp_sec=0.0,
                    prompt="Describe the scene.",
                    output_text="ok",
                )
            ],
        )


def test_validate_results_against_samples_detects_mismatch():
    with pytest.raises(ValueError, match="Result frame mismatch"):
        validate_results_against_samples(
            _sampled_frames(),
            [
                InferenceResult(
                    sample_id=0,
                    frame_idx=1,
                    timestamp_sec=0.0,
                    prompt="Describe the scene.",
                    output_text="ok",
                ),
                InferenceResult(
                    sample_id=1,
                    frame_idx=15,
                    timestamp_sec=0.5,
                    prompt="Describe the scene.",
                    output_text="ok",
                ),
            ],
        )
