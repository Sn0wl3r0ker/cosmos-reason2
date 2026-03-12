# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cv2
import numpy as np
import pytest

from cosmos_reason2_utils.video_overlay.config import VideoOverlayConfig
from cosmos_reason2_utils.video_overlay.pipeline import run_infer, run_overlay, run_sample


def test_end_to_end_mock_pipeline(tmp_path: Path):
    input_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (160, 120),
    )
    if not writer.isOpened():
        pytest.skip("MP4 video writer codec is unavailable in this environment.")

    for index in range(20):
        frame = np.full((120, 160, 3), index * 10, dtype=np.uint8)
        cv2.putText(
            frame,
            f"frame {index}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()

    sampled_dir = tmp_path / "sampled"
    debug_dir = tmp_path / "debug"
    metadata_path = tmp_path / "sampled_metadata.json"
    results_path = tmp_path / "inference_results.json"
    segments_path = tmp_path / "overlay_segments.json"
    output_path = tmp_path / "overlay.mp4"

    sampling_manifest = run_sample(
        input_path=input_path,
        infer_fps=2.0,
        sampled_dir=sampled_dir,
        metadata_out=metadata_path,
    )
    config = VideoOverlayConfig(
        backend="mock",
        prompt="Describe the scene.",
        debug_dir=str(debug_dir),
        results_out=str(results_path),
        metadata_out=str(metadata_path),
        output=str(output_path),
        segments_out=str(segments_path),
    )
    inference_manifest = run_infer(
        metadata_path=metadata_path,
        config=config,
        results_out=results_path,
        debug_dir=debug_dir,
        overlay_style=config.overlay_style(),
    )
    overlay_manifest = run_overlay(
        input_path=input_path,
        results_path=results_path,
        output_path=output_path,
        overlay_style=config.overlay_style(),
        segments_out=segments_path,
    )

    assert metadata_path.exists()
    assert results_path.exists()
    assert segments_path.exists()
    assert output_path.exists()
    assert len(list(sampled_dir.glob("*.jpg"))) == len(sampling_manifest.sampled_frames)
    assert len(list(debug_dir.glob("*.jpg"))) == len(inference_manifest.results)
    assert len(overlay_manifest.segments) == len(inference_manifest.results)
