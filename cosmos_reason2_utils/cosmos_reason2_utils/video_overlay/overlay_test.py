# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cosmos_reason2_utils.video_overlay.models import InferenceResult
from cosmos_reason2_utils.video_overlay.overlay import build_overlay_segments


def _result(sample_id: int, frame_idx: int) -> InferenceResult:
    return InferenceResult(
        sample_id=sample_id,
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
        prompt="Describe the scene.",
        output_text=f"mock result for frame {frame_idx}",
    )


def test_build_overlay_segments_multiple_samples():
    segments = build_overlay_segments([_result(0, 45), _result(1, 60)], last_video_frame_idx=89)
    assert [(segment.start_frame_idx, segment.end_frame_idx) for segment in segments] == [
        (45, 59),
        (60, 89),
    ]


def test_build_overlay_segments_last_segment_extends_to_video_end():
    segments = build_overlay_segments([_result(0, 30), _result(1, 45)], last_video_frame_idx=47)
    assert segments[-1].end_frame_idx == 47


def test_build_overlay_segments_single_sample_covers_full_tail():
    segments = build_overlay_segments([_result(0, 0)], last_video_frame_idx=29)
    assert [(segments[0].start_frame_idx, segments[0].end_frame_idx)] == [(0, 29)]
