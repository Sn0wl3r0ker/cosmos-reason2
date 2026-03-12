# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Overlay segment generation and video rendering."""

from __future__ import annotations

from pathlib import Path

import cv2

from cosmos_reason2_utils.video_overlay.config import OverlayStyleConfig
from cosmos_reason2_utils.video_overlay.draw import draw_multiline_text, render_debug_text
from cosmos_reason2_utils.video_overlay.io_utils import write_json
from cosmos_reason2_utils.video_overlay.models import (
    InferenceManifest,
    InferenceResult,
    OverlayManifest,
    OverlaySegment,
    VideoInfo,
)
from cosmos_reason2_utils.video_overlay.sampling import read_video_info


def build_overlay_segments(
    results: list[InferenceResult],
    *,
    last_video_frame_idx: int,
) -> list[OverlaySegment]:
    """Convert per-sample results into frame-interval overlay segments."""
    if last_video_frame_idx < 0:
        raise ValueError("last_video_frame_idx must be non-negative.")
    ordered = sorted(results, key=lambda item: item.frame_idx)
    if not ordered:
        return []
    segments: list[OverlaySegment] = []
    for index, result in enumerate(ordered):
        if index < len(ordered) - 1:
            end_frame_idx = ordered[index + 1].frame_idx - 1
        else:
            end_frame_idx = last_video_frame_idx
        if end_frame_idx < result.frame_idx:
            raise ValueError(
                f"Invalid overlay segment for sample_id {result.sample_id}: "
                f"{result.frame_idx}..{end_frame_idx}."
            )
        segments.append(
            OverlaySegment(
                start_frame_idx=result.frame_idx,
                end_frame_idx=end_frame_idx,
                text=render_debug_text(
                    sample_id=result.sample_id,
                    frame_idx=result.frame_idx,
                    timestamp_sec=result.timestamp_sec,
                    output_text=result.output_text,
                ),
                sample_id=result.sample_id,
                frame_idx=result.frame_idx,
                timestamp_sec=result.timestamp_sec,
            )
        )
    return segments


def validate_video_matches_manifest(
    input_video: VideoInfo,
    manifest_video: VideoInfo,
) -> None:
    """Ensure the input video matches the metadata embedded in the results."""
    if input_video.frame_count != manifest_video.frame_count:
        raise ValueError("Input video frame count does not match inference metadata.")
    if input_video.width != manifest_video.width or input_video.height != manifest_video.height:
        raise ValueError("Input video dimensions do not match inference metadata.")
    if abs(input_video.fps - manifest_video.fps) > 1e-6:
        raise ValueError("Input video FPS does not match inference metadata.")


def render_overlay_video(
    *,
    input_path: str | Path,
    inference_manifest: InferenceManifest,
    output_path: str | Path,
    overlay_style: OverlayStyleConfig,
    segments_out: str | Path | None = None,
) -> OverlayManifest:
    """Render overlay text onto the original video."""
    input_video = read_video_info(input_path)
    validate_video_matches_manifest(input_video, inference_manifest.video)

    segments = build_overlay_segments(
        inference_manifest.results,
        last_video_frame_idx=input_video.frame_count - 1,
    )
    overlay_manifest = OverlayManifest(video=input_video, segments=segments)
    if segments_out is not None:
        write_json(overlay_manifest, segments_out)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")
    writer = _create_video_writer(output_path, input_video)

    segment_index = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            while segment_index < len(segments) and frame_idx > segments[segment_index].end_frame_idx:
                segment_index += 1
            if segment_index < len(segments):
                segment = segments[segment_index]
                if segment.start_frame_idx <= frame_idx <= segment.end_frame_idx:
                    draw_multiline_text(frame, segment.text, overlay_style)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    return overlay_manifest


def _create_video_writer(output_path: str | Path, video: VideoInfo) -> cv2.VideoWriter:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ["mp4v", "avc1"]:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            video.fps,
            (video.width, video.height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Failed to open a video writer for {output_path}.")
