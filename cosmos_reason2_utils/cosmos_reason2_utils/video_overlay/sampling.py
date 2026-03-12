# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame sampling for video overlay inference."""

from __future__ import annotations

from pathlib import Path

import cv2

from cosmos_reason2_utils.video_overlay.io_utils import ensure_dir, write_json
from cosmos_reason2_utils.video_overlay.models import SampledFrame, SamplingManifest, VideoInfo


def calculate_sample_every(video_fps: float, infer_fps: float) -> int:
    """Calculate the frame interval for sampling."""
    if video_fps <= 0:
        raise ValueError(f"Video FPS must be positive, got {video_fps}.")
    if infer_fps <= 0:
        raise ValueError(f"Inference FPS must be positive, got {infer_fps}.")
    return max(1, round(video_fps / infer_fps))


def timestamp_from_frame(frame_idx: int, video_fps: float) -> float:
    """Convert a frame index to seconds."""
    if frame_idx < 0:
        raise ValueError(f"Frame index must be non-negative, got {frame_idx}.")
    if video_fps <= 0:
        raise ValueError(f"Video FPS must be positive, got {video_fps}.")
    return frame_idx / video_fps


def sampled_frame_name(sample_id: int, frame_idx: int) -> str:
    """Create a deterministic filename for sampled frames."""
    return f"sample_{sample_id:06d}_frame_{frame_idx:06d}.jpg"


def read_video_info(input_path: str | Path) -> VideoInfo:
    """Read basic metadata from a video file."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    if fps <= 0:
        raise ValueError(f"Video reported invalid FPS: {fps}")
    if frame_count <= 0:
        raise ValueError(f"Video reported invalid frame count: {frame_count}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Video reported invalid dimensions: {width}x{height}")
    return VideoInfo(
        source_path=str(Path(input_path)),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def sample_video(
    *,
    input_path: str | Path,
    infer_fps: float,
    sampled_dir: str | Path,
    metadata_out: str | Path | None = None,
) -> SamplingManifest:
    """Sample frames from a video and optionally persist metadata."""
    video = read_video_info(input_path)
    sample_every = calculate_sample_every(video.fps, infer_fps)
    sampled_dir = ensure_dir(sampled_dir)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")

    sampled_frames: list[SampledFrame] = []
    frame_idx = 0
    sample_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every == 0:
                image_path = sampled_dir / sampled_frame_name(sample_id, frame_idx)
                if not cv2.imwrite(str(image_path), frame):
                    raise ValueError(f"Failed to write sampled frame: {image_path}")
                sampled_frames.append(
                    SampledFrame(
                        sample_id=sample_id,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_from_frame(frame_idx, video.fps),
                        image_path=str(image_path),
                    )
                )
                sample_id += 1
            frame_idx += 1
    finally:
        cap.release()

    manifest = SamplingManifest(
        video=video,
        infer_fps=infer_fps,
        sample_every=sample_every,
        sampled_frames=sampled_frames,
    )
    if metadata_out is not None:
        write_json(manifest, metadata_out)
    return manifest
