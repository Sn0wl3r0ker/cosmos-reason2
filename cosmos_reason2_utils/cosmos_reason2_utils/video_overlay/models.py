# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed models for the video overlay pipeline."""

from __future__ import annotations

from typing import Literal

import pydantic


class VideoInfo(pydantic.BaseModel):
    """Video metadata used for alignment."""

    model_config = pydantic.ConfigDict(extra="forbid")

    source_path: str
    fps: float
    frame_count: int
    width: int
    height: int


class SampledFrame(pydantic.BaseModel):
    """Metadata for one sampled frame."""

    model_config = pydantic.ConfigDict(extra="forbid")

    sample_id: int
    frame_idx: int
    timestamp_sec: float
    image_path: str


class SamplingManifest(pydantic.BaseModel):
    """Persisted sampling output."""

    model_config = pydantic.ConfigDict(extra="forbid")

    video: VideoInfo
    infer_fps: float
    sample_every: int
    sampled_frames: list[SampledFrame]


class InferenceResult(pydantic.BaseModel):
    """Inference output bound to sampled-frame metadata."""

    model_config = pydantic.ConfigDict(extra="forbid")

    sample_id: int
    frame_idx: int
    timestamp_sec: float
    prompt: str
    output_text: str


class InferenceManifest(pydantic.BaseModel):
    """Persisted inference results."""

    model_config = pydantic.ConfigDict(extra="forbid")

    video: VideoInfo
    infer_fps: float
    sample_every: int
    backend: Literal["mock", "online", "offline"]
    prompt: str
    sampled_metadata_path: str
    results: list[InferenceResult]


class OverlaySegment(pydantic.BaseModel):
    """Overlay time range for one sampled result."""

    model_config = pydantic.ConfigDict(extra="forbid")

    start_frame_idx: int
    end_frame_idx: int
    text: str
    sample_id: int
    frame_idx: int
    timestamp_sec: float


class OverlayManifest(pydantic.BaseModel):
    """Persisted overlay segment output."""

    model_config = pydantic.ConfigDict(extra="forbid")

    video: VideoInfo
    segments: list[OverlaySegment]
