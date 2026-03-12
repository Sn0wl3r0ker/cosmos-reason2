# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video overlay pipeline for Cosmos-Reason2."""

from cosmos_reason2_utils.video_overlay.config import (
    DEFAULT_PROMPT,
    OverlayStyleConfig,
    VideoOverlayConfig,
)
from cosmos_reason2_utils.video_overlay.models import (
    InferenceManifest,
    InferenceResult,
    OverlayManifest,
    OverlaySegment,
    SampledFrame,
    SamplingManifest,
    VideoInfo,
)

__all__ = [
    "DEFAULT_PROMPT",
    "InferenceManifest",
    "InferenceResult",
    "OverlayManifest",
    "OverlaySegment",
    "OverlayStyleConfig",
    "SampledFrame",
    "SamplingManifest",
    "VideoInfo",
    "VideoOverlayConfig",
]
