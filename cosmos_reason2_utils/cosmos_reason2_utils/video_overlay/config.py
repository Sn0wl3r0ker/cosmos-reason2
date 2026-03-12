# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the video overlay pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pydantic
import yaml

DEFAULT_PROMPT = """AMR perception task.
Describe visible objects in the forward scene.
Output concise structured text.
No extra explanation."""

DEFAULT_DATA_ROOT = Path("cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/")
DEFAULT_SAMPLED_DIR = DEFAULT_DATA_ROOT / "sampled_frames"
DEFAULT_DEBUG_DIR = DEFAULT_DATA_ROOT / "debug_frames"
DEFAULT_RESULTS_DIR = DEFAULT_DATA_ROOT / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "output"


class OverlayStyleConfig(pydantic.BaseModel):
    """Overlay drawing configuration."""

    model_config = pydantic.ConfigDict(extra="forbid")

    font_scale: float = 0.7
    line_spacing: int = 28
    margin_x: int = 20
    margin_y: int = 30
    max_width_ratio: float = 0.45
    box_alpha: float = 0.65
    thickness: int = 1


class VideoOverlayConfig(pydantic.BaseModel):
    """Default configuration for the CLI."""

    model_config = pydantic.ConfigDict(extra="forbid")

    input: str | None = None
    infer_fps: float = 2.0
    prompt: str = DEFAULT_PROMPT
    prompt_file: str | None = None

    sampled_dir: str = str(DEFAULT_SAMPLED_DIR)
    debug_dir: str = str(DEFAULT_DEBUG_DIR)
    metadata_out: str = str(DEFAULT_RESULTS_DIR / "sampled_metadata.json")
    metadata: str | None = None
    results_out: str = str(DEFAULT_RESULTS_DIR / "inference_results.json")
    results: str | None = None
    segments_out: str = str(DEFAULT_RESULTS_DIR / "overlay_segments.json")
    output: str = str(DEFAULT_OUTPUT_DIR / "overlay.mp4")

    backend: Literal["mock", "online", "offline"] = "mock"
    system_prompt: str = "You are a helpful assistant."
    model: str = "nvidia/Cosmos-Reason2-2B"
    revision: str | None = None
    host: str = "localhost"
    port: int = 8000
    max_model_len: int = 16384
    min_pixels: int | None = None
    total_pixels: int | None = None
    max_tokens: int = 1024
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None

    font_scale: float = 0.7
    line_spacing: int = 28
    margin_x: int = 20
    margin_y: int = 30
    max_width_ratio: float = 0.45
    box_alpha: float = 0.65
    thickness: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "VideoOverlayConfig":
        """Load configuration defaults from YAML."""
        if path is None:
            return cls()
        with open(path, "rb") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)

    def overlay_style(self) -> OverlayStyleConfig:
        """Create a drawing config from the flat settings model."""
        return OverlayStyleConfig(
            font_scale=self.font_scale,
            line_spacing=self.line_spacing,
            margin_x=self.margin_x,
            margin_y=self.margin_y,
            max_width_ratio=self.max_width_ratio,
            box_alpha=self.box_alpha,
            thickness=self.thickness,
        )
