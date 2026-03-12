# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Overlay drawing helpers."""

from __future__ import annotations

import textwrap

import cv2
import numpy as np

from cosmos_reason2_utils.video_overlay.config import OverlayStyleConfig

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX


def wrap_text(
    text: str,
    *,
    max_width: int,
    font_scale: float,
    thickness: int,
) -> list[str]:
    """Wrap text to fit within a maximum pixel width."""
    wrapped_lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph.strip():
            wrapped_lines.append("")
            continue
        words = paragraph.split()
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            text_width = cv2.getTextSize(
                candidate, FONT_FACE, font_scale, thickness
            )[0][0]
            if text_width <= max_width:
                current = candidate
            else:
                wrapped_lines.append(current)
                current = word
        wrapped_lines.append(current)
    return wrapped_lines


def draw_multiline_text(
    frame: np.ndarray,
    text: str,
    style: OverlayStyleConfig,
) -> np.ndarray:
    """Draw wrapped overlay text in the top-left corner."""
    max_width = max(100, int(frame.shape[1] * style.max_width_ratio))
    lines = wrap_text(
        text,
        max_width=max_width,
        font_scale=style.font_scale,
        thickness=style.thickness,
    )
    if not lines:
        return frame

    sizes = [
        cv2.getTextSize(line or " ", FONT_FACE, style.font_scale, style.thickness)[0]
        for line in lines
    ]
    line_height = max(style.line_spacing, max(height for _, height in sizes) + 8)
    box_width = max(width for width, _ in sizes) + 24
    box_height = line_height * len(lines) + 16

    x0 = style.margin_x
    y0 = style.margin_y
    x1 = min(frame.shape[1] - style.margin_x, x0 + box_width)
    y1 = min(frame.shape[0] - style.margin_y, y0 + box_height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, style.box_alpha, frame, 1.0 - style.box_alpha, 0, frame)

    baseline_y = y0 + line_height
    for i, line in enumerate(lines):
        text_y = baseline_y + i * line_height
        cv2.putText(
            frame,
            line,
            (x0 + 12, text_y),
            FONT_FACE,
            style.font_scale,
            (255, 255, 255),
            thickness=style.thickness,
            lineType=cv2.LINE_AA,
        )
    return frame


def render_debug_text(
    *,
    sample_id: int,
    frame_idx: int,
    timestamp_sec: float,
    output_text: str,
) -> str:
    """Build the text shown in debug and overlay frames."""
    body = output_text.strip() or "<empty>"
    return textwrap.dedent(
        f"""\
        sample_id: {sample_id}
        frame_idx: {frame_idx}
        time_sec: {timestamp_sec:.2f}
        {body}
        """
    ).strip()
