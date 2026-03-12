# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cosmos_reason2_utils.video_overlay.sampling import (
    calculate_sample_every,
    timestamp_from_frame,
)


def test_calculate_sample_every_normal_case():
    assert calculate_sample_every(30.0, 2.0) == 15


def test_calculate_sample_every_low_infer_fps():
    assert calculate_sample_every(29.97, 1.0) == 30


def test_calculate_sample_every_infer_fps_higher_than_video_fps():
    assert calculate_sample_every(2.0, 5.0) == 1


def test_timestamp_from_frame():
    assert timestamp_from_frame(45, 30.0) == 1.5
