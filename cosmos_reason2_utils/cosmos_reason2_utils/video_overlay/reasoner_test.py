# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cosmos_reason2_utils.video_overlay.reasoner import _select_online_model_id


class _Model:
    def __init__(self, model_id: str):
        self.id = model_id


def test_select_online_model_id_uses_requested_model():
    models = [_Model("nvidia/Cosmos-Reason2-2B"), _Model("other-model")]
    assert _select_online_model_id("nvidia/Cosmos-Reason2-2B", models) == "nvidia/Cosmos-Reason2-2B"


def test_select_online_model_id_defaults_to_first_available():
    models = [_Model("nvidia/Cosmos-Reason2-2B"), _Model("other-model")]
    assert _select_online_model_id(None, models) == "nvidia/Cosmos-Reason2-2B"


def test_select_online_model_id_rejects_missing_requested_model():
    models = [_Model("other-model")]
    with pytest.raises(ValueError, match="Available models"):
        _select_online_model_id("nvidia/Cosmos-Reason2-2B", models)
